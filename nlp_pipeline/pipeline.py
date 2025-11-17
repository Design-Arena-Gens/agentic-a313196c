#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
from spacy.lang.en import English

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


@dataclass
class DreamRecord:
    id: str
    raw_text: str
    clean_text: str
    emotion_score: float
    topic_id: int
    topic_keywords: str


def ensure_nltk_data() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> None:
    try:
        spacy.load(model_name)
    except Exception:
        try:
            from spacy.cli import download  # type: ignore

            download(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to ensure spaCy model '{model_name}'. Please install manually: python -m spacy download {model_name}"
            ) from e


def collect_from_reddit(limit: int = 500, user_agent: str = "dreams-nlp/1.0") -> List[str]:
    """
    Collect dream-like posts from r/Dreams via public JSON. No auth, best-effort.
    Falls back gracefully if blocked.
    """
    collected: List[str] = []
    after = None
    per_page = min(100, max(1, limit))
    headers = {"User-Agent": user_agent}

    while len(collected) < limit:
        url = f"https://www.reddit.com/r/Dreams/top.json?limit={per_page}&t=year"
        if after:
            url += f"&after={after}"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                logging.warning("Reddit request failed: %s", r.status_code)
                break
            data = r.json()
            children = data.get("data", {}).get("children", [])
            if not children:
                break
            for ch in children:
                post = ch.get("data", {})
                title = post.get("title") or ""
                selftext = post.get("selftext") or ""
                text = (title + ". " + selftext).strip()
                if len(text.split()) >= 5:
                    collected.append(text)
                if len(collected) >= limit:
                    break
            after = data.get("data", {}).get("after")
            if not after:
                break
        except Exception as e:
            logging.warning("Reddit fetch exception: %s", e)
            break
    return collected


def collect_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def clean_texts(texts: Iterable[str], nlp, stop_words: set) -> List[List[str]]:
    cleaned: List[List[str]] = []
    # basic normalize
    pattern = re.compile(r"[^a-z\s]")
    for doc in nlp.pipe((t.lower() for t in texts), batch_size=64):
        text = doc.text
        text = pattern.sub(" ", text)
        tokens: List[str] = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            lemma = tok.lemma_.strip().lower()
            if not lemma or lemma in stop_words or len(lemma) < 2:
                continue
            if lemma.isnumeric():
                continue
            tokens.append(lemma)
        cleaned.append(tokens)
    return cleaned


def compute_sentiment(texts: List[str]) -> List[float]:
    sia = SentimentIntensityAnalyzer()
    return [float(sia.polarity_scores(t).get("compound", 0.0)) for t in texts]


def train_lda_and_choose_k(
    tokenized: List[List[str]], k_min: int = 5, k_max: int = 15, random_state: int = 42
) -> Tuple[LdaModel, corpora.Dictionary, List[List[Tuple[int, int]]]]:
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(toks) for toks in tokenized]

    best_model = None
    best_k = None
    best_coh = -1.0

    for k in range(k_min, k_max + 1):
        if k <= 1:
            continue
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=random_state,
            passes=5,
            iterations=200,
            chunksize=2000,
        )
        cm = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary, coherence="c_v")
        coh = float(cm.get_coherence())
        logging.info("k=%d coherence=%.4f", k, coh)
        if coh > best_coh:
            best_coh = coh
            best_model = model
            best_k = k

    assert best_model is not None and best_k is not None
    logging.info("Best topics: k=%d coherence=%.4f", best_k, best_coh)

    # rebuild corpus for return
    corpus = [dictionary.doc2bow(toks) for toks in tokenized]
    return best_model, dictionary, corpus


def topic_for_doc(model: LdaModel, bow: List[Tuple[int, int]]) -> Tuple[int, float]:
    dist = model.get_document_topics(bow, minimum_probability=0.0)
    if not dist:
        return 0, 0.0
    topic_id, prob = max(dist, key=lambda x: x[1])
    return int(topic_id), float(prob)


def top_keywords_for_topic(model: LdaModel, topic_id: int, topn: int = 8) -> str:
    terms = model.show_topic(topic_id, topn=topn)
    return ", ".join([w for w, _ in terms])


def run_pipeline(
    out_csv: str,
    source: str = "reddit",
    limit: int = 500,
    sample_file: str = "nlp_pipeline/sample_dreams.txt",
    k_min: int = 5,
    k_max: int = 15,
) -> None:
    logging.info("Collecting dreams: source=%s limit=%d", source, limit)
    texts: List[str] = []
    if source == "reddit":
        texts = collect_from_reddit(limit=limit)
        if len(texts) < max(50, int(0.6 * limit)):
            logging.warning("Falling back to local sample due to limited fetch (%d)", len(texts))
            texts.extend(collect_from_file(sample_file))
    elif source == "file":
        texts = collect_from_file(sample_file)
    else:
        raise ValueError("Unsupported source. Use 'reddit' or 'file'.")

    # keep first `limit` if too many
    texts = texts[:limit]
    logging.info("Collected %d dreams", len(texts))

    ensure_nltk_data()
    ensure_spacy_model()

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    stop_words = set(stopwords.words("english"))

    logging.info("Cleaning & lemmatizing texts")
    tokenized = clean_texts(texts, nlp, stop_words)

    logging.info("Computing sentiment")
    sentiments = compute_sentiment(texts)

    logging.info("Training LDA and choosing k")
    model, dictionary, corpus = train_lda_and_choose_k(tokenized, k_min=k_min, k_max=k_max)

    logging.info("Assigning dominant topics & exporting CSV")
    records: List[DreamRecord] = []
    for i, (raw, toks, bow, emo) in enumerate(zip(texts, tokenized, corpus, sentiments)):
        t_id, _ = topic_for_doc(model, bow)
        kw = top_keywords_for_topic(model, t_id)
        records.append(
            DreamRecord(
                id=str(i + 1),
                raw_text=raw,
                clean_text=" ".join(toks),
                emotion_score=float(emo),
                topic_id=int(t_id),
                topic_keywords=kw,
            )
        )

    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "raw_text",
                "clean_text",
                "emotion_score",
                "topic_id",
                "topic_keywords",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r.id,
                "raw_text": r.raw_text,
                "clean_text": r.clean_text,
                "emotion_score": f"{r.emotion_score:.6f}",
                "topic_id": r.topic_id,
                "topic_keywords": r.topic_keywords,
            })

    logging.info("Saved %d rows to %s", len(records), out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreams NLP pipeline: collect, preprocess, sentiment, LDA")
    parser.add_argument("--output", default="outputs/dreams_results.csv", help="Path to output CSV")
    parser.add_argument("--source", choices=["reddit", "file"], default="reddit", help="Data source")
    parser.add_argument("--limit", type=int, default=500, help="Number of dreams to attempt to collect")
    parser.add_argument("--k_min", type=int, default=5, help="Min topics for search")
    parser.add_argument("--k_max", type=int, default=15, help="Max topics for search")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run_pipeline(out_csv=args.output, source=args.source, limit=args.limit, k_min=args.k_min, k_max=args.k_max)
