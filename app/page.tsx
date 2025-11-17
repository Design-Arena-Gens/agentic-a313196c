import CSVUploader from "@/components/CSVUploader";

export default function Page() {
  return (
    <main className="grid" style={{ gap: 24 }}>
      <div className="card">
        <h1 style={{ marginTop: 0 }}>Dromen Analyse ? Sentiment & Onderwerpen</h1>
        <p>
          1) Voer de Python pipeline uit in Google Colab om dromen te verzamelen, pre-processen, sentiment te berekenen en onderwerpen te vinden. Dit produceert een CSV-bestand.
        </p>
        <p>
          2) Upload hieronder de CSV om een snelle verkenning te zien (topicverdeling + voorbeeld).
        </p>
      </div>
      <CSVUploader />
      <div className="card">
        <h3 style={{ marginTop: 0 }}>CSV kolommen</h3>
        <ul>
          <li><b>id</b>: unieke id</li>
          <li><b>raw_text</b>: originele droom</li>
          <li><b>clean_text</b>: schoongemaakte/lemmatized tekst</li>
          <li><b>emotion_score</b>: VADER compound-score [-1, 1]</li>
          <li><b>topic_id</b>: gekozen onderwerp id</li>
          <li><b>topic_keywords</b>: top termen van het onderwerp</li>
        </ul>
      </div>
    </main>
  );
}
