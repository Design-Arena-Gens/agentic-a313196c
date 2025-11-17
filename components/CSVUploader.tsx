"use client";

import Papa from "papaparse";
import { useMemo, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

type Row = {
  id?: string | number;
  raw_text?: string;
  clean_text?: string;
  emotion_score?: string | number;
  topic_id?: string | number;
  topic_keywords?: string;
};

export default function CSVUploader() {
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File) {
    setError(null);
    Papa.parse<Row>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors?.length) {
          setError(result.errors[0]?.message || "Parsing error");
        }
        setRows(result.data);
      },
      error: (err) => setError(err.message),
    });
  }

  const topicCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of rows) {
      const key = String(r.topic_id ?? "unknown");
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .sort((a, b) => Number(a[0]) - Number(b[0]));
  }, [rows]);

  const emotionStats = useMemo(() => {
    const values = rows
      .map((r) => Number(r.emotion_score))
      .filter((n) => Number.isFinite(n));
    if (!values.length) return { avg: 0, min: 0, max: 0 };
    const sum = values.reduce((a, b) => a + b, 0);
    return { avg: sum / values.length, min: Math.min(...values), max: Math.max(...values) };
  }, [rows]);

  return (
    <div className="card" style={{ display: "grid", gap: 16 }}>
      <h2 style={{ margin: 0 }}>Upload results CSV</h2>
      <input
        className="input"
        type="file"
        accept=".csv,text/csv"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {error && <div style={{ color: "#ff6b6b" }}>{error}</div>}
      {!!rows.length && (
        <div className="grid" style={{ gridTemplateColumns: "1fr" }}>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Dataset</h3>
            <div style={{ fontSize: 14, opacity: 0.8 }}>
              {rows.length} dreams | avg sentiment {emotionStats.avg.toFixed(3)} (min {emotionStats.min.toFixed(3)}, max {emotionStats.max.toFixed(3)})
            </div>
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Topic distribution</h3>
            <Bar
              data={{
                labels: topicCounts.map((t) => t[0]),
                datasets: [
                  {
                    label: "Count",
                    data: topicCounts.map((t) => t[1]),
                    backgroundColor: "rgba(30, 144, 255, 0.6)",
                  },
                ],
              }}
              options={{
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } },
              }}
            />
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>Preview (first 10)</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #1e2a44" }}>id</th>
                    <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #1e2a44" }}>emotion_score</th>
                    <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #1e2a44" }}>topic_id</th>
                    <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #1e2a44" }}>topic_keywords</th>
                    <th style={{ textAlign: "left", padding: 8, borderBottom: "1px solid #1e2a44" }}>clean_text</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 10).map((r, i) => (
                    <tr key={i}>
                      <td style={{ padding: 8, borderBottom: "1px solid #1e2a44" }}>{r.id ?? i}</td>
                      <td style={{ padding: 8, borderBottom: "1px solid #1e2a44" }}>{r.emotion_score}</td>
                      <td style={{ padding: 8, borderBottom: "1px solid #1e2a44" }}>{r.topic_id}</td>
                      <td style={{ padding: 8, borderBottom: "1px solid #1e2a44" }}>{r.topic_keywords}</td>
                      <td style={{ padding: 8, borderBottom: "1px solid #1e2a44", maxWidth: 560, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.clean_text}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
