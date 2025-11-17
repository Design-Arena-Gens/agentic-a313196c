import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dreams NLP Explorer",
  description: "Collect, preprocess, analyze dreams (sentiment + topics)",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div style={{ maxWidth: 960, margin: "0 auto", padding: "24px" }}>
          {children}
        </div>
      </body>
    </html>
  );
}
