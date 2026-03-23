import "./globals.css";

export const metadata = {
  title: "JustCall Help - RAG Search",
  description: "Ask questions about JustCall and get AI-powered answers",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
