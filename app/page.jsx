"use client";

import { useState, useRef, useCallback, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";

function getOrCreateId(key) {
  if (typeof window === "undefined") return null;
  const stored = localStorage.getItem(key);
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

function getOrCreateSessionId() {
  if (typeof window === "undefined") return null;
  const key = "justcall-help:sessionId";
  const tsKey = "justcall-help:sessionTs";
  const SESSION_TTL_MS = 30 * 60 * 1000; // 30 minutes

  const stored = localStorage.getItem(key);
  const ts = Number(localStorage.getItem(tsKey) || 0);

  if (stored && Date.now() - ts < SESSION_TTL_MS) {
    localStorage.setItem(tsKey, String(Date.now()));
    return stored;
  }

  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  localStorage.setItem(tsKey, String(Date.now()));
  return id;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const userId = useMemo(() => getOrCreateId("justcall-help:userId"), []);

  const isSafeUrl = useCallback((url) => {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "https:" || parsed.protocol === "http:";
    } catch {
      return false;
    }
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const sessionId = getOrCreateSessionId();
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          userId,
          sessionId,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Something went wrong");
      }

      if (data.noContext) {
        setAnswer("No matching context found. Try rephrasing your question.");
      } else {
        setAnswer(data.answer);
      }
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background flex items-start justify-center px-4 pt-16 pb-8">
      <div className="w-full max-w-2xl space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">JustCall Help</h1>
          <p className="text-muted-foreground">
            Ask anything about JustCall — powered by RAG
          </p>
        </div>

        <Card>
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <Input
                ref={inputRef}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="How do I set up call forwarding?"
                disabled={loading}
              />
              <Button type="submit" disabled={loading || !question.trim()}>
                {loading ? "Searching..." : "Ask"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <p className="text-sm text-destructive">{error}</p>
            </CardContent>
          </Card>
        )}

        {answer && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Answer</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none whitespace-pre-wrap text-sm leading-relaxed">
                {answer}
              </div>
            </CardContent>
          </Card>
        )}

        {sources.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Sources</CardTitle>
              <CardDescription>Top retrieved documents</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {sources.map((src, i) => (
                  <li key={i} className="text-sm">
                    <span className="font-medium">{src.title}</span>
                    {src.source && isSafeUrl(src.source) && (
                      <a
                        href={src.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-2 text-muted-foreground underline hover:text-foreground"
                      >
                        link
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
