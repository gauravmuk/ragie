// gemini-embeddings.js — Gemini embeddings with outputDimensionality support
import { Embeddings } from "@langchain/core/embeddings";

const DEFAULT_MODEL = "gemini-embedding-001";
const DEFAULT_DIMENSIONS = 1536;
const API_BASE = "https://generativelanguage.googleapis.com/v1beta";

export class GeminiEmbeddings extends Embeddings {
  constructor(fields = {}) {
    super(fields);
    this.apiKey = fields.apiKey || process.env.GEMINI_API_KEY;
    this.model = fields.model || DEFAULT_MODEL;
    this.dimensions = fields.dimensions || DEFAULT_DIMENSIONS;
    if (!this.apiKey) {
      throw new Error("GEMINI_API_KEY is required");
    }
  }

  async _embed(texts) {
    const requests = texts.map((text) => ({
      model: `models/${this.model}`,
      content: { parts: [{ text }] },
      outputDimensionality: this.dimensions,
    }));

    const res = await fetch(
      `${API_BASE}/models/${this.model}:batchEmbedContents?key=${this.apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requests }),
      }
    );

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Gemini embedding API error ${res.status}: ${body}`);
    }

    const json = await res.json();
    return json.embeddings.map((e) => e.values);
  }

  async embedDocuments(documents) {
    // Batch in chunks of 100 (API limit)
    const results = [];
    for (let i = 0; i < documents.length; i += 100) {
      const batch = documents.slice(i, i + 100);
      const embeddings = await this.caller.call(this._embed.bind(this), batch);
      results.push(...embeddings);
    }
    return results;
  }

  async embedQuery(document) {
    const [embedding] = await this.caller.call(this._embed.bind(this), [document]);
    return embedding;
  }
}
