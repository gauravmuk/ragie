import { Langfuse } from "langfuse";
import { CallbackHandler } from "@langfuse/langchain";

let langfuseInstance = null;

function isConfigured() {
  return !!(
    process.env.LANGFUSE_SECRET_KEY &&
    process.env.LANGFUSE_PUBLIC_KEY
  );
}

export function getLangfuse() {
  if (!isConfigured()) return null;
  if (!langfuseInstance) {
    langfuseInstance = new Langfuse({
      secretKey: process.env.LANGFUSE_SECRET_KEY,
      publicKey: process.env.LANGFUSE_PUBLIC_KEY,
      baseUrl: process.env.LANGFUSE_BASE_URL,
    });
  }
  return langfuseInstance;
}

export function createTraceHandler({ name, userId, sessionId, tags = [], metadata = {} } = {}) {
  const langfuse = getLangfuse();
  if (!langfuse) return { handler: null, trace: null };

  const trace = langfuse.trace({
    name: name ?? "rag-query",
    userId,
    sessionId,
    tags,
    metadata,
  });

  const handler = new CallbackHandler({ root: trace, updateRoot: true });
  return { handler, trace };
}

export function createSpanHandler(trace, { name }) {
  if (!trace) return null;
  const span = trace.span({ name });
  return new CallbackHandler({ root: span });
}

export async function fetchPrompt(name, { fallback } = {}) {
  const langfuse = getLangfuse();
  if (!langfuse) return null;
  try {
    return await langfuse.getPrompt(name, undefined, {
      label: "production",
      cacheTtlSeconds: 300,
    });
  } catch (err) {
    console.warn(`Langfuse: failed to fetch prompt "${name}", using fallback.`, err.message);
    return null;
  }
}

export async function flushLangfuse() {
  const langfuse = getLangfuse();
  if (langfuse) {
    try {
      await langfuse.flushAsync();
    } catch {
      // Tracing must never break application behavior
    }
  }
}
