if (!process.env.VERCEL) {
  await import("dotenv/config");
}
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { GeminiEmbeddings } from "./gemini-embeddings.js";
import { Document } from "@langchain/core/documents";
import { makeSupabaseClient } from "./supabase-client.js";
import { createRateLimiter } from "./rate-limiter.js";
import { existsSync, readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import facetConfigJson from "./facet-config.json" with { type: "json" };
import { createTraceHandler, createSpanHandler, fetchPrompt, flushLangfuse } from "./langfuse-client.js";

const TABLE_NAME = "documents";
const QUERY_NAME = "match_documents";
const CHAT_PROVIDER = process.env.CHAT_PROVIDER || "groq";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
const GEMINI_CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || "gemini-2.0-flash";
const CHAT_MODEL = CHAT_PROVIDER === "groq" ? GROQ_MODEL : GEMINI_CHAT_MODEL;

function makeChatLlm({ model, temperature = 0 } = {}) {
  const m = model || CHAT_MODEL;
  if (CHAT_PROVIDER === "groq") {
    return new ChatGroq({
      apiKey: process.env.GROQ_API_KEY,
      model: m,
      temperature,
    });
  }
  return new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model: m,
    temperature,
  });
}
const TOP_K = Number(process.env.RAG_TOP_K || 5);
const CANDIDATE_K = Number(process.env.RAG_CANDIDATE_K || Math.max(TOP_K * 4, 20));
const PARENT_CONTEXT_LIMIT = Number(process.env.RAG_PARENT_CONTEXT_LIMIT || 3);
const LEXICAL_PER_TERM = Number(process.env.RAG_LEXICAL_PER_TERM || 8);
const BM25_K1 = Number(process.env.RAG_BM25_K1 || 1.2);
const BM25_B = Number(process.env.RAG_BM25_B || 0.75);
const SEMANTIC_WEIGHT = Number(process.env.RAG_SEMANTIC_WEIGHT || 0.6);
const BM25_WEIGHT = Number(process.env.RAG_BM25_WEIGHT || 0.4);
const INTENT_SIGNAL_WEIGHT = Number(process.env.RAG_INTENT_SIGNAL_WEIGHT || 0.15);
const MAX_SIGNAL_PENALTY = Number(process.env.RAG_MAX_SIGNAL_PENALTY || 0.18);
const MAX_SIGNAL_BONUS = Number(process.env.RAG_MAX_SIGNAL_BONUS || 0.1);
const FACET_ROUTING_WEIGHT = Number(process.env.RAG_FACET_ROUTING_WEIGHT || 0.45);
const FACET_CONFIG_PATH =
  process.env.RAG_FACET_CONFIG
  || join(dirname(fileURLToPath(import.meta.url)), "facet-config.json");

function getQuestionFromArgs() {
  const question = process.argv.slice(2).join(" ").trim();
  if (question) return question;
  console.error('Usage: node query.js "your question"');
  process.exit(1);
}

function formatContext(docs, parentDocs = []) {
  const parentBlock = parentDocs
    .map((doc, i) => {
      const source = doc.metadata?.source ?? "unknown";
      const section = doc.metadata?.section ?? "Untitled section";
      return `[P${i + 1}] ${section}\nSource: ${source}\n${doc.pageContent}`;
    })
    .join("\n\n---\n\n");

  const childBlock = docs
    .map((doc, i) => {
      const source = doc.metadata?.source ?? "unknown";
      const title = doc.metadata?.title ?? "Untitled";
      const section = doc.metadata?.section ? `\nSection: ${doc.metadata.section}` : "";
      return `[#${i + 1}] ${title}\nSource: ${source}${section}\n${doc.pageContent}`;
    })
    .join("\n\n---\n\n");

  if (!parentBlock) return childBlock;
  return `Parent context:\n${parentBlock}\n\n====================\n\nTop child chunks:\n${childBlock}`;
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

const STOPWORDS = new Set([
  "the", "and", "for", "that", "with", "from", "this", "have", "what", "how",
  "can", "you", "are", "was", "were", "when", "where", "which", "into", "about",
  "your", "will", "then", "than", "them", "they", "their", "there", "justcall",
  "help", "docs", "article", "guide",
]);

function extractQueryTerms(question) {
  const stopwords = STOPWORDS;
  const unique = [];
  for (const token of bm25Tokenize(question)) {
    if (!stopwords.has(token) && !unique.includes(token)) unique.push(token);
  }
  return unique.slice(0, 12);
}

export function loadFacetConfig() {
  let raw;
  if (process.env.RAG_FACET_CONFIG) {
    const customPath = process.env.RAG_FACET_CONFIG;
    if (!existsSync(customPath)) {
      throw new Error(`Facet config not found: ${customPath}`);
    }
    raw = JSON.parse(readFileSync(customPath, "utf-8"));
  } else {
    raw = facetConfigJson;
  }
  if (!raw.defaultFacet || !Array.isArray(raw.facets) || raw.facets.length === 0) {
    throw new Error("facet-config.json must define defaultFacet and a non-empty facets[]");
  }
  const ids = new Set(raw.facets.map((f) => f.id));
  if (!ids.has(raw.defaultFacet)) {
    throw new Error(`defaultFacet "${raw.defaultFacet}" must exist in facets[].id`);
  }
  return raw;
}

function getValidFacetIds(facetConfig) {
  return new Set(facetConfig.facets.map((f) => f.id));
}

function inferDocFacet(doc, facetConfig) {
  const valid = getValidFacetIds(facetConfig);
  const meta = doc.metadata?.facet;
  if (meta && valid.has(meta)) return meta;

  const source = (doc.metadata?.source ?? "").toLowerCase();
  const title = (doc.metadata?.title ?? "").toLowerCase();

  for (const f of facetConfig.facets) {
    const m = f.docMatch;
    if (!m) continue;
    for (const sub of m.sourceContains || []) {
      if (source.includes(String(sub).toLowerCase())) return f.id;
    }
    for (const sub of m.titleContains || []) {
      if (title.includes(String(sub).toLowerCase())) return f.id;
    }
  }
  return facetConfig.defaultFacet;
}

function detectQueryFacetFromConfig(question, facetConfig) {
  const raw = question.toLowerCase();
  for (const f of facetConfig.facets) {
    const hints = f.queryHints || [];
    for (const h of hints) {
      if (raw.includes(String(h).toLowerCase())) return f.id;
    }
  }
  return facetConfig.defaultFacet;
}

function computeFacetAdjustment(docFacet, requestedFacet, facetConfig) {
  const s = facetConfig.scoring || {};
  const matchBoost = Number(s.matchBoost ?? 0.22);
  const mismatchPenalty = Number(s.mismatchPenalty ?? -0.55);
  const defMatch = Number(s.defaultFacetMatchBoost ?? 0.12);
  const defMis = Number(s.defaultFacetMismatchPenalty ?? -0.08);

  if (requestedFacet === facetConfig.defaultFacet) {
    return docFacet === facetConfig.defaultFacet ? defMatch : defMis;
  }
  return docFacet === requestedFacet ? matchBoost : mismatchPenalty;
}

function enforceFacetPriority(sortedItems, requestedFacet, topK) {
  const preferred = [];
  const secondary = [];
  for (const item of sortedItems) {
    if (item.docFacet === requestedFacet) preferred.push(item);
    else secondary.push(item);
  }
  return [...preferred, ...secondary].slice(0, topK);
}

function parseFacetRouterJson(text, facetConfig) {
  let cleaned = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "");
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("Router response missing JSON object");
  }
  const parsed = JSON.parse(cleaned.slice(start, end + 1));
  const primary = typeof parsed.primary === "string" ? parsed.primary.trim() : "";
  let confidence = Number(parsed.confidence);
  if (!Number.isFinite(confidence) && parsed.confidence != null) {
    confidence = Number(String(parsed.confidence).replace(/,/g, ""));
  }
  if (!Number.isFinite(confidence)) {
    confidence = 0.55;
  }
  confidence = Math.min(1, Math.max(0, confidence));
  const reasoning = typeof parsed.reasoning === "string" ? parsed.reasoning : "";
  const valid = getValidFacetIds(facetConfig);
  if (!primary || !valid.has(primary)) {
    throw new Error(`Router returned invalid primary facet: ${primary}`);
  }
  return { primary, confidence, reasoning };
}

const RAG_ANSWER_FALLBACK = [
  "You are a support assistant for JustCall docs.",
  "Answer only using the provided context.",
  "If the context is insufficient, clearly say you do not have enough information.",
  "Cite supporting chunks inline like [#1], [#2].",
  "You may use parent context [P1], [P2] for broader grounding, but final claims should cite child chunks when possible.",
  "Do not cite any chunk number that is not in context.",
  "",
  "Question: {{question}}",
  "",
  "Context:",
  "{{context}}",
].join("\n");

const FACET_ROUTER_FALLBACK = [
  "You route a user question to exactly one knowledge-base facet.",
  "Reply with ONLY a single JSON object. No markdown fences, no extra text.",
  "",
  "Facets:",
  "{{facetLines}}",
  "",
  'If the question is ambiguous or could apply equally to multiple facets, choose "{{defaultFacet}}".',
  "",
  "Set confidence between 0 and 1. Use 0.75 or higher when the facet is clear from the question.",
  "",
  "JSON shape:",
  '{"primary":"<facet_id>","confidence":0.85,"reasoning":"<one short sentence>"}',
  "",
  "User question: {{question}}",
].join("\n");

async function routeFacetWithLlm(question, facetConfig, llm, { callbacks } = {}) {
  const facetLines = facetConfig.facets
    .map((f) => `- "${f.id}": ${f.description}\n  Query hints: ${f.queryHints?.join(", ") || "none"}`)
    .join("\n");
  const defaultId = facetConfig.defaultFacet;

  const langfusePrompt = await fetchPrompt("facet-router");
  const promptText = langfusePrompt
    ? langfusePrompt.compile({ facetLines, defaultFacet: defaultId, question: JSON.stringify(question) })
    : FACET_ROUTER_FALLBACK
        .replace("{{facetLines}}", facetLines)
        .replace("{{defaultFacet}}", defaultId)
        .replace("{{question}}", JSON.stringify(question));

  const invokeOpts = callbacks
    ? { callbacks, metadata: langfusePrompt ? { langfusePrompt } : undefined }
    : langfusePrompt ? { metadata: { langfusePrompt } } : {};
  const res = await llm.invoke(promptText, invokeOpts);
  const text = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
  return parseFacetRouterJson(text, facetConfig);
}

async function resolveRequestedFacet(question, facetConfig, chatModel, { trace } = {}) {
  const minConf = Number(
    process.env.RAG_ROUTER_CONFIDENCE_MIN ?? facetConfig.router?.confidenceMin ?? 0.45
  );
  const routerDisabled =
    process.env.RAG_DISABLE_LLM_ROUTER === "1"
    || process.env.RAG_DISABLE_LLM_ROUTER === "true";

  if (routerDisabled) {
    return {
      primary: detectQueryFacetFromConfig(question, facetConfig),
      confidence: 1,
      reasoning: "",
      source: "keyword_config",
    };
  }

  try {
    const routerModel = process.env.RAG_ROUTER_MODEL || chatModel;
    const routerLlm = makeChatLlm({ model: routerModel, temperature: 0 });
    const routerHandler = createSpanHandler(trace, { name: "facet-routing" });
    const callbacks = routerHandler ? [routerHandler] : undefined;
    const routed = await routeFacetWithLlm(question, facetConfig, routerLlm, { callbacks });
    if (routed.confidence >= minConf) {
      return { ...routed, source: "llm" };
    }
    return {
      primary: facetConfig.defaultFacet,
      confidence: routed.confidence,
      reasoning: routed.reasoning,
      source: "low_confidence_default",
    };
  } catch (err) {
    return {
      primary: detectQueryFacetFromConfig(question, facetConfig),
      confidence: 0,
      reasoning: "",
      source: "llm_error_keyword_fallback",
      error: err.message,
    };
  }
}

function extractDocSignals(doc) {
  const stopwords = STOPWORDS;
  const title = doc.metadata?.title ?? "";
  const source = doc.metadata?.source ?? "";
  const merged = `${title} ${source}`;
  const signals = [];
  for (const token of tokenize(merged)) {
    if (!stopwords.has(token) && !signals.includes(token)) signals.push(token);
  }
  return signals.slice(0, 20);
}

function buildSignalDocumentFrequency(candidates) {
  const df = new Map();
  for (const item of candidates) {
    const uniq = new Set(item.docSignals);
    for (const signal of uniq) df.set(signal, (df.get(signal) || 0) + 1);
  }
  return df;
}

function computeIntentSignalAdjustment(docSignals, queryTerms, signalDf, candidateCount) {
  if (docSignals.length === 0 || queryTerms.length === 0 || candidateCount === 0) return 0;
  const querySet = new Set(queryTerms);
  const rarityThreshold = Math.max(1, Math.floor(candidateCount * 0.25));
  const specificSignals = docSignals.filter((signal) => (signalDf.get(signal) || 0) <= rarityThreshold);
  if (specificSignals.length === 0) return 0;

  let matched = 0;
  let unmatched = 0;
  for (const signal of specificSignals) {
    if (querySet.has(signal)) matched += 1;
    else unmatched += 1;
  }

  const bonus = Math.min(MAX_SIGNAL_BONUS, matched * 0.04);
  const penalty = Math.min(MAX_SIGNAL_PENALTY, unmatched * 0.03);
  return bonus - penalty;
}

function snippet(text, length = 180) {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= length ? clean : `${clean.slice(0, length)}...`;
}

function hasInlineCitation(text) {
  return /\[#\d+\]/.test(text);
}

function enforceCitationDiscipline(answer, docs) {
  const trimmed = (answer || "").trim();
  if (trimmed.length === 0) return trimmed;
  if (hasInlineCitation(trimmed)) return trimmed;
  if (!Array.isArray(docs) || docs.length === 0) return trimmed;

  const references = docs
    .slice(0, Math.min(3, docs.length))
    .map((_, idx) => `[#${idx + 1}]`)
    .join(" ");
  return `${trimmed}\n\nSources: ${references}`;
}

function dedupeByChunkId(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    const id = item.doc.id || `${item.doc.metadata?.source ?? "unknown"}:${item.doc.pageContent.slice(0, 40)}`;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push(item);
  }
  return out;
}

function dedupeBySource(items, maxPerSource = 2) {
  const sourceCount = new Map();
  const out = [];
  for (const item of items) {
    const source = item.doc.metadata?.source || "unknown";
    const count = sourceCount.get(source) || 0;
    if (count < maxPerSource) {
      out.push(item);
      sourceCount.set(source, count + 1);
    }
  }
  return out;
}

function bm25Tokenize(text) {
  return tokenize(text);
}

function computeBm25Scores(docs, queryTerms) {
  if (docs.length === 0 || queryTerms.length === 0) return new Map();

  const docTokens = docs.map((doc) => bm25Tokenize(`${doc.metadata?.title ?? ""} ${doc.pageContent}`));
  const docTermFreqs = docTokens.map((tokens) => {
    const freq = new Map();
    for (const token of tokens) freq.set(token, (freq.get(token) || 0) + 1);
    return freq;
  });

  const docFreq = new Map();
  for (const terms of docTermFreqs) {
    const uniqueTerms = new Set(terms.keys());
    for (const term of uniqueTerms) docFreq.set(term, (docFreq.get(term) || 0) + 1);
  }

  const avgdl = docTokens.reduce((acc, tokens) => acc + tokens.length, 0) / docs.length;
  const scores = new Map();
  const uniqueQueryTerms = Array.from(new Set(queryTerms));
  const N = docs.length;

  for (let i = 0; i < docs.length; i += 1) {
    let score = 0;
    const dl = docTokens[i].length || 1;
    const tfMap = docTermFreqs[i];

    for (const term of uniqueQueryTerms) {
      const tf = tfMap.get(term) || 0;
      if (tf === 0) continue;
      const df = docFreq.get(term) || 0;
      const idf = Math.log(1 + ((N - df + 0.5) / (df + 0.5)));
      const numerator = tf * (BM25_K1 + 1);
      const denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (dl / (avgdl || 1)));
      score += idf * (numerator / denominator);
    }

    scores.set(i, score);
  }

  return scores;
}

function escapeLikePattern(term) {
  return term.replace(/%/g, "\\%").replace(/_/g, "\\_");
}

async function fetchLexicalCandidates(supabase, terms) {
  if (terms.length === 0) return [];
  const seenIds = new Set();
  const docs = [];

  const results = await Promise.all(
    terms.slice(0, 8).map((term) =>
      supabase
        .from(TABLE_NAME)
        .select("id, content, metadata")
        .eq("metadata->>chunkType", "child")
        .ilike("content", `%${escapeLikePattern(term)}%`)
        .limit(LEXICAL_PER_TERM)
    )
  );

  for (const { data, error } of results) {
    if (error || !data) continue;
    for (const row of data) {
      if (seenIds.has(row.id)) continue;
      seenIds.add(row.id);
      if (typeof row.content !== "string" || row.content.trim().length === 0) continue;
      docs.push(new Document({
        id: String(row.id),
        pageContent: row.content,
        metadata: row.metadata ?? {},
      }));
    }
  }

  return docs;
}

async function fetchParentDocs(supabase, childItems) {
  const parentIds = [];
  const seenParent = new Set();

  for (const item of childItems) {
    const parentId = item.doc.metadata?.parentId;
    if (!parentId || seenParent.has(parentId)) continue;
    seenParent.add(parentId);
    parentIds.push(parentId);
    if (parentIds.length >= PARENT_CONTEXT_LIMIT) break;
  }

  if (parentIds.length === 0) return [];

  const results = await Promise.all(
    parentIds.map((parentId) =>
      supabase
        .from(TABLE_NAME)
        .select("id, content, metadata")
        .eq("metadata->>parentId", parentId)
        .eq("metadata->>chunkType", "parent")
        .limit(1)
    )
  );

  const parentDocs = [];
  for (const { data, error } of results) {
    if (error || !data || data.length === 0) continue;
    const row = data[0];
    parentDocs.push(new Document({
      id: String(row.id),
      pageContent: row.content,
      metadata: row.metadata ?? {},
    }));
  }
  return parentDocs;
}

async function hybridRetrieveFromResults(childSemantic, lexicalDocs, vectorStore, supabase, question, queryTerms, topK, facetConfig, requestedFacet) {
  const semanticHits = childSemantic.length > 0
    ? childSemantic
    : await vectorStore.similaritySearchWithScore(question, CANDIDATE_K);

  const byId = new Map();
  for (const [doc, distance] of semanticHits) {
    const id = doc.id || `${doc.metadata?.source ?? "unknown"}:${doc.pageContent.slice(0, 60)}`;
    byId.set(id, {
      doc,
      distance,
      semanticScore: 1 / (1 + distance),
      semanticRank: byId.size + 1,
    });
  }

  for (const doc of lexicalDocs) {
    const id = doc.id || `${doc.metadata?.source ?? "unknown"}:${doc.pageContent.slice(0, 60)}`;
    if (!byId.has(id)) {
      byId.set(id, {
        doc,
        distance: null,
        semanticScore: 0,
        semanticRank: null,
      });
    }
  }

  const candidates = Array.from(byId.values());
  for (const item of candidates) {
    item.docSignals = extractDocSignals(item.doc);
    item.docFacet = inferDocFacet(item.doc, facetConfig);
  }
  const signalDf = buildSignalDocumentFrequency(candidates);
  const bm25Raw = computeBm25Scores(candidates.map((c) => c.doc), queryTerms);
  const maxBm25 = Math.max(...Array.from(bm25Raw.values()), 0);

  const rescored = candidates.map((item, idx) => {
    const bm25Score = bm25Raw.get(idx) || 0;
    const bm25Normalized = maxBm25 > 0 ? bm25Score / maxBm25 : 0;
    const lexicalScore = bm25Normalized;
    const intentAdjustment = computeIntentSignalAdjustment(
      item.docSignals,
      queryTerms,
      signalDf,
      candidates.length
    );
    const facetAdjustment = computeFacetAdjustment(item.docFacet, requestedFacet, facetConfig);
    const hybridScore = (
      item.semanticScore * SEMANTIC_WEIGHT
      + lexicalScore * BM25_WEIGHT
      + intentAdjustment * INTENT_SIGNAL_WEIGHT
      + facetAdjustment * FACET_ROUTING_WEIGHT
    );
    return {
      ...item,
      bm25Score,
      bm25Normalized,
      lexicalScore,
      intentAdjustment,
      facetAdjustment,
      hybridScore,
    };
  });

  const sorted = dedupeByChunkId(
    rescored.sort((a, b) => b.hybridScore - a.hybridScore)
  );

  // Diverisfy results by deduping source-level redundancy
  const diversified = dedupeBySource(sorted, 2);

  const reranked = enforceFacetPriority(diversified, requestedFacet, topK);

  const parentDocs = await fetchParentDocs(supabase, reranked);
  return { queryTerms, reranked, parentDocs, usedStructured: childSemantic.length > 0, requestedFacet };
}

export async function querySystem(
  question,
  {
    topK = TOP_K,
    chatModel = CHAT_MODEL,
    temperature = 0,
    facetConfig: facetConfigOverride = null,
    userId = undefined,
    sessionId = undefined,
    tags = [],
    streaming = false,
  } = {}
) {
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required. Get one at https://aistudio.google.com");
  }

  // Initialize Langfuse trace for this query
  const { trace } = createTraceHandler({
    name: "rag-query",
    userId,
    sessionId,
    tags: ["rag", `provider:${CHAT_PROVIDER}`, ...tags],
    metadata: { model: chatModel, topK, temperature },
  });

  try {
    // Set trace input
    if (trace) {
      trace.update({ input: question });
    }

    const facetConfig = facetConfigOverride ?? loadFacetConfig();
    const rateLimiter = createRateLimiter();
    const supabase = makeSupabaseClient();
    const embeddings = new GeminiEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      dimensions: 1536,
    });
    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabase,
      tableName: TABLE_NAME,
      queryName: QUERY_NAME,
    });

    // Run facet routing and initial retrieval in parallel
    const queryTerms = extractQueryTerms(question);

    // Log retrieval as a span
    const retrievalSpan = trace?.span({
      name: "hybrid-retrieval",
      input: { question, queryTerms, candidateK: CANDIDATE_K },
    });

    const [facetRouter, semanticResults, lexicalDocs, answerPromptResult] = await Promise.all([
      resolveRequestedFacet(question, facetConfig, chatModel, { trace }),
      vectorStore.similaritySearchWithScore(question, CANDIDATE_K, { chunkType: "child" }),
      fetchLexicalCandidates(supabase, queryTerms),
      fetchPrompt("rag-answer"),
    ]);

    const {
      reranked,
      parentDocs,
      usedStructured,
      requestedFacet,
    } = await hybridRetrieveFromResults(semanticResults, lexicalDocs, vectorStore, supabase, question, queryTerms, topK, facetConfig, facetRouter.primary);
    const docs = reranked.map((r) => r.doc);

    if (retrievalSpan) {
      retrievalSpan.end({
        output: {
          facet: requestedFacet,
          facetSource: facetRouter.source,
          docsRetrieved: docs.length,
          usedStructured,
        },
      });
    }

    if (docs.length === 0) {
      if (retrievalSpan) {
        retrievalSpan.end({ output: { docsRetrieved: 0 } });
      }
      if (trace) {
        trace.update({ output: { answer: "", noContext: true } });
      }
      flushLangfuse();
      return {
        question,
        queryTerms,
        requestedFacet,
        facetRouter,
        usedStructured,
        docs: [],
        reranked,
        parentDocs,
        answer: "",
        noContext: true,
      };
    }

    const context = formatContext(docs, parentDocs);
    const llm = makeChatLlm({ model: chatModel, temperature });

    const langfusePrompt = answerPromptResult;
    const prompt = langfusePrompt
      ? langfusePrompt.compile({ question, context })
      : RAG_ANSWER_FALLBACK
          .replace("{{question}}", question)
          .replace("{{context}}", context);

    // Create a span handler for the answer generation LLM call
    const answerHandler = createSpanHandler(trace, { name: "answer-generation" });
    const answerCallbacks = answerHandler ? [answerHandler] : undefined;
    const answerInvokeOpts = {
      ...(answerCallbacks ? { callbacks: answerCallbacks } : {}),
      ...(langfusePrompt ? { metadata: { langfusePrompt } } : {}),
    };

    if (streaming) {
      const stream = await rateLimiter.call(() =>
        llm.stream(prompt, answerInvokeOpts)
      );

      const searchResult = {
        question,
        queryTerms,
        requestedFacet,
        facetRouter,
        usedStructured,
        docs,
        reranked,
        parentDocs,
        stream,
        noContext: false,
      };

      // Wrap the stream to record the full answer in Langfuse when it completes
      const originalStream = searchResult.stream;
      const wrappedStream = (async function* () {
        let fullAnswer = "";
        for await (const chunk of originalStream) {
          const content = typeof chunk.content === "string" ? chunk.content : JSON.stringify(chunk.content);
          fullAnswer += content;
          yield chunk;
        }
        const answerWithCitations = enforceCitationDiscipline(fullAnswer, docs);
        const citationSuffix = answerWithCitations.slice(fullAnswer.length);
        if (citationSuffix) {
          yield { content: citationSuffix };
          fullAnswer = answerWithCitations;
        }

        if (trace) {
          trace.update({
            output: { answer: fullAnswer.trim(), noContext: false },
            metadata: {
              model: chatModel,
              topK,
              temperature,
              facet: requestedFacet,
              facetSource: facetRouter.source,
              docsRetrieved: docs.length,
            },
          });
        }
        flushLangfuse();
      })();

      searchResult.stream = wrappedStream;
      return searchResult;
    }

    const response = await rateLimiter.call(() =>
      llm.invoke(prompt, answerInvokeOpts)
    );
    const rawAnswer = typeof response.content === "string"
      ? response.content
      : JSON.stringify(response.content);
    const answer = enforceCitationDiscipline(rawAnswer, docs);

    // Update trace with final output
    if (trace) {
      trace.update({
        output: { answer: answer.trim(), noContext: false },
        metadata: {
          model: chatModel,
          topK,
          temperature,
          facet: requestedFacet,
          facetSource: facetRouter.source,
          docsRetrieved: docs.length,
        },
      });
    }

    flushLangfuse();

    return {
      question,
      queryTerms,
      requestedFacet,
      facetRouter,
      usedStructured,
      docs,
      reranked,
      parentDocs,
      answer: answer.trim(),
      noContext: false,
    };
  } catch (err) {
    if (trace) {
      trace.update({
        output: { error: err.message },
        level: "ERROR",
        statusMessage: err.message,
      });
    }
    flushLangfuse();
    throw err;
  }
}

function printQueryResult(result) {
  if (result.noContext) {
    console.log("No matching context found in the vector store.");
    return;
  }

  console.log("\nAnswer:\n");
  console.log(result.answer);

  if (result.queryTerms.length > 0) {
    console.log(`\nLexical terms used for BM25: ${result.queryTerms.join(", ")}`);
  }
  const fr = result.facetRouter;
  console.log(
    `Facet router: facet=${result.requestedFacet} source=${fr?.source ?? "?"} confidence=${fr?.confidence?.toFixed?.(2) ?? fr?.confidence ?? "?"}`
  );
  if (fr?.reasoning) console.log(`  reasoning: ${fr.reasoning}`);
  if (fr?.error) console.log(`  router_error: ${fr.error}`);
  console.log(
    `Retrieval mode: ${result.usedStructured ? "structured-child + parent expansion" : "semantic fallback"}`
  );

  if (result.parentDocs.length > 0) {
    console.log(`\nParent context chunks (${result.parentDocs.length}):`);
    for (let i = 0; i < result.parentDocs.length; i += 1) {
      const doc = result.parentDocs[i];
      const source = doc.metadata?.source ?? "unknown";
      const section = doc.metadata?.section ?? "Untitled section";
      console.log(`[P${i + 1}] ${section} — ${source}`);
    }
  }

  console.log(`\nTop ${result.docs.length} retrieved chunks:`);
  for (let i = 0; i < result.reranked.length; i += 1) {
    const item = result.reranked[i];
    const title = item.doc.metadata?.title ?? "Untitled";
    const source = item.doc.metadata?.source ?? "unknown";
    console.log(
      `[#${i + 1}] ${title} — ${source}\n` +
      `  hybrid=${item.hybridScore.toFixed(4)} semantic=${item.semanticScore.toFixed(4)} lexical=${item.lexicalScore.toFixed(4)} bm25=${item.bm25Normalized.toFixed(4)} intentAdj=${item.intentAdjustment.toFixed(4)} facetAdj=${item.facetAdjustment.toFixed(4)} facet=${item.docFacet}\n` +
      `  ${snippet(item.doc.pageContent)}`
    );
  }

  const articleMap = new Map();
  for (const doc of result.docs) {
    const source = doc.metadata?.source ?? "unknown";
    const title = doc.metadata?.title ?? "Untitled";
    if (!articleMap.has(source)) articleMap.set(source, title);
  }

  console.log("\nReferenced articles:");
  for (const [source, title] of articleMap.entries()) {
    console.log(`- ${title} — ${source}`);
  }
}

async function main() {
  const question = getQuestionFromArgs();
  const result = await querySystem(question);
  printQueryResult(result);
}

const isDirectExecution = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isDirectExecution) {
  main().catch((err) => {
    console.error("Query failed:", err.message);
    process.exit(1);
  });
}
