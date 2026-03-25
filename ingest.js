// ingest.js
import "dotenv/config";
import * as cheerio from "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GeminiEmbeddings } from "./gemini-embeddings.js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createRateLimiter } from "./rate-limiter.js";
import { makeSupabaseClient } from "./supabase-client.js";
import { createHash } from "crypto";
import pLimit from "p-limit";
import { createProgressStore } from "./progress-store.js";

// ─── Config ───────────────────────────────────────────────────────────────────
const BASE_URL       = "https://help.justcall.io";
const START_URL      = `${BASE_URL}/en/`;
const TABLE_NAME     = "documents";
const QUERY_NAME     = "match_documents";
const PROGRESS_DB_FILE = process.env.PROGRESS_DB_PATH || "progress.db";
const LEGACY_PROGRESS_JSON_FILE = "progress.json";
const CONCURRENCY    = 10;
const PROGRESS_SCHEMA_VERSION = 1;
const STRUCTURED_PARENT_CHUNK_SIZE = 1200;
const STRUCTURED_PARENT_CHUNK_OVERLAP = 120;
const STRUCTURED_CHILD_CHUNK_SIZE = 400;
const STRUCTURED_CHILD_CHUNK_OVERLAP = 60;
const SEMANTIC_FALLBACK_CHUNK_SIZE = 550;
const SEMANTIC_FALLBACK_CHUNK_OVERLAP = 80;

// ─── Helpers ──────────────────────────────────────────────────────────────────
async function fetchHTML(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return cheerio.load(await res.text());
}

function normalizeUrl(u) {
  const noQueryHash = u.split("?")[0].split("#")[0];
  return noQueryHash.endsWith("/") ? noQueryHash.slice(0, -1) : noQueryHash;
}

function normalizeText(text) {
  return text.replace(/\s+/g, " ").trim();
}

function parseFlags() {
  const flags = new Set(process.argv.slice(2));
  const reset = flags.has("--reset");
  return {
    help: flags.has("--help"),
    resetCollection: reset || flags.has("--reset-collection"),
    resetProgress: reset || flags.has("--reset-progress"),
    resetOnly: flags.has("--reset-only"),
  };
}

function computeArticleChecksum({ title, text, sections }) {
  const normalizedSections = sections
    .map((section) => `${normalizeText(section.heading)}\n${normalizeText(section.text)}`)
    .join("\n---\n");
  const canonical = [
    "checksum:v1",
    normalizeText(title),
    normalizeText(text),
    normalizedSections,
  ].join("\n\n");
  return createHash("sha256").update(canonical).digest("hex");
}

function toAbsolute(href, base = START_URL) {
  if (!href || href.startsWith("javascript:") || href.startsWith("mailto:")) return null;
  try {
    const abs = new URL(href, base).href;
    if (!abs.startsWith(BASE_URL)) return null;
    return normalizeUrl(abs);
  } catch {
    return null;
  }
}

function isCollection(url) { return url.includes("/en/collections/"); }
function isArticle(url)    { return url.includes("/en/articles/"); }

function inferArticleFacet({ title, source }) {
  const t = (title || "").toLowerCase();
  const s = (source || "").toLowerCase();
  if (
    s.includes("justcall-email")
    || s.includes("justcall_email")
    || t.includes("justcall email")
    || t.includes(" email ")
  ) {
    return "email";
  }
  return "core";
}

function extractStructuredSections($, title) {
  const root = $("article").first().length ? $("article").first() : $("main").first();
  if (!root.length) return [];

  const sections = [];
  let currentHeading = title;
  let currentLines = [];

  const flush = () => {
    const text = normalizeText(currentLines.join("\n"));
    if (text.length >= 60) {
      sections.push({
        heading: currentHeading || title,
        text,
      });
    }
    currentLines = [];
  };

  root.find("h1,h2,h3,h4,p,li,pre,blockquote,td,th").each((_, el) => {
    const tag = el.tagName?.toLowerCase() ?? "";
    const text = normalizeText($(el).text());
    if (!text) return;

    if (tag.startsWith("h")) {
      flush();
      currentHeading = text;
      return;
    }

    currentLines.push(text);
  });

  flush();
  return sections;
}

// ─── Stage 1: Home → collection URLs ─────────────────────────────────────────
async function discoverCollections() {
  console.log(`\n[Stage 1] Fetching home: ${START_URL}`);
  const $ = await fetchHTML(START_URL);

  const urls = new Set();
  $("a[href]").each((_, el) => {
    const url = toAbsolute($(el).attr("href"));
    if (url && isCollection(url)) urls.add(url);
  });

  console.log(`  Found ${urls.size} collections`);
  return [...urls];
}

// ─── Stage 2: Collection → article URLs ───────────────────────────────────────
async function discoverArticles(collectionUrls) {
  console.log(`\n[Stage 2] Crawling ${collectionUrls.length} collections for article links...`);
  const limit = pLimit(CONCURRENCY);
  const allArticles = new Set();

  await Promise.all(
    collectionUrls.map((url) =>
      limit(async () => {
        try {
          const $ = await fetchHTML(url);
          let count = 0;
          $("a[href]").each((_, el) => {
            const articleUrl = toAbsolute($(el).attr("href"));
            if (articleUrl && isArticle(articleUrl)) {
              allArticles.add(articleUrl);
              count++;
            }
          });
          console.log(`  ${url.split("/").pop()} → ${count} articles`);
        } catch (err) {
          console.error(`  Failed collection: ${url} — ${err.message}`);
        }
      })
    )
  );

  console.log(`  Total unique articles: ${allArticles.size}`);
  return [...allArticles];
}

// ─── Stage 3: Scrape article content ─────────────────────────────────────────
async function scrapeArticle(url) {
  const $ = await fetchHTML(url);

  const title = $("h1").first().text().trim();

  // Intercom help centers use <article> for the body
  const body = $("article").text().trim() || $("main").text().trim();

  // Clean up excessive whitespace
  const text = `${title}\n\n${body}`.replace(/\n{3,}/g, "\n\n").trim();
  const sections = extractStructuredSections($, title);

  return { title, text, sections, url };
}

async function buildArticleDocuments({
  title,
  text,
  source,
  sections,
  parentSplitter,
  childSplitter,
  semanticFallbackSplitter,
}) {
  const docs = [];
  let structuredChildCount = 0;
  let parentCount = 0;
  const facet = inferArticleFacet({ title, source });

  const normalizedSections = sections.length > 0
    ? sections
    : [{ heading: title, text }];

  for (let sectionIndex = 0; sectionIndex < normalizedSections.length; sectionIndex += 1) {
    const section = normalizedSections[sectionIndex];
    const sectionText = normalizeText(section.text);
    if (sectionText.length < 80) continue;

    const parentSeed = `${title}\n\nSection: ${section.heading}\n\n${sectionText}`;
    const parentDocs = await parentSplitter.createDocuments([parentSeed]);

    for (let parentChunkIndex = 0; parentChunkIndex < parentDocs.length; parentChunkIndex += 1) {
      const parentDoc = parentDocs[parentChunkIndex];
      const parentId = `${source}#s${sectionIndex + 1}-p${parentChunkIndex + 1}`;
      parentDoc.metadata = {
        source,
        title,
        section: section.heading,
        parentId,
        facet,
        chunkType: "parent",
        strategy: "structured",
      };
      docs.push(parentDoc);
      parentCount += 1;

      const childDocs = await childSplitter.createDocuments(
        [parentDoc.pageContent],
        [{
          source,
          title,
          section: section.heading,
          parentId,
          facet,
          chunkType: "child",
          strategy: "structured",
        }]
      );
      docs.push(...childDocs);
      structuredChildCount += childDocs.length;
    }
  }

  // Semantic fallback when structure extraction is weak.
  if (structuredChildCount < 3) {
    const fallbackDocs = await semanticFallbackSplitter.createDocuments(
      [text],
      [{
        source,
        title,
        section: title,
        parentId: `${source}#fallback`,
        facet,
        chunkType: "child",
        strategy: "semantic_fallback",
      }]
    );
    docs.push(...fallbackDocs);
  }

  return { docs, structuredChildCount, parentCount };
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  const flags = parseFlags();
  if (flags.help) {
    console.log("Usage: node ingest.js [--reset] [--reset-only] [--reset-progress] [--reset-collection]");
    console.log("  --reset             Reset collection + progress, then ingest");
    console.log("  --reset-only        Only perform reset actions and exit");
    console.log("  --reset-progress    Delete local progress state before ingest");
    console.log("  --reset-collection  Delete all rows in Supabase documents table");
    return;
  }

  const progressStore = createProgressStore({
    dbPath: PROGRESS_DB_FILE,
    schemaVersion: PROGRESS_SCHEMA_VERSION,
  });
  const migration = progressStore.migrateFromJsonIfNeeded({
    jsonPath: LEGACY_PROGRESS_JSON_FILE,
    archiveMigratedFile: true,
  });
  if (migration.migrated && migration.importedCount > 0) {
    console.log(
      `[Progress] Migrated ${migration.importedCount} records from ${LEGACY_PROGRESS_JSON_FILE} into ${PROGRESS_DB_FILE}`
    );
  }

  try {
    const supabase = makeSupabaseClient();

    if (flags.resetCollection) {
      const { error } = await supabase.from(TABLE_NAME).delete().neq("id", 0);
      if (error) {
        console.log(`[Reset] Could not clear "${TABLE_NAME}" (${error.message})`);
      } else {
        console.log(`[Reset] Cleared all rows from "${TABLE_NAME}"`);
      }
    }

  if (flags.resetProgress) {
    progressStore.reset();
    console.log(`[Reset] Cleared progress rows in ${PROGRESS_DB_FILE}`);
  }

    if (flags.resetOnly) {
      console.log("[Reset] Completed reset-only run.");
      return;
    }

  // Stage 1 — home → collections
  const collectionUrls = await discoverCollections();

  // Stage 2 — collections → articles
  const articleUrls = await discoverArticles(collectionUrls);

  console.log(`\n[Stage 3] Scanning ${articleUrls.length} articles for checksum changes...`);

  // Set up LangChain + Supabase
  const parentSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: STRUCTURED_PARENT_CHUNK_SIZE,
    chunkOverlap: STRUCTURED_PARENT_CHUNK_OVERLAP,
  });
  const childSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: STRUCTURED_CHILD_CHUNK_SIZE,
    chunkOverlap: STRUCTURED_CHILD_CHUNK_OVERLAP,
  });
  const semanticFallbackSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: SEMANTIC_FALLBACK_CHUNK_SIZE,
    chunkOverlap: SEMANTIC_FALLBACK_CHUNK_OVERLAP,
  });
  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is required. Get one at https://aistudio.google.com");
  }
  const embeddings = new GeminiEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    dimensions: 1536,
  });
  const rateLimiter = createRateLimiter();
  const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabase,
    tableName: TABLE_NAME,
    queryName: QUERY_NAME,
  });

  const limit = pLimit(CONCURRENCY);
  let ingestedCount = 0;
  let unchangedCount = 0;
  let changedCount = 0;
  let newCount = 0;
  const total = articleUrls.length;

  await Promise.all(
    articleUrls.map((url) =>
      limit(async () => {
        try {
          const { title, text, sections, url: source } = await scrapeArticle(url);

          if (text.length < 100) {
            console.log(`  Skipped (too short): ${title}`);
            return;
          }

          const checksum = computeArticleChecksum({ title, text, sections });
          const previous = progressStore.getArticle(source);
          if (previous && previous.checksum === checksum) {
            unchangedCount += 1;
            console.log(`  [UNCHANGED] ${title}`);
            return;
          }

          if (previous) changedCount += 1;
          else newCount += 1;

          // Remove old chunks for this article before re-ingestion.
          await supabase
            .from(TABLE_NAME)
            .delete()
            .eq("metadata->>source", source);

          const { docs, structuredChildCount, parentCount } = await buildArticleDocuments({
            title,
            text,
            source,
            sections,
            parentSplitter,
            childSplitter,
            semanticFallbackSplitter,
          });

          for (const doc of docs) {
            doc.metadata = {
              ...doc.metadata,
              checksum,
              contentHash: createHash("md5").update(doc.pageContent).digest("hex"),
            };
          }

          // Use a transaction or single-batch insert if supported by the underlying client
          // LangChain vectorStore.addDocuments usually does this in a single call.
          await vectorStore.addDocuments(docs);

          progressStore.upsertArticle({
            source,
            checksum,
            updatedAt: new Date().toISOString(),
            title,
            chunkCount: docs.length,
          });
          ingestedCount += 1;

          console.log(
            `  [INGESTED ${ingestedCount}] ${title} (sections=${sections.length}, parents=${parentCount}, children=${structuredChildCount})`
          );
        } catch (err) {
          console.error(`  Failed article: ${url} — ${err.message}`);
          console.error(err.stack);
        }
      })
    )
  );

    const trackedRows = progressStore.countArticles();
    console.log(
      `\nDone! scanned=${total} ingested=${ingestedCount} new=${newCount} changed=${changedCount} unchanged=${unchangedCount} tracked=${trackedRows} into "${TABLE_NAME}"`
    );
  } finally {
    progressStore.close();
  }
}

main().catch(console.error);
