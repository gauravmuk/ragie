import Database from "better-sqlite3";
import { existsSync, readFileSync, renameSync } from "fs";

const DEFAULT_DB_PATH = process.env.PROGRESS_DB_PATH || "progress.db";

export function createProgressStore({
  dbPath = DEFAULT_DB_PATH,
  schemaVersion = 1,
} = {}) {
  const db = new Database(dbPath);

  // Safer defaults for local durability and concurrent reads.
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");

  db.exec(`
    CREATE TABLE IF NOT EXISTS metadata (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS article_progress (
      source TEXT PRIMARY KEY,
      checksum TEXT,
      updated_at TEXT,
      title TEXT,
      chunk_count INTEGER NOT NULL DEFAULT 0
    );
  `);

  const getMetadataStmt = db.prepare(
    "SELECT value FROM metadata WHERE key = ?"
  );
  const setMetadataStmt = db.prepare(`
    INSERT INTO metadata(key, value)
    VALUES(?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
  `);

  const currentVersionRaw = getMetadataStmt.get("schema_version")?.value;
  const currentVersion = Number(currentVersionRaw || 0);
  if (Number.isNaN(currentVersion) || currentVersion < schemaVersion) {
    setMetadataStmt.run("schema_version", String(schemaVersion));
  }

  const getArticleStmt = db.prepare(`
    SELECT source, checksum, updated_at AS updatedAt, title, chunk_count AS chunkCount
    FROM article_progress
    WHERE source = ?
  `);
  const upsertArticleStmt = db.prepare(`
    INSERT INTO article_progress(source, checksum, updated_at, title, chunk_count)
    VALUES(@source, @checksum, @updatedAt, @title, @chunkCount)
    ON CONFLICT(source) DO UPDATE SET
      checksum = excluded.checksum,
      updated_at = excluded.updated_at,
      title = excluded.title,
      chunk_count = excluded.chunk_count
  `);
  const countArticlesStmt = db.prepare(
    "SELECT COUNT(*) AS count FROM article_progress"
  );
  const clearArticlesStmt = db.prepare("DELETE FROM article_progress");

  const upsertTransaction = db.transaction((record) => {
    upsertArticleStmt.run(record);
  });

  function migrateFromJsonIfNeeded({
    jsonPath = "progress.json",
    archiveMigratedFile = true,
  } = {}) {
    const migrationKey = `legacy_json_migrated:${jsonPath}`;
    if (getMetadataStmt.get(migrationKey)?.value === "1") return { migrated: false, importedCount: 0 };
    if (!existsSync(jsonPath)) {
      setMetadataStmt.run(migrationKey, "1");
      return { migrated: false, importedCount: 0 };
    }

    const raw = JSON.parse(readFileSync(jsonPath, "utf-8"));
    const rows = [];

    if (Array.isArray(raw)) {
      for (const source of raw) {
        if (typeof source !== "string") continue;
        rows.push({
          source,
          checksum: null,
          updatedAt: null,
          title: null,
          chunkCount: 0,
        });
      }
    } else if (raw && typeof raw === "object" && raw.articles && typeof raw.articles === "object") {
      for (const [source, article] of Object.entries(raw.articles)) {
        rows.push({
          source,
          checksum: article?.checksum ?? null,
          updatedAt: article?.updatedAt ?? null,
          title: article?.title ?? null,
          chunkCount: Number(article?.chunkCount || 0),
        });
      }
    }

    const ingestLegacyRows = db.transaction((records) => {
      for (const record of records) {
        upsertArticleStmt.run(record);
      }
    });
    ingestLegacyRows(rows);
    setMetadataStmt.run(migrationKey, "1");

    if (archiveMigratedFile) {
      const archivedPath = `${jsonPath}.migrated`;
      if (!existsSync(archivedPath)) renameSync(jsonPath, archivedPath);
    }

    return { migrated: true, importedCount: rows.length };
  }

  function getArticle(source) {
    return getArticleStmt.get(source) || null;
  }

  function upsertArticle(record) {
    upsertTransaction(record);
  }

  function countArticles() {
    return Number(countArticlesStmt.get()?.count || 0);
  }

  function reset() {
    clearArticlesStmt.run();
  }

  function close() {
    db.close();
  }

  return {
    dbPath,
    migrateFromJsonIfNeeded,
    getArticle,
    upsertArticle,
    countArticles,
    reset,
    close,
  };
}
