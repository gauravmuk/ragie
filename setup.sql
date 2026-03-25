-- Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- Create a table to store your documents
create table if not exists documents (
  id bigserial primary key,
  content text, -- corresponds to Document.pageContent
  metadata jsonb, -- corresponds to Document.metadata
  embedding vector(1536) -- Gemini embedding-004 is 768, embedding-001 is 1536
);

-- Create a function to search for documents
create or replace function match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  filter jsonb default '{}'
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
    and 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;

-- Index for faster semantic search
create index on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Unique constraint to prevent exact duplicates (content + source URL)
-- We'll use a generated column for the hash or just a unique index on the metadata->>source and md5(content)
create unique index if not exists idx_documents_unique_content_source
on documents ((metadata->>'source'), md5(content));
