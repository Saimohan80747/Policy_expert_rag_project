-- Run once in the Supabase SQL editor.

create table if not exists chat_sessions (
  id uuid primary key,
  title text,
  created_at timestamptz default now()
);

create table if not exists chat_messages (
  id bigint generated always as identity primary key,
  session_id uuid references chat_sessions(id) on delete cascade,
  question text, answer text, file_name text, file_url text,
  created_at timestamptz default now()
);

create table if not exists documents (
  id bigint generated always as identity primary key,
  file_name text,
  file_hash text unique,
  file_url text,
  file_size bigint,
  created_at timestamptz default now()
);

-- Versioning columns (safe to re-run; needed from ingestion step 1 onward).
alter table documents add column if not exists doc_id text;
alter table documents add column if not exists version int default 1;
alter table documents add column if not exists is_current boolean default true;
alter table documents add column if not exists doc_type text;
alter table documents add column if not exists chunk_count int;
create index if not exists documents_doc_id_idx on documents(doc_id);
create index if not exists documents_file_name_idx on documents(file_name);

-- Chat: keep the sources shown with each answer.
alter table chat_messages add column if not exists sources jsonb default '[]'::jsonb;

-- Storage: create a public bucket named "documents" in the Storage tab.
