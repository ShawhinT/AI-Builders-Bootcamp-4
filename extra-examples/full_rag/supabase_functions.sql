-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create a function to get unique sources
create or replace function get_unique_sources()
returns table (
    source text,
    url_count bigint
)
language sql
as $$
    select 
        metadata->>'source' as source,
        count(distinct url) as url_count
    from site_pages
    where metadata->>'source' is not null
    group by metadata->>'source'
    order by url_count desc, source;
$$;


-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the table
alter table site_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);


/**
-- Optional: create a keywords table

-- Create the keywords table
create table keywords (
    id bigserial primary key,
    url text not null,
    chunk_number integer not null,
    keyword text not null,
    definition text not null,
    created_at timestamp with time zone default now(),
    
    -- Composite unique constraint to prevent duplicates
    unique(url, chunk_number, keyword),
    
    -- Check constraints
    constraint chunk_number_positive check (chunk_number >= 0)
);

-- Index for looking up keywords by URL and chunk
create index idx_keywords_url_chunk on keywords(url, chunk_number);

-- Index for keyword lookups
create index idx_keywords_keyword on keywords(keyword);

-- Full text search indexes for RAG
create index idx_keywords_keyword_trgm on keywords using gin (keyword gin_trgm_ops);
create index idx_keywords_definition_trgm on keywords using gin (definition gin_trgm_ops);

-- Enable the pg_trgm extension if not already enabled
create extension if not exists pg_trgm;

-- Create a function to get pages without keywords
CREATE OR REPLACE FUNCTION get_pages_without_keywords()
RETURNS TABLE (
    url varchar,
    chunk_number integer,
    summary varchar,
    content text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sp.url,
        sp.chunk_number,
        sp.summary,
        sp.content
    FROM site_pages sp
    LEFT JOIN keywords k 
        ON sp.url = k.url 
    WHERE k.url IS NULL;
END;
$$ LANGUAGE plpgsql;

 */
