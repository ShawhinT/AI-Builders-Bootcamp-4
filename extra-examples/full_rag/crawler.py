import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""

    system_prompt = """
        You are an AI that extracts titles and summaries from documentation chunks.
        Return a JSON object with 'title' and 'summary' keys.
        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
        For the summary: Create a concise summary of the main points in this chunk.
        Keep both title and summary concise but informative.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
            "source": urlparse(url).netloc,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_site_urls(base_url: str, check_sitemap: bool = True) -> List[str]:
    """Get URLs from a website, first checking for sitemap and falling back to crawling.
    
    Args:
        base_url: The website's base URL
        check_sitemap: Whether to check for sitemap first (default: True)
        
    Returns:
        List of URLs found
    """
    if check_sitemap:
        # Try common sitemap paths
        sitemap_paths = ['sitemap.xml', 'sitemap_index.xml', 'sitemap/sitemap.xml']
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        for path in sitemap_paths:
            sitemap_url = f"{base_domain}/{path}"
            try:
                response = requests.get(sitemap_url)
                response.raise_for_status()
                
                # Parse the XML
                root = ElementTree.fromstring(response.content)
                
                # Extract all URLs from the sitemap
                namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
                
                if urls:
                    return urls
                    
            except Exception as e:
                print(f"Error fetching sitemap: {e}")
    return []

async def get_site_urls_recursive(base_url: str, max_depth: int = 2) -> List[str]:
    """Get URLs from a website by recursively crawling internal links using crawl4ai.
    
    Args:
        base_url: The website's base URL
        max_depth: Maximum depth for recursive crawling (default: 2)
        
    Returns:
        List of URLs found
    """
    # Normalize the base URL
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        visited_urls = set()
        urls_to_crawl = [(base_url, 0)]  # (url, depth) pairs
        base_domain = urlparse(base_url).netloc
        
        while urls_to_crawl:
            current_url, depth = urls_to_crawl.pop(0)
            
            # Skip if URL already visited or depth exceeded
            if current_url in visited_urls or depth > max_depth:
                continue
            
            visited_urls.add(current_url)
            logging.info(f"Discovering links from: {current_url} (depth {depth})")
            
            try:
                result = await crawler.arun(current_url, crawl_config)
                if result and result.success:
                    # Process both internal and external links
                    for link_type in ['internal', 'external']:
                        for link in result.links.get(link_type, []):
                            link_url = link['href']
                            
                            # Normalize the URL
                            if link_url.endswith('/'):
                                link_url = link_url[:-1]
                            
                            # Skip invalid or already processed URLs
                            if (not link_url or 
                                'undefined' in link_url or 
                                link_url in visited_urls):
                                continue
                            
                            # Check if URL belongs to same domain
                            link_domain = urlparse(link_url).netloc
                            if link_domain == base_domain:
                                urls_to_crawl.append((link_url, depth + 1))
                else:
                    logging.error(f"Failed to crawl {current_url}: {result.error_message}")
            
            except Exception as e:
                logging.error(f"Error crawling {current_url}: {str(e)}")
                continue
    
    # Return list of discovered URLs
    return list(visited_urls)

async def main(base_urls: List[str]):
    """Main function to crawl multiple websites.
    
    Args:
        base_urls: List of base URLs to crawl
    """
    all_urls = []
    
    # First try to get URLs from sitemaps for each base URL
    for base_url in base_urls:
        urls = get_site_urls(base_url)
        
        # If no URLs found via sitemap, use recursive crawling
        if not urls:
            urls = await get_site_urls_recursive(base_url)
            
        all_urls.extend(urls)
    
    # Remove duplicates while preserving order
    all_urls = list(dict.fromkeys(all_urls))
    
    if not all_urls:
        print("No URLs found to crawl from any of the provided sites")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main([
        "https://e2b.dev/docs",
        "https://apify.com/templates",
        "https://ai.pydantic.dev/",
        "https://crawl4ai.com/mkdocs/"
    ]))