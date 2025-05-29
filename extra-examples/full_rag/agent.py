from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class DocsAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    selected_sources: list[str] = None  # Add selected sources to deps

system_prompt = """
You are an expert Python AI agent with deep knowledge of various libraries and tools. Your primary function is to assist with queries related to specific documentation and provide code examples or methodologies to another LLM. You have access to comprehensive documentation, including examples, API references, and other resources.

Instructions:

1. Analyze the user's query or the provided document.
2. Always start by using RAG (Retrieval-Augmented Generation) to find relevant information in the documentation.
3. Check the list of available documentation pages and retrieve the content of relevant pages.
4. If you can't find the answer in the documentation or the right URL, always be honest and inform the user.
5. Do not ask for permission before taking an action; proceed directly with your analysis and response.
6. Focus solely on assisting with queries related to the provided documentation. Do not answer questions outside this scope.
7. Provide detailed code examples or methodologies in your response, optimized for consumption by another LLM.

Structure your response inside the following tags:

<analysis>
[Analyze the query or document by:
a. Summarizing the query/document
b. Identifying key concepts or keywords
c. Listing relevant documentation sections to search
d. Outlining the approach for code example creation]
</analysis>

<documentation_reference>
[Cite relevant sections from the documentation, including URLs if available]
</documentation_reference>

<code_example>
[Provide a clear, well-commented code example or methodology]
</code_example>

<explanation>
[Explain the provided solution, its relevance to the query, and any important considerations]
</explanation>

Remember to be thorough in your analysis, precise in your code examples, and clear in your explanations. Your goal is to provide accurate, documentation-based responses that can be easily understood and utilized by another LLM.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

docs_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=DocsAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
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

@docs_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[DocsAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Use selected sources from deps
        sources = ctx.deps.selected_sources or []
        
        # Create a filter object for source filtering
        filter_obj = {}
        
        # Handle source filtering
        if sources and len(sources) == 1:
            # For a single source, we can use the direct filter
            filter_obj = {'source': sources[0]}
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': filter_obj
            }
        ).execute()
        
        # If no results with filter, try without filter
        if not result.data and filter_obj:
            # Try again without filter
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 5,
                    'filter': {}
                }
            ).execute()
        
        if not result.data:
            return "No relevant documentation found. Please check that your database contains indexed documentation."
            
        # Format the results with more detailed information
        formatted_chunks = []
        for doc in result.data:
            source = doc['metadata'].get('source', 'Unknown') if doc.get('metadata') else 'Unknown'
            similarity = doc.get('similarity', 0)
            similarity_percentage = f"{similarity * 100:.1f}%" if similarity else 'N/A'
            
            chunk_text = f"""
                ## {doc['title']} (Relevance: {similarity_percentage})
                **Source**: {source}
                **URL**: {doc['url']}

                {doc['content']}
            """
            formatted_chunks.append(chunk_text)
            
        # Add a header with summary of results
        header = f"# Found {len(formatted_chunks)} relevant documentation chunks\n\n"
        
        # Join all chunks with a separator
        return header + "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@docs_expert.tool
async def list_documentation_pages(ctx: RunContext[DocsAIDeps]) -> List[str]:
    """
    Retrieve a list of all available documentation pages for the selected sources.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Get selected sources from deps
        sources = ctx.deps.selected_sources or []
        
        # Build the query
        query = ctx.deps.supabase.from_('site_pages').select('url, title')
        
        # If sources are selected, filter by them
        if sources:
            # Use in_ operator for multiple sources
            query = query.in_('metadata->>source', sources)
        
        # Execute the query
        result = query.execute()
        
        if not result.data:
            # Return a message indicating no documentation is available
            return ["No documentation pages available. Please check your database connection or add documentation."]
        
        # Extract unique URLs with titles
        unique_pages = {}
        for doc in result.data:
            if doc['url'] not in unique_pages:
                unique_pages[doc['url']] = doc['title']
        
        # Format as a list of strings with title and URL
        formatted_urls = [f"{title} - {url}" for url, title in unique_pages.items()]
        return sorted(formatted_urls)
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return [f"Error retrieving documentation pages: {str(e)}"]

@docs_expert.tool
async def get_page_content(ctx: RunContext[DocsAIDeps], url_or_formatted_string: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url_or_formatted_string: The URL of the page to retrieve, or a formatted string containing the URL
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Extract the URL if it's in a formatted string (e.g., "Title - https://example.com")
        if ' - ' in url_or_formatted_string and ('http://' in url_or_formatted_string or 'https://' in url_or_formatted_string):
            # Extract the URL part after the last ' - '
            url = url_or_formatted_string.split(' - ')[-1].strip()
        else:
            url = url_or_formatted_string.strip()
        
        # Get selected sources from deps
        sources = ctx.deps.selected_sources or []
        
        # Build the query
        query = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url)
            
        # If sources are selected, filter by them
        if sources:
            query = query.in_('metadata->>source', sources)
            
        # Execute the query with ordering
        result = query.order('chunk_number').execute()
        
        if not result.data:
            return f"No content found for URL: {url}. Please check that the URL is correct and exists in the database."
        
        # Get metadata from the first chunk
        metadata = result.data[0].get('metadata', {})
        source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
        
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        
        # Create a header with metadata
        header = f"# {page_title}\n\n**Source**: {source}\n**URL**: {url}\n\n## Content:\n"
        
        # Add each chunk's content
        content_parts = []
        for chunk in result.data:
            content_parts.append(chunk['content'])
            
        # Join everything together
        return header + "\n\n".join(content_parts)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
