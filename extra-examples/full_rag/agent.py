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
        
        # Query Supabase for relevant documents with source filtering
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'sources': sources}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
                <document>
                {doc['title']}

                {doc['content']}
                </document>
                """
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
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
        query = ctx.deps.supabase.from_('site_pages').select('url')
        
        # If sources are selected, filter by them
        if sources:
            query = query.in_('metadata->>source', sources)
            
        # Execute the query
        result = query.execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@docs_expert.tool
async def get_page_content(ctx: RunContext[DocsAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Get selected sources from deps
        sources = ctx.deps.selected_sources or []
        
        # Build the query
        query = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url)
            
        # If sources are selected, filter by them
        if sources:
            query = query.in_('metadata->>source', sources)
            
        # Execute the query with ordering
        result = query.order('chunk_number').execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"