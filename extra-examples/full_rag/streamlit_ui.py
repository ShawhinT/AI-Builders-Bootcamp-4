from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from agent import docs_expert, DocsAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


def display_retrieved_urls(urls: list[str], title: str = "Retrieved URLs"):
    """Display retrieved URLs in the sidebar with a title."""
    with st.sidebar:
        st.subheader(title)
        for url in urls:
            st.markdown(f"- [{url}]({url})")
        st.markdown("---")

async def run_agent_with_streaming(user_input: str, selected_sources: list[str]):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = DocsAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        selected_sources=selected_sources
    )

    # Run the agent in a stream
    async with docs_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Track retrieved URLs
        retrieved_urls = set()

        # Render partial text as it arrives and track tool calls
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        
        # Extract URLs from tool returns
        for msg in filtered_messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if part.part_kind == 'tool-return':
                        try:
                            # Handle retrieve_relevant_documentation returns
                            if 'documents' in part.content:
                                docs = json.loads(part.content)['documents']
                                for doc in docs:
                                    if 'url' in doc:
                                        retrieved_urls.add(doc['url'])
                            # Handle list_documentation_pages returns
                            elif isinstance(part.content, list):
                                retrieved_urls.update(url for url in part.content if isinstance(url, str))
                        except (json.JSONDecodeError, KeyError):
                            continue

        # Display retrieved URLs in sidebar
        if retrieved_urls:
            display_retrieved_urls(sorted(list(retrieved_urls)))

        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def get_unique_sources(supabase: Client) -> list[str]:
    """Fetch unique sources from Supabase using the get_unique_sources function."""
    try:
        result = supabase.rpc('get_unique_sources').execute()
        if not result.data:
            return []
        # Sort by url_count (descending) and return just the source names
        return [item['source'] for item in result.data]
    except Exception as e:
        print(f"Error fetching sources: {str(e)}")
        return []


async def main():
    st.title("AI Agentic RAG")
    st.write("Ask your questions here")

    # Get unique sources for filtering
    if "sources" not in st.session_state:
        st.session_state.sources = await get_unique_sources(supabase)

    # Add source filter multiselect
    selected_sources = st.multiselect(
        "Select documentation sources to search:",
        st.session_state.sources,
        default=st.session_state.sources
    )

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text with selected sources
            await run_agent_with_streaming(user_input, selected_sources)


if __name__ == "__main__":
    asyncio.run(main())
