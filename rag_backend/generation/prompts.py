"""
Prompt Utilities for COMP5423 RAG System.

This module provides prompt creation utilities for the RAG system.
It contains templates and functions for formatting prompts for the Qwen models.

Key Features:
- RAG prompt template creation
- Support for context and query formatting

Important Notes:
- Designed for use with Qwen chat templates
- Provides basic prompt structure for RAG tasks
"""

# generation/prompts.py

def create_rag_prompt(context: str, query: str) -> str:
    """Creates a basic prompt string for RAG tasks."""
    
    prompt = f"""
            Context Information:
            {context}

            Please answer the user's question based on the context above. If there is no relevant information in the context, please answer "I cannot answer based on the provided information."

            Question: {query}
            """
    return prompt.strip()