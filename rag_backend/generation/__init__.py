"""
Generation Package Initialization for COMP5423 RAG System.

"""

# generation/__init__.py

from .generator import QwenGenerator
from .prompts import create_rag_prompt

__all__ = ["QwenGenerator", "create_rag_prompt"]