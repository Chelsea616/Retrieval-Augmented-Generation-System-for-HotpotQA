"""
Retriever Package Initialization for COMP5423 RAG System.

This module serves as the package initialization file for the retrieval module.
It imports all retriever classes and provides a factory function to instantiate
the appropriate retriever based on configuration.

Key Components:
- BaseRetriever: Abstract base class for all retrievers
- Various retriever implementations (BM25, Dense, ColBERT, etc.)
- Factory function for dynamic retriever creation

Important Notes:
- Supports multiple retrieval methods through configuration
- Enables easy switching between different retrieval strategies
- Hybrid retriever combines multiple retrieval methods
"""

from .base import BaseRetriever  # Abstract base class for all retrievers
from .bm25 import BM25Retriever  # Sparse retrieval using BM25 algorithm
from .dense import DenseRetriever  # Dense vector retrieval with encoder models
from .colbert import ColBERTRetriever  # Multi-vector retrieval using ColBERT
from .hybrid import HybridRetriever  # Combination of multiple retrieval methods
from .instructed_dense import InstructedDenseRetriever  # Instruction-aware dense retrieval
from .word2vec import Word2VecRetriever  # Retrieval using Word2Vec embeddings


def get_retriever(config: dict) -> BaseRetriever:
    """
    Factory function to create a retriever instance based on the configuration.
    """
    method = config.get('method', 'bm25').lower()
    
    if method == 'bm25':
        return BM25Retriever(config)
    elif method == 'dense':
        return DenseRetriever(config)
    elif method == 'colbert':
        return ColBERTRetriever(config)
    elif method == 'instructed_dense':
        return InstructedDenseRetriever(config)
    elif method == 'word2vec':
        return Word2VecRetriever(config)
    elif method == 'hybrid':
        # For hybrid, you need to define the sub-retrievers in the config
        sub_configs = config.get('sub_retrievers', [])
        if not sub_configs:
            raise ValueError("Hybrid retriever requires 'sub_retrievers' in the config.")
        retrievers = [get_retriever(sub_conf) for sub_conf in sub_configs]
        return HybridRetriever(config, retrievers)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")


__all__ = [
    'BaseRetriever',
    'BM25Retriever',
    'DenseRetriever',
    'ColBERTRetriever',
    'HybridRetriever',
    'Word2VecRetriever',
    'get_retriever'
]