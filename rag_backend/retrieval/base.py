"""
Base Retriever Abstract Class for COMP5423 RAG System.

This module defines the abstract base class for all retriever implementations.
It provides the interface that all concrete retriever classes must implement.

Key Components:
- BaseRetriever: Abstract base class with required methods
- Standardized interface for index building and retrieval

Important Notes:
- All retrievers must inherit from this base class
- Ensures consistent interface across different retrieval methods
- Abstract methods must be implemented by subclasses
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval modules.
    """
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """
        Builds the index for the given documents.

        Args:
            documents (List[str]): A list of document texts.
            doc_ids (List[str]): A list of corresponding document IDs.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k most relevant documents for a given query.

        Args:
            query (str): The query string.
            top_k (int): The number of documents to retrieve.

        Returns:
            A list of tuples, where each tuple contains a document ID and its relevance score.
        """
        pass