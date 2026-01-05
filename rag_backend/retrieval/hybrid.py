"""
Hybrid Retriever Implementation for COMP5423 RAG System.

This module implements a hybrid retriever that combines multiple retrieval methods
using score fusion to improve retrieval performance and robustness.

Key Features:
- Combines results from multiple retrievers using score fusion
- Delegates index management to component retrievers
- Configurable combination of different retrieval approaches

Important Notes:
- Does not manage indices directly - delegates to component retrievers
- Uses simple score fusion by default (sum of scores)
- All component retrievers must be initialized before use
"""

from collections import defaultdict
from retrieval.base import BaseRetriever
from typing import List, Tuple, Dict


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines results from multiple retrievers using score fusion.
    
    Note: This class does not manage index files directly. It delegates all index
    building and loading operations to its component retrievers, which are
    expected to handle their own persistence.
    """
    def __init__(self, config: dict, retrievers: List[BaseRetriever]):
        super().__init__(config)
        if not retrievers:
            raise ValueError("At least one retriever must be provided for HybridRetriever.")
        
        self.retrievers = retrievers
        print("Initializing Hybrid Retriever...")
        for i, retriever in enumerate(self.retrievers):
            print(f"  - Component {i+1}: {retriever.__class__.__name__}")
        print("Hybrid Retriever initialized.")

    def build_index(self, documents: List[str], doc_ids: List[str]):
        """
        Delegates index building to all component retrievers.
        Each retriever is responsible for its own index persistence.
        """
        print("Building indices for all retrievers in the hybrid system...")
        for retriever in self.retrievers:
            retriever_name = retriever.__class__.__name__
            print(f"-> Building index for {retriever_name}...")
            retriever.build_index(documents, doc_ids)
            print(f"Index for {retriever_name} is ready.")
        print("All hybrid component indices have been built and saved.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieves documents by fusing scores from all component retrievers.
        
        This method assumes that all component retrievers have their indices
        loaded or built successfully. If a retriever's index is not ready,
        it will raise a RuntimeError.
        """
        # print(f"Performing hybrid retrieval for query: '{query}'")
        
        # Simple score fusion: sum scores from all retrievers
        fused_scores = defaultdict(float)
        
        for retriever in self.retrievers:
            retriever_name = retriever.__class__.__name__
            # Retrieve more candidates from each retriever to have a better pool for fusion
            results = retriever.retrieve(query, top_k * 2)
            print(f"  - {retriever_name} retrieved {len(results)} candidates.")
            
            for doc_id, score in results:
                fused_scores[doc_id] += score
        
        # Sort by the aggregated score
        sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_results = sorted_results[:top_k]
        print(f"Hybrid retrieval complete. Returning top {len(final_results)} results.")
        
        return final_results