"""
ColBERT Retriever Implementation for COMP5423 RAG System.

This module implements a ColBERT-based multi-vector retriever using the RAGatouille library.
ColBERT uses late interaction to compute fine-grained similarity between queries and documents.

Key Features:
- Multi-vector retrieval with late interaction
- Automatic index persistence and loading
- Support for both PLAID and legacy ColBERTv2 index formats

Important Notes:
- Requires RAGatouille library for ColBERT implementation
- Indexes are saved to disk for reuse across sessions
- Handles different runtime environments (Kaggle, Colab, local)
"""

# retrieval/colbert.py

import os
import numpy as np
from .base import BaseRetriever
# from ragatouille import RAGPretrainedModel


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT Retriever with automatic index loading and saving.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get('model_name', 'colbert-ir/colbertv2.0')
        
        # --- Determine safe writeable index directory ---
        try:
            kaggle_working = "/kaggle/working"
            if os.path.exists(kaggle_working):
                # Running on Kaggle → use writeable working directory
                base_index_dir = os.path.join(kaggle_working, "indexes")
            else:
                # Local / Colab / server
                base_index_dir = "indexes"
        except:
            base_index_dir = "indexes"

        # Create directory if missing
        os.makedirs(base_index_dir, exist_ok=True)

        # Final index folder for ColBERT
        self.index_path = os.path.join(base_index_dir, "colbert_index")

        self.top_k = config.get('top_k', 10)
        
        print(f"Initializing ColBERT Retriever with model: {self.model_name}")
        
        try:
            print("Downloading ColBERT model from HuggingFace...")
            self.RAG = RAGPretrainedModel.from_pretrained(self.model_name)
            print("ColBERT model loaded successfully from HF Hub.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ColBERT model online: {e}")
        
        self.index_built = False
        
        # Try to load existing index during initialization
        self._try_load_index()

    def _try_load_index(self):
        """Load existing ColBERT index if present (supports PLAID format)."""

        # New-format PLAID index metadata
        metadata_new = os.path.join(self.index_path, "metadata.json")

        # Old-format ColBERTv2 metadata
        metadata_old = os.path.join(self.index_path, "indexMetadata.json")

        # Prefer new PLAID format
        index_to_load = None
        if os.path.exists(metadata_new):
            index_to_load = self.index_path
        elif os.path.exists(metadata_old):
            index_to_load = self.index_path

        if index_to_load:
            print(f"Found existing ColBERT index at '{self.index_path}'. Loading...")
            try:
                self.RAG = RAGPretrainedModel.from_index(self.index_path)
                self.index_built = True
                print("ColBERT index loaded successfully.")
                return True
            except Exception as e:
                print(f"Failed to load ColBERT index: {e}. A new index will be built.")
                self.index_built = False
                return False

        print(f"No existing ColBERT index found at '{self.index_path}'. Will build new index.")
        return False


    def build_index(self, documents: list[str], doc_ids: list[str]):
        """
        Build the ColBERT index using pure text documents (ragatouille requirement).
        Metadata such as doc_ids is saved separately.
        """
        import time as _time
        time = _time
        print("Building ColBERT index. This may take a while...")

        os.makedirs(self.index_path, exist_ok=True)

        self.RAG.index(
            collection=documents,              # <-- MUST be list[str]
            index_name=self.index_path,        # Save in self.index_path directory
            split_documents=True
        )

        
        docid_path = os.path.join(self.index_path, "colbert_doc_ids.npy")
        np.save(docid_path, np.array(doc_ids))
        print(f"Saved doc_ids mapping to {docid_path}")

        
        self.index_built = True
        print(f"ColBERT index built successfully at '{self.index_path}'.")



    def retrieve(self, query: str, top_k: int = None) -> list[tuple[str, float]]:
        """
        Retrieve top_k documents for a given query.
        """
        if not self.index_built:
            raise RuntimeError(
                "ColBERT index has not been built. "
                "Please build or load an index before calling retrieve()."
            )

        if top_k is None:
            top_k = self.top_k

        print(f"Retrieving for query: '{query}' (top_k={top_k})")
        results = self.RAG.search(query=query, k=top_k)

        retrieved_docs = []
        for res in results:
            doc_id = res.get('doc_id')
            score = res.get('score')
            if doc_id is not None and score is not None:
                retrieved_docs.append((doc_id, float(score)))

        return retrieved_docs