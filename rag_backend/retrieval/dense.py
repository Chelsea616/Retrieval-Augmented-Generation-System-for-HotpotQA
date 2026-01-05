"""
Dense Retriever Implementation for COMP5423 RAG System.

This module implements a dense vector retriever using sentence transformers and FAISS.
It encodes documents and queries into dense vectors and uses cosine similarity for retrieval.

Key Features:
- Dense vector retrieval with sentence transformers
- FAISS-based efficient similarity search
- Automatic index persistence and loading
- Support for GPU acceleration

Important Notes:
- Uses FAISS for fast similarity search
- Normalizes embeddings for cosine similarity
- Handles different runtime environments (Kaggle, Colab, local)
"""

# retrieval/dense.py

import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from rag_backend.retrieval.base import BaseRetriever
from typing import List, Tuple
import torch


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using sentence-transformers and FAISS with index persistence.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # Read model_name from 'retrieval' config block
        model_name = config.get('model_name', 'intfloat/e5-large-v2') # Use default model from your config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Index related configuration
        # --- Determine safe writeable index directory ---
        # Kaggle/Colab/local all OK
        try:
           # /kaggle/working exists ONLY in Kaggle runtime
           kaggle_working = "/kaggle/working"
           if os.path.exists(kaggle_working):
               index_dir = os.path.join(kaggle_working, "indexes")
           else:
               # Local or Colab fallback
               index_dir = "rag_backend\indexes"
        except:
           index_dir = "rag_backend\indexes"

        # Create directory if missing
        os.makedirs(index_dir, exist_ok=True)

        # Build paths
        self.index_path = os.path.join(index_dir, "dense_index.faiss")
        self.doc_ids_path = os.path.join(index_dir, "dense_index_doc_ids.npy")


        print(f"Initializing Dense Retriever with model: {model_name}")
        print(f"Using device: {self.device}")

        # More concise code, relies on HF_HUB_OFFLINE=1 environment variable for forced offline loading
        self.model = SentenceTransformer(model_name, device=self.device)

        self.index = None
        self.doc_ids = None
        self.index_built = False

        # Try to load existing index during initialization
        self._try_load_index()

    def _try_load_index(self):
        """Try to load existing index and doc_ids from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_ids_path):
            print(f"Found existing Dense index at '{self.index_path}'. Loading...")
            try:
                self.index = faiss.read_index(self.index_path)
                self.doc_ids = list(np.load(self.doc_ids_path, allow_pickle=True))
                self.index_built = True
                print("Dense index and doc_ids loaded successfully.")
                return True
            except Exception as e:
                print(f"Failed to load Dense index: {e}. A new index will be built.")
                self.index_built = False
                return False
        else:
            print(f"No existing Dense index found at '{self.index_path}'. A new index will be built when needed.")
            return False

    def save_index(self):
        """Save index and doc_ids to disk"""
        if not self.index_built or self.index is None:
            print("Cannot save index because it has not been built.")
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save doc_ids list
            np.save(self.doc_ids_path, np.array(self.doc_ids))

            print(f"Dense index and doc_ids saved to '{self.index_path}' and '{self.doc_ids_path}'.")

        except Exception as e:
            print(f"Error saving Dense index: {e}")

    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Dense index and save to disk"""
        print("Building Dense index...")
        self.doc_ids = doc_ids
        print("Encoding documents...")
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
        doc_embeddings = doc_embeddings.cpu().numpy()

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(doc_embeddings)

        # Create a FAISS index (Inner Product is equivalent to cosine similarity on normalized vectors)
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)

        self.index_built = True
        print(f"Dense index built successfully with {self.index.ntotal} vectors.")

        # Save index to disk
        self.save_index()

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents"""
        if not self.index_built:
            raise RuntimeError(
                "Dense index has not been built. "
                "Please ensure the `build_index()` method was called first, "
                f"or an index exists at '{self.index_path}'."
            )

        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding)

        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)

        results = [(self.doc_ids[i], scores[0][j]) for j, i in enumerate(indices[0])]
        return results