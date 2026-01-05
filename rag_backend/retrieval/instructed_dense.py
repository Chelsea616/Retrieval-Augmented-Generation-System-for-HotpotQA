"""
Instructed Dense Retriever Implementation for COMP5423 RAG System.

This module implements a dense retriever using instruction-aware models like GTE-Qwen2.
These models are specifically trained to follow instructions for better retrieval performance.

Key Features:
- Instruction-aware dense retrieval with models like GTE-Qwen2
- FAISS-based efficient similarity search
- Automatic index persistence and loading
- Support for query and document instructions

Important Notes:
- Uses instruction prompts to enhance retrieval quality
- Supports models specifically trained for retrieval tasks
- Handles both query and document encoding with instructions
"""

# retrieval/instructed_dense.py

import numpy as np
import faiss
import os
from transformers import AutoTokenizer, AutoModel
import torch
from retrieval.base import BaseRetriever
from typing import List, Tuple


class InstructedDenseRetriever(BaseRetriever):
    """
    Dense retriever using instruction-aware models like GTE-Qwen2, with index persistence.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config.get('model_name', 'Alibaba-NLP/gte-Qwen2-7B-instruct')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Index related configuration
        self.index_path = config.get('index_path', "indexes/instructed_dense_index.faiss")
        self.doc_ids_path = self.index_path.replace('.faiss', '_doc_ids.npy')
        
        print(f"Initializing Instructed Dense Retriever with model: {model_name}")
        print(f"Using device: {self.device}")
        
        # More concise code, relies on HF_HUB_OFFLINE=1 environment variable for forced offline loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Define instructions for query and document
        # These are specific to the model, check its model card on Hugging Face
        # GTE-Qwen2 models typically don't need explicit instructions in the prompt,
        # but we keep the structure for flexibility. For this model, empty strings are often fine.
        self.query_instruction = "" # "Represent this sentence for searching relevant passages: "
        self.doc_instruction = ""  # "Represent this document for retrieval: "
        
        self.index = None
        self.doc_ids = None
        self.index_built = False
        
        # Try to load existing index during initialization
        self._try_load_index()

    def _try_load_index(self):
        """Try to load existing index and doc_ids from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_ids_path):
            print(f"Found existing Instructed Dense index at '{self.index_path}'. Loading...")
            try:
                self.index = faiss.read_index(self.index_path)
                self.doc_ids = list(np.load(self.doc_ids_path, allow_pickle=True))
                self.index_built = True
                print("Instructed Dense index and doc_ids loaded successfully.")
                return True
            except Exception as e:
                print(f"Failed to load Instructed Dense index: {e}. A new index will be built.")
                self.index_built = False
                return False
        else:
            print(f"No existing Instructed Dense index found at '{self.index_path}'. A new index will be built when needed.")
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
            
            print(f"Instructed Dense index and doc_ids saved to '{self.index_path}' and '{self.doc_ids_path}'.")
            
        except Exception as e:
            print(f"Error saving Instructed Dense index: {e}")

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Helper function for pooling - averages hidden states using attention mask"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encode(self, texts: List[str], instruction: str) -> np.ndarray:
        """Encodes a list of texts with a given instruction using the model"""
        # For some models, instruction needs to be prepended to text
        if instruction:
            texts_with_instruction = [instruction + text for text in texts]
        else:
            texts_with_instruction = texts

        # Tokenize and move to device
        batch_dict = self.tokenizer(texts_with_instruction, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Instructed Dense index and save to disk"""
        print("Building Instructed Dense index...")
        self.doc_ids = doc_ids
        print("Encoding documents with instruction...")
        doc_embeddings = self._encode(documents, self.doc_instruction)
        
        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)
        
        self.index_built = True
        print(f"Instructed Dense index built successfully with {self.index.ntotal} vectors.")
        
        # Save index to disk
        self.save_index()

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents using instruction-aware encoding"""
        if not self.index_built:
            raise RuntimeError(
                "Instructed Dense index has not been built. "
                "Please ensure the `build_index()` method was called first, "
                f"or an index exists at '{self.index_path}'."
            )
        
        query_embedding = self._encode([query], self.query_instruction)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = [(self.doc_ids[i], scores[0][j]) for j, i in enumerate(indices[0])]
        return results