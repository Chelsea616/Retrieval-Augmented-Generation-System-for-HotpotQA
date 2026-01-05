"""
Word2Vec Retriever Implementation for COMP5423 RAG System.

This module implements a Word2Vec-based retriever that uses word embeddings
for document retrieval with TF-IDF weighting for improved performance.

Key Features:
- Word2Vec embeddings with TF-IDF weighting
- Cosine similarity for document retrieval
- Comprehensive text preprocessing with stop word removal
- Index persistence and evaluation capabilities

Important Notes:
- Trains Word2Vec model on the document corpus
- Uses TF-IDF weighting for better term importance
- Supports both skip-gram and CBOW training algorithms
"""

# word2vec_retrieval.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import heapq
from gensim.models import Word2Vec
import multiprocessing
from retrieval.base import BaseRetriever
from typing import List, Tuple, Dict
import joblib
from collections import defaultdict


class Word2VecRetriever(BaseRetriever):
    """
    Word2Vec retriever with index persistence capabilities.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.collection = None
        self.doc_ids = None
        self.doc_texts = None

        # Word2Vec parameters
        self.vector_size = config.get('vector_size', 100)
        self.window = config.get('window', 5)
        self.min_count = config.get('min_count', 2)
        self.workers = config.get('workers', multiprocessing.cpu_count())
        self.sg = config.get('sg', 1)  # skip-gram
        self.use_weighting = config.get('use_weighting', True)
        
        # Index related
        self.index_path = config.get('index_path', "indexes/word2vec_index.pkl")
        self.force_rebuild = config.get('force_rebuild', False)  # New: whether to force index rebuild
        self.index_built = False
        
        # Word2Vec data
        self.documents = []  # Preprocessed documents
        self.document_vectors = []
        self.model = None
        self.idf_weights = {}
        
        # Stop words
        self.stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'this', 'that', 
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        ])
        
        # Try to load existing index (unless force rebuild is set)
        if not self.force_rebuild:
            self._try_load_index()
        
    def preprocess_text(self, text: str, is_query: bool = False) -> List[str]:
        """Text preprocessing - tokenize, clean, and remove stop words"""
        # Convert to lowercase
        text = text.lower()
        
        # Smart text cleaning - preserve important information
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!\-\']', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            # Handle words with punctuation
            if any(char in token for char in ".,!?'-"):
                clean_token = re.sub(r'[^a-zA-Z0-9]', '', token)
                if len(clean_token) >= 2:  # Minimum word length
                    processed_tokens.append(clean_token)
            else:
                if len(token) >= 2 and (is_query or token not in self.stop_words):
                    processed_tokens.append(token)
        
        return processed_tokens
        
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build Word2Vec index and save to disk"""
        print("Building Word2Vec index...")
        
        self.doc_ids = doc_ids
        self.doc_texts = documents
        
        # Preprocess documents
        print("Preprocessing documents...")
        self.documents = [self.preprocess_text(doc) for doc in self.doc_texts]
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=self.documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            epochs=10,
            compute_loss=True,
            seed=42
        )
        
        print(f"Word2Vec model trained. Vocabulary size: {len(self.model.wv.key_to_index)}")
        
        # Calculate IDF weights
        print("Calculating IDF weights...")
        self._calculate_idf_weights()
        
        # Compute document vectors
        print("Computing document vectors...")
        self._compute_document_vectors()
        
        self.index_built = True
        print(f"Word2Vec index built. Corpus size: {len(self.documents)}")
        
        # Save index to disk
        self.save_index()
    
    def _calculate_idf_weights(self):
        """Calculate IDF weights for terms in the corpus"""
        doc_freq = defaultdict(int)
        total_docs = len(self.documents)
        
        for doc_tokens in self.documents:
            for token in set(doc_tokens):
                doc_freq[token] += 1
        
        self.idf_weights = {}
        for token, freq in doc_freq.items():
            self.idf_weights[token] = np.log((total_docs + 1) / (freq + 1)) + 1
    
    def _compute_document_vectors(self):
        """Compute document vectors using TF-IDF weighted average"""
        self.document_vectors = []
        
        for doc_tokens in self.documents:
            doc_vector = np.zeros(self.vector_size)
            total_weight = 0
            
            # Calculate term frequency
            token_freq = {}
            for token in doc_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            for token, freq in token_freq.items():
                if token in self.model.wv:
                    # TF-IDF weight
                    tf = freq / len(doc_tokens)
                    idf = self.idf_weights.get(token, 1.0)
                    weight = tf * idf
                    
                    doc_vector += self.model.wv[token] * weight
                    total_weight += weight
            
            if total_weight > 0:
                doc_vector /= total_weight
            
            self.document_vectors.append(doc_vector)
        
        self.document_vectors = np.array(self.document_vectors)
    
    def _get_query_vector(self, query: str) -> np.ndarray:
        """Get query vector using term frequency weighting"""
        query_tokens = self.preprocess_text(query, is_query=True)
        
        if not query_tokens:
            return np.zeros(self.vector_size)
        
        query_vector = np.zeros(self.vector_size)
        
        if self.use_weighting:
            # Use term frequency weighting
            token_freq = {}
            for token in query_tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            total_weight = 0
            for token, freq in token_freq.items():
                if token in self.model.wv:
                    weight = freq / len(query_tokens)
                    query_vector += self.model.wv[token] * weight
                    total_weight += weight
            
            if total_weight > 0:
                query_vector /= total_weight
        else:
            # Simple average
            valid_tokens = 0
            for token in query_tokens:
                if token in self.model.wv:
                    query_vector += self.model.wv[token]
                    valid_tokens += 1
            
            if valid_tokens > 0:
                query_vector /= valid_tokens
        
        return query_vector
    
    def save_index(self):
        """Save index to disk including model and vectors"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Prepare data to save
            index_data = {
                'doc_ids': self.doc_ids,
                'doc_texts': self.doc_texts,
                'documents': self.documents,
                'document_vectors': self.document_vectors,
                'idf_weights': self.idf_weights,
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'sg': self.sg,
                'use_weighting': self.use_weighting
            }
            
            # Save index data
            joblib.dump(index_data, self.index_path)
            
            # Save Word2Vec model separately
            model_path = self.index_path.replace('.pkl', '_model.bin')
            self.model.save(model_path)
            
            print(f"Word2Vec index saved to: {self.index_path}")
            print(f"Word2Vec model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error saving Word2Vec index: {e}")
    
    def _try_load_index(self):
        """Try to load index from disk"""
        if not os.path.exists(self.index_path):
            print(f"No existing index found at: {self.index_path}")
            return False
            
        try:
            print(f"Loading Word2Vec index from: {self.index_path}")
            index_data = joblib.load(self.index_path)
            
            # Load index data
            self.doc_ids = index_data['doc_ids']
            self.doc_texts = index_data['doc_texts']
            self.documents = index_data['documents']
            self.document_vectors = index_data['document_vectors']
            self.idf_weights = index_data['idf_weights']
            
            # Load Word2Vec model
            model_path = self.index_path.replace('.pkl', '_model.bin')
            self.model = Word2Vec.load(model_path)
            
            # Verify parameter consistency
            if (self.vector_size != index_data['vector_size'] or 
                self.window != index_data['window'] or 
                self.min_count != index_data['min_count'] or 
                self.sg != index_data['sg'] or 
                self.use_weighting != index_data['use_weighting']):
                print("Warning: Word2Vec parameters differ from saved index. Rebuilding...")
                return False
            
            self.index_built = True
            print(f"Word2Vec index loaded successfully. Corpus size: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"Error loading Word2Vec index: {e}")
            return False
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents using cosine similarity"""
        if not self.index_built:
            raise RuntimeError("Word2Vec index not built. Please call build_index() first.")
            
        query_vector = self._get_query_vector(query)
        
        if np.all(query_vector == 0):
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_vector], self.document_vectors)[0]
        
        # Get top_k results
        top_indices = heapq.nlargest(top_k, range(len(similarities)), key=similarities.__getitem__)
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(similarities[idx])
            results.append((doc_id, score))
        
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch retrieval for multiple queries"""
        return [self.retrieve(query, top_k) for query in queries]
    
    def evaluate_on_train(self, train_data: List[Dict], top_k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval performance on training data"""
        print("Evaluating Word2Vec retrieval performance on train data...")
        
        total_questions = min(100, len(train_data))  # Only evaluate first 100 to save time
        recall_at_k = 0
        total_ndcg = 0
        
        for item in train_data[:total_questions]:
            question = item["text"]
            supporting_ids = set(item["supporting_ids"])
            
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(question, top_k=top_k)
            retrieved_ids = [doc_id for doc_id, score in retrieved_docs]
            
            # Calculate recall@k
            matched = len(set(retrieved_ids) & supporting_ids)
            if len(supporting_ids) > 0:
                recall_at_k += matched / len(supporting_ids)
            
            # Calculate nDCG@k
            relevance_scores = [1.0 if doc_id in supporting_ids else 0.0 for doc_id in retrieved_ids]
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(supporting_ids), top_k)))
            total_ndcg += dcg / idcg if idcg > 0 else 0
        
        metrics = {
            "recall@10": recall_at_k / total_questions,
            "nDCG@10": total_ndcg / total_questions
        }
        
        print(f"Word2Vec evaluation results: {metrics}")
        return metrics

    def get_similar_words(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get similar words from Word2Vec vocabulary"""
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=top_n)
        return []

    def analyze_query(self, query: str):
        """Analyze query tokens and their similarities"""
        query_tokens = self.preprocess_text(query, is_query=True)
        print(f"Query: '{query}'")
        print(f"Processed tokens: {query_tokens}")
        
        # Show vector information for each token
        for token in query_tokens:
            if token in self.model.wv:
                similar = self.get_similar_words(token, top_n=3)
                print(f"  '{token}': similar to {similar}")
            else:
                print(f"  '{token}': NOT in vocabulary")