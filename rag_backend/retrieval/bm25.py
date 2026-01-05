"""
BM25 Retriever Implementation for COMP5423 RAG System.

This module implements a BM25-based sparse retriever with index persistence.
It uses the classic BM25 algorithm for term-based document retrieval.

Key Features:
- BM25 algorithm implementation with configurable parameters
- Index persistence using joblib for efficient storage
- Batch retrieval and evaluation capabilities

Important Notes:
- Supports index saving and loading to avoid rebuilding
- Includes text preprocessing and tokenization
- Provides evaluation metrics for retrieval performance
"""

# retrieval/bm25.py

from rank_bm25 import BM25Okapi
from retrieval.base import BaseRetriever
from typing import List, Tuple, Dict
import math
import heapq
import numpy as np
from collections import defaultdict
import os
import joblib  # For efficient serialization and deserialization of Python objects


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever with index persistence capabilities.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.collection = None
        self.doc_ids = None
        self.doc_texts = None

        # BM25 parameters
        self.k1 = config.get('k1', 1.5)
        self.b = config.get('b', 0.75)
        self.epsilon = config.get('epsilon', 0.25)
        
        # Index related
        self.index_path = config.get('index_path', "indexes/bm25_index.pkl")  # Index file path
        self.index_built = False  # Flag indicating whether index is built or loaded
        
        # BM25 index data
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.corpus_size = 0
        self.word_doc_count = defaultdict(int)
        self.idf = {}
        self.doc_term_freqs = []
        
        # Try to load existing index
        self._try_load_index()
        
    def preprocess_text(self, text: str) -> List[str]:
        """Text preprocessing - tokenize and clean text"""
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return words
        
    def build_index(self, documents: List[str], doc_ids: List[str]):
        """Build BM25 index and save to disk"""
        print("Building BM25 index...")
        
        self.doc_ids = doc_ids
        self.doc_texts = documents
        self.corpus_size = len(documents)
        
        # First pass: calculate document lengths and term frequencies
        doc_terms_list = []
        total_length = 0
        
        for doc_text in self.doc_texts:
            words = self.preprocess_text(doc_text)
            doc_terms_list.append(words)
            doc_length = len(words)
            self.doc_lengths.append(doc_length)
            total_length += doc_length
            
            # Count document frequency for each word
            for word in set(words):
                self.word_doc_count[word] += 1
        
        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate inverse document frequency (IDF)
        self.idf = {}
        for word, doc_count in self.word_doc_count.items():
            idf_value = math.log((self.corpus_size - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
            self.idf[word] = idf_value
        
        # Build document term frequency index
        self.doc_term_freqs = []
        for words in doc_terms_list:
            term_freq = defaultdict(int)
            for word in words:
                term_freq[word] += 1
            self.doc_term_freqs.append(term_freq)
        
        self.index_built = True
        print(f"BM25 index built. Corpus size: {self.corpus_size}, Avg doc length: {self.avg_doc_length:.2f}")
        
        # Save index to disk
        self.save_index()
    
    def save_index(self):
        """Save index to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Prepare data to save
            index_data = {
                'doc_ids': self.doc_ids,
                'doc_texts': self.doc_texts,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'corpus_size': self.corpus_size,
                'word_doc_count': dict(self.word_doc_count),
                'idf': self.idf,
                'doc_term_freqs': self.doc_term_freqs,
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
            
            # Use joblib to save (more efficient than pickle)
            joblib.dump(index_data, self.index_path)
            print(f"BM25 index saved to: {self.index_path}")
            
        except Exception as e:
            print(f"Error saving BM25 index: {e}")
    
    def _try_load_index(self):
        """Try to load index from disk"""
        if not os.path.exists(self.index_path):
            return False
            
        try:
            print(f"Loading BM25 index from: {self.index_path}")
            index_data = joblib.load(self.index_path)
            
            # Load index data
            self.doc_ids = index_data['doc_ids']
            self.doc_texts = index_data['doc_texts']
            self.doc_lengths = index_data['doc_lengths']
            self.avg_doc_length = index_data['avg_doc_length']
            self.corpus_size = index_data['corpus_size']
            self.word_doc_count = defaultdict(int, index_data['word_doc_count'])
            self.idf = index_data['idf']
            self.doc_term_freqs = index_data['doc_term_freqs']
            
            # Verify parameter consistency
            if (abs(self.k1 - index_data['k1']) > 1e-6 or 
                abs(self.b - index_data['b']) > 1e-6 or 
                abs(self.epsilon - index_data['epsilon']) > 1e-6):
                print("Warning: BM25 parameters differ from saved index. Rebuilding...")
                return False
            
            self.index_built = True
            print(f"BM25 index loaded successfully. Corpus size: {self.corpus_size}")
            return True
            
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False
    
    def get_idf(self, word: str) -> float:
        """Get IDF value for a word"""
        return self.idf.get(word, 0.0)
    
    def compute_bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_term_freq = self.doc_term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]
        
        for term in query_terms:
            if term not in doc_term_freq:
                continue
            
            tf = doc_term_freq[term]
            idf = self.get_idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            if denominator > 0:
                score += idf * numerator / denominator
        
        return score
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query"""
        if not self.index_built:
            raise RuntimeError("BM25 index not built. Please call build_index() first.")
            
        query_terms = self.preprocess_text(query)
        
        if not query_terms:
            return []
        
        # Calculate BM25 scores for each document
        scores = []
        for doc_index in range(len(self.doc_texts)):
            score = self.compute_bm25_score(query_terms, doc_index)
            scores.append(score)
        
        # Get top_k results
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            results.append((doc_id, score))
        return results

    
    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """Batch retrieval for multiple queries"""
        return [self.retrieve(query, top_k) for query in queries]
    
    def evaluate_on_train(self, train_data: List[Dict], top_k: int = 10) -> Dict[str, float]:
        """Evaluate retrieval performance on training data"""
        print("Evaluating BM25 retrieval performance on train data...")
        
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
        
        print(f"BM25 evaluation results: {metrics}")
        return metrics