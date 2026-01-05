"""
Main execution script for the COMP5423 RAG System.

This script serves as the main entry point for the Retrieval-Augmented Generation (RAG) system.
It loads configuration, initializes data loaders, sets up retrieval and generation modules,
processes queries, and saves the prediction results.

Key Components:
1. Data Loading: Loads document collection and query datasets
2. Retrieval: Initializes retriever and builds document index
3. Generation: Generates answers using retrieved documents 
4. Output: Saves predictions in specified format

Important Notes:
- Configuration is loaded from config.yaml
- Currently processes validation queries (test queries are commented out)
- Generation component is currently disabled (commented out)
- Execution time is measured and printed for performance monitoring
"""

import yaml
import time
from data.load_data import DataLoader
from retrieval import get_retriever  # Import the factory function
from generation import QwenGenerator
from utils.helpers import save_predictions


def main():
    # Load configuration from YAML file
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. Load data
    loader = DataLoader()
    collection = loader.load_collection()  # Returns {doc_id: doc_text}
    # train_queries = loader.load_queries("train")
    # val_queries = loader.load_queries("validation")
    test_queries = loader.load_queries("test")
    # print(f"val len: {len(val_queries)}, test len: {len(test_queries)}")

    # 2. Initialize and build the retriever index
    retriever = get_retriever(config['retrieval'])
    
    # Prepare documents and IDs for indexing
    doc_ids = list(collection.keys())
    documents = list(collection.values())
    
    retriever.build_index(documents, doc_ids)

    # 3. Initialize generator
    generator = QwenGenerator(config['generation'])

    # 4. Process queries and generate results
    results = []
    print("--------Start to prepare result--------")
    start_time = time.time()
    for query in test_queries:
        retrieved_ids = [(doc_id, float(score)) for doc_id, score in retriever.retrieve(query['text'], config['top_k'])]

        retrieved_docs = [collection[id] for id, _ in retrieved_ids]
        answer = generator.generate(query['text'], retrieved_docs)
        results.append({
            "id": query['id'],
            "question": query['text'],
            "answer": answer,
            "retrieved_docs": retrieved_ids
        })
        # break

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"retrieval and generation {len(test_queries)} questions cost: {execution_time}s.")
    # 5. Save prediction results
    save_predictions(results, config['output']['test_prediction_file'])


if __name__ == "__main__":
    main()