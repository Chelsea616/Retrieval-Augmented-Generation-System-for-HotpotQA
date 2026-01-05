"""
Index Building Script for COMP5423 RAG System.

This script is responsible for building the document index used by the retrieval module.
It loads the document collection from the dataset and constructs a searchable index
that enables efficient document retrieval during query processing.

Key Functionalities:
1. Load configuration and document collection
2. Initialize retriever based on configuration
3. Build search index from document texts and IDs

Important Notes:
- Must be run before the main RAG system can perform retrieval
- Index building is specific to the retriever type configured
- Document collection is loaded from the HQ-small dataset
- The built index is typically saved to disk for reuse
"""

import yaml
from data.load_data import DataLoader
from retrieval import get_retriever


def main():
    """Main function to build the document index for the retrieval system."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DataLoader()
    collection = loader.load_collection()

    documents = list(collection.values())
    doc_ids = list(collection.keys())

    retriever = get_retriever(config["retrieval"])
    retriever.build_index(documents, doc_ids)


if __name__ == "__main__":
    main()
