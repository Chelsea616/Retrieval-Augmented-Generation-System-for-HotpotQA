# README for RAG System
This project implements a Retrieval-Augmented Generation (RAG) system for the COMP5423 Natural Language Processing course. The system is built on the HotpotQA dataset and supports multiple retrieval methods integrated with Qwen language models for answer generation.

## Program Structure
```
COMP5423-RAG/
├── data/                    # Data loading and processing
│   ├── dataset/            # HQ-small dataset files
│   └── load_data.py        # Data loader implementation
├── retrieval/              # Retrieval module implementations
│   ├── base.py             # Abstract base retriever
│   ├── bm25.py             # BM25 sparse retriever
│   ├── dense.py            # Dense vector retriever (E5)
│   ├── colbert.py          # ColBERT multi-vector retriever
│   ├── instructed_dense.py # Instruction-aware dense retriever
│   ├── word2vec.py         # Word2Vec-based retriever
│   ├── hybrid.py           # Hybrid retrieval combination
│   └── __init__.py         # Retriever factory function
├── generation/             # Answer generation components
│   ├── generator.py        # Qwen model generator
│   ├── prompts.py          # Prompt templates
│   └── __init__.py         # Generation module exports
├── evaluation/             # Performance evaluation scripts
│   ├── eval_hotpotqa.py    # End-to-end system evaluation
│   └── eval_retrieval.py   # Retrieval-only evaluation
├── indexes/                # Built retrieval indices
├── results/                # Output prediction files
├── utils/                  # Utility functions
│   └── helpers.py          # Data conversion and file operations
├── config.yaml             # System configuration
├── build_index.py          # Index building script
├── download_models.py      # Model downloading script
└── main.py                 # Main execution script
```
## Environment Setup
Prerequisites:
- Python 3.8+
- 16GB RAM minimum (32GB recommended for larger models)
- NVIDIA GPU with 8GB+ VRAM (recommended for generation)

Installation:
Install dependencies:pip install -r requirements.txt


## Step-by-Step Execution
1. **Download Models**:This downloads the configured models (currently E5-Mistral) to the local cache.
   ```bash
   python download_models.py

2. **Build Retrieval Index**:This builds the search index for the configured retriever method.
   ```bash
   python build_index.py

3. **Run the System**
   ```bash
   python main.py

4. **Generate Test Predictions**:This processes test queries and generates predictions in results/test_prediction.jsonl.

## Configuration
- Modify config.yaml to:
- Change retrieval method (BM25, dense, colbert, etc.)
- Adjust generation parameters
- Set different model paths
- Configure output file locations