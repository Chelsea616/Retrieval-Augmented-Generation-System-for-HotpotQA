
Project Overview
----------------
A modular RAG-based QA system built on a subset of HotpotQA, integrating five retrieval methods (BM25, Word2Vec, gte-Qwen2, e5-Mistral, ColBERT) with Qwen2.5-0.5B-Instruct. Supports multi-turn search, cross-document reasoning, and evidence-grounded answer generation with evaluation and UI support.

Directory Structure
-------------------
```
- rag_backend/                 # Backend code for the RAG system
  ├── README.md                # Backend-specific documentation
  ├── data/                    # Dataset directory (not included in repo)
  │   ├── dataset/             # HotpotQA subset (to be prepared locally)
  │   └── data_load.py         # Dataset loading utilities
  ├── evaluate/                # Evaluation scripts
  ├── generation/              # Answer generation modules
  ├── retrieval/               # Retrieval modules (BM25, dense, ColBERT, etc.)
  ├── indexes/                 # Retrieval indexes (created automatically during index building)
  ├── result/                  # Generated outputs and predictions
  ├── utils/                   # Utility functions
  ├── build_index.py           # Script to build retrieval indexes
  ├── download_models.py       # Download retrieval and generation models
  └── main.py                  # Entry point for generating test_prediction.jsonl

- static/                      # Frontend static assets (CSS, JS)
- templates/                   # Frontend HTML templates
- app.py                       # Flask application entry point
- model_runner.py              # Core model inference logic
- requirements.txt             # Python dependencies
- README.md                    # Project documentation

```
How to Run
----------
1. (Optional) Create and activate a virtual environment.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare required directories (if not already created):
   ```
   mkdir -p rag_backend/data/dataset
   mkdir -p rag_backend/indexes
   ```

4. Download required models:
   ```
   python rag_backend/download_models.py
   ```
5. Run the Flask application:
   ```
   python app.py
   ```
6. Access the web interface at: http://127.0.0.1:5000

This interface allows you to enter a question, view the retrieved documents, and get the generated answer.



Generating test_prediction.jsonl
--------------------------------
To generate the `test_prediction.jsonl` file, please refer to
`rag_backend/README.md` for detailed backend instructions.

The general workflow is as follows:

1. Update `config.yaml` to select the retrieval and generation models,
   and to specify output paths.

2. Run the main pipeline:
   ```
   python rag_backend/main.py
   ```

3. The pipeline performs the following steps:
(1) Load data from the dataset  
(2) Build or load retrieval indexes  
(3) Initialize the generator  
(4) Process queries and generate answers  
(5) Save results to the output directory

---





