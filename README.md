COMP5423 Group Project
======================

Project Overview
----------------
This project is built with Flask and includes a modular backend for Retrieval-Augmented Generation (RAG), along with a basic web interface.

Directory Structure
-------------------
```
- indexes/             # Stores index or data files
- rag_backend/         # Backend code for RAG 
   └── README.md           # Dedicated documentation for RAG backend
   └── data                # Save dataset and data load code
      └── dataset             # Dataset
      └── data_load.py        # Load data code
   └── evaluate            # Evaluate code
   └── generation          # Generation 
   └── indexes/            # Stores index or data files
   └── result              # Result folder
   └── retrieval           # Retrieval code
   └── utils               # Some helpful code such as to save test result
   └── bulid_index.py      # Build retrieval indexes for retrival
   └── download_models.py  # Download retrival and generation models
   └── main.py             # Generate test_prediction.jsonl file
- static/              # Static assets such as CSS
- templates/           # Frontend HTML templates
- venv_flask/          # Flask-related environment or libraries (use requirements.txt for dependencies)
- app.py               # Main application entry point, runs the Flask server
- model_runner.py      # Core module for model inference and processing
- requirements.txt     # List of Python dependencies
- README.md/      # Project documentation
```
How to Run
----------
1. Install dependencies (recommended to use a virtual environment):
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```
   python app.py
   ```

3. Access the app at:
   http://127.0.0.1:5000

This interface allows you to enter a question, view the retrieved documents, and get the generated answer.

---

To generate the test_prediction.jsonl file, please navigate to the rag_backend\README.md  and follow the steps outlined in the section below.

1. change config.yaml
   Change 'config.yaml' to choose which retrival and generation to use and to define where to store the output data.

2. run code:
   Use command: 'python main.py' to get test_prediction.jsonl
   (1) Load data from dataset;
   (2) Initialize and build the retriever index
   (3) Initialize generator
   (4) Process queries and generate results
   (5) Save result to output


