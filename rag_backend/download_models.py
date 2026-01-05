"""
Model Download Script for COMP5423 RAG System.

This script handles the automated download of Hugging Face models specified in the configuration.
It extracts model repository IDs from the configuration and downloads them to the local cache
for offline usage in the RAG system.

Key Functionalities:
1. Load configuration from YAML file
2. Extract unique Hugging Face model names
3. Download models to local cache directory
4. Provide download status and error reporting

Important Notes:
- Models are downloaded to the default Hugging Face cache directory
- Currently only downloads E5-Mistral model (other models are commented out)
- Supports offline mode after download by setting HF_HUB_OFFLINE=1
- Requires huggingface_hub package and valid internet connection
"""

# download_models.py

import yaml
import os
from huggingface_hub import snapshot_download
from typing import Set


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the configuration from the YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return {}


def get_model_names() -> Set[str]:
    """
    Extracts all unique Hugging Face model names from the configuration.
    """
    model_names = set()

    models_to_download = [
        # {
        #     "type": "Dense Retrieval (E5)",
        #     "repo_id": "intfloat/e5-large-v2"
        # },
        {
            "type": "Dense Retrieval with Instruction (GTE-Qwen2)",
            "repo_id": "intfloat/e5-mistral-7b-instruct"
            # },
            # {
            #     "type": "Multi-vector Retrieval (ColBERT)",
            #     "repo_id": "colbert-ir/colbertv2.0"
            # },
            # # --- Generation Model ---
            # {
            #     "type": "Generation Model (Qwen2.5-0.5B-Instruct)",
            #     "repo_id": "Qwen/Qwen2.5-0.5B-Instruct"
        }
    ]
    return models_to_download


def download_all_models():
    """
    Downloads all specified models to the local cache.
    """
    print("--- Starting Model Download Process ---")

    model_names = get_model_names()

    print(f"Found the following models to download: {model_names}")

    for model in model_names:
        repo_id = model['repo_id']
        model_type = model['type']
        print(f"\n>>> Downloading model: {repo_id}...")
        try:
            # This will download the model to the default cache directory
            # ~/.cache/huggingface/hub on Linux/MacOS
            # C:\Users\<username>\.cache\huggingface\hub on Windows
            download_path = snapshot_download(repo_id=repo_id, repo_type="model")
            print(f"Successfully downloaded '{repo_id}' to: {download_path}")
        except Exception as e:
            print(f"Failed to download '{repo_id}'. Error: {e}")

    print("\n--- Model Download Process Finished ---")
    print("You can now run your project in offline mode by setting the HF_HUB_OFFLINE=1 environment variable.")


if __name__ == "__main__":
    download_all_models()
