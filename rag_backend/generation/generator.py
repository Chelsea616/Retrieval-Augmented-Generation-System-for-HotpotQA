"""
Qwen Generator Implementation for COMP5423 RAG System.

This module implements the generation component using Qwen2.5-Instruct models.
It supports both basic RAG and advanced agentic workflows with query decomposition
and self-checking capabilities.

Key Features:
- Basic single-turn RAG generation
- Feature B: Agentic workflow with query decomposition and self-checking
- Support for multiple Qwen2.5 model sizes
- Configurable generation parameters

Important Notes:
- Requires Qwen2.5-Instruct models from Hugging Face
- Supports both basic and advanced generation modes
- Agentic workflow requires retriever integration
"""

# generation/generator.py

import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import create_rag_prompt
from typing import List, Dict, Tuple, Optional, Any


class QwenGenerator:
    """
    Generator based on the Qwen2.5-Instruct model.
    Supports:
    - Basic Single-Turn RAG (original functionality)
    - Feature B: Agentic Workflow (query decomposition, multi-step retrieval, synthesis, self-checking)
    """
    def __init__(self, config: dict):
        """
        Initializes the generator with model configuration and parameters.
        """
        self.model_name = config.get("model_name")
        self.cache_dir = config.get("cache_dir")
        self.generation_params = config.get("generation_params", {})
        
        # --- Feature B Switch ---
        self.enable_feature_b = config.get("enable_feature_b", False)

        if not self.model_name or not self.cache_dir:
            raise ValueError("config.yaml must contain both 'model_name' and 'cache_dir' for generation.")

        print(f"Loading model '{self.model_name}' from cache directory: {self.cache_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Generator loaded successfully.")

    # --- Core Methods for Feature B: Agentic Workflow ---
    
    def _call_llm(self, prompt: str) -> str:
        """A unified LLM calling interface for reuse - generates response for given prompt"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_params
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def _decompose_query(self, query: str) -> List[str]:
        """
        Step 1: Plan - Decompose a complex question into simpler, independently retrievable sub-questions.
        """
        print("[Feature B] Step 1: Decomposing query into sub-questions...")
        
        planning_prompt = f"""
You are a professional query planning assistant. Your task is to break down a complex, multi-hop question into a series of simpler, independent sub-questions.
These sub-questions should be answerable by retrieving text and, when combined, can answer the original question.
Please output the result as a JSON list, for example: ["sub-question 1", "sub-question 2", "sub-question 3"].

Original question: "{query}"

List of sub-questions:"""

        response = self._call_llm(planning_prompt)
        
        # Attempt to parse JSON
        try:
            # Use regex to extract the JSON array
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group(0))
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    print(f"[Feature B]   Decomposed into {len(sub_queries)} sub-questions.")
                    return sub_queries
        except json.JSONDecodeError:
            print("[Feature B]   Warning: Failed to parse JSON from LLM response. Using fallback.")
        
        # Fallback: If JSON parsing fails, try to split by lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        numbered_lines = [re.sub(r'^\d+[\.\)]\s*', '', line) for line in lines if re.match(r'^\d+[\.\)]', line)]
        if numbered_lines:
            print(f"[Feature B]   Fallback: Parsed {len(numbered_lines)} sub-questions from lines.")
            return numbered_lines[:3] # Take up to 3

        # Final Fallback: Return the original query
        print("[Feature B]   Warning: Could not decompose query. Using the original query.")
        return [query]

    def _synthesize_answer(self, original_query: str, all_contexts: List[str]) -> str:
        """
        Step 3: Reflect - Synthesize a final answer based on all contexts retrieved from sub-questions.
        """
        print("[Feature B] Step 3: Synthesizing final answer from all retrieved contexts...")
        
        combined_context = "\n\n---\n\n".join(all_contexts)
        
        synthesis_prompt = f"""
You are a meticulous Q&A assistant. Please synthesize information from the multiple context snippets provided below to answer the final question.
The contexts may come from different documents; please merge them into a coherent, accurate, and complete answer.
Do not add information that is not in the context. If the context is insufficient to answer the question, please state so.

Original question: "{original_query}"

Context Information:
{combined_context}

Please answer the original question based on the information above.
Answer:"""

        return self._call_llm(synthesis_prompt)

    def _self_check(self, answer: str, context: str) -> str:
        """
        Step 4: Self-Check - Verify if the generated answer is supported by the context and correct any potential hallucinations.
        """
        print("[Feature B] Step 4: Self-checking the generated answer for hallucinations...")
        
        self_check_prompt = f"""
You are a fact-checker. Please carefully verify if the "Generated Answer" is fully supported by the "Provided Context".
If the answer is fully supported, output the answer as is.
If the answer contains information not present in the context (i.e., hallucination) or contradicts the context, please correct the answer based on the context to make it more accurate.
If the context information is insufficient to answer the question, please output "Insufficient information to answer."

Provided Context:
{context}

Generated Answer:
{answer}

Fact-checked Answer:"""

        return self._call_llm(self_check_prompt)

    def agentic_generate(self, query: str, retriever) -> str:
        """
        Main workflow for Feature B: Implements a Plan-Act-Reflect loop with query decomposition and self-checking.
        """
        print("\n=== Starting Agentic Workflow (Feature B) ===")
        
    
        sub_queries = self._decompose_query(query)
        
        all_retrieved_docs = []
        all_contexts = []

        
        print("[Feature B] Step 2: Retrieving documents for each sub-query...")
        for i, sub_q in enumerate(sub_queries):
            print(f"   [Action {i+1}/{len(sub_queries)}] Retrieving for: '{sub_q}'")
            # Assume the retriever has a retrieve method that returns a list of (doc_id, score)
            # And a method to get document text by doc_id
            retrieved_results = retriever.retrieve(sub_q, top_k=3) # Retrieve 3 docs per sub-question
            
            # Here, you need your retriever to be able to get document content by ID
            # Assume the retriever has a get_doc_by_id method
            sub_contexts = [retriever.get_doc_by_id(doc_id) for doc_id, score in retrieved_results]
            
            all_contexts.extend(sub_contexts)
            all_retrieved_docs.extend(retrieved_results)

        
        final_answer = self._synthesize_answer(query, all_contexts)
        
       
        checked_answer = self._self_check(final_answer, "\n\n".join(all_contexts))
        
        print("=== Agentic Workflow Finished ===\n")
        return checked_answer,sub_queries

    # --- Original Functionality ---

    def generate(self, query: str, retrieved_docs: List[str]) -> dict[str, list[str] | Any]:
        """
        Basic single-turn RAG generation using retrieved documents and query.
        """
        context = "\n\n".join(retrieved_docs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please answer the user's question based on the provided context."},
            {"role": "user", "content": create_rag_prompt(context, query)}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_params
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        sub_queries = self._decompose_query(query)
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {
            "answer": response.strip(),
            "sub_queries": sub_queries
        }

    def summarize(self, text: str) -> str:
        prompt = f"Please summarize the following content in one sentence:{text}"
        messages = [
            {"role": "system", "content": "You are a document summarization assistant. Please summarize the content "
                                          "provided by the user in one sentence in English."},
            {"role": "user", "content": prompt}
        ]

        chat_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([chat_input], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, **self.generation_params)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return summary.strip()

    def __call__(self, query: str, retriever) -> str:
        """
        Unified call entry point - routes to agentic or basic generation based on configuration.
        """
        if self.enable_feature_b:
            if retriever is None:
                raise ValueError("Feature B requires a retriever instance to be passed.")
            return self.agentic_generate(query, retriever)
        else:
            if retriever is None:
                 raise ValueError("Basic RAG requires a list of retrieved documents.")
             # Assume when Feature B is off, the 'retriever' argument is actually the list of documents
             # To keep the interface consistent, this requires an adjustment in the calling method
             # A better practice is to have separate methods, but for demonstration, we assume this.
            raise NotImplementedError("When enable_feature_b is False, please call the 'generate' method directly with a list of documents.")