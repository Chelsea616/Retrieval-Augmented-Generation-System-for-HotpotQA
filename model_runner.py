# ---------------------------  Top Dependencies  ---------------------------
import sys
import os
import textwrap
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "rag_backend"))

import sys
import os
from collections import deque
import re
import spacy
import yaml
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "rag_backend"))

from rag_backend.data.load_data import DataLoader
from rag_backend.retrieval import get_retriever
from rag_backend.generation import QwenGenerator

# ===== Load config settings =====
with open("rag_backend/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ===== Load collection and model components =====
loader = DataLoader()
collection = loader.load_collection()

generator = QwenGenerator(config['generation'])
retriever = get_retriever(config['retrieval'])

# retriever.load_index(config['retrieval']['index_path'])


nlp = spacy.load("en_core_web_sm")

# -------- Categorized queues: store only the last N entries to avoid overflow ----------
dialog_state = {
    "persons": deque(maxlen=4),
    "orgs": deque(maxlen=4),
    "places": deque(maxlen=4),
    "dates": deque(maxlen=2),
    "last_subject": ""
}


# ----------------------- Utility: Extract and categorize entities -----------------------
def extract_named_entities(text: str):
    """Return [(ent_text, label)] with common entity types only"""
    doc = nlp(text)
    return [(e.text, e.label_) for e in doc.ents
            if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE"}]


def update_entity_queues(entities):
    """Push new entities into categorized queues"""
    for ent_text, ent_label in entities:
        if ent_label == "PERSON":
            dialog_state["persons"].append(ent_text)
        elif ent_label == "ORG":
            dialog_state["orgs"].append(ent_text)
        elif ent_label in {"GPE", "LOC"}:
            dialog_state["places"].append(ent_text)
        elif ent_label == "DATE":
            dialog_state["dates"].append(ent_text)


def get_recent(queue_name):
    """Get the most recent entity from the given category; return empty string if none"""
    q = dialog_state[queue_name]
    return q[-1] if q else ""


# ----------------------- Enhanced Question Rewriting ----------------------------
def rewrite_question(question: str, history_turns: list[dict]) -> str:
    q = question.strip()
    q_lower = q.lower()

    # ----- (1) Handle ellipses: and / what about / how about -----
    if q_lower.startswith(("and ", "what about", "how about")) and dialog_state["last_subject"]:
        stripped = re.sub(r"^(and |what about |how about )", "", q, flags=re.I)
        q = f"{dialog_state['last_subject']} {stripped}"

    # ----- (2) Pronoun replacement -----
    def replace_pron(match):
        token = match.group(0).lower()
        if token in {"he", "his", "him", "she", "her"}:
            name = get_recent("persons")
            return name if name else token
        elif token in {"they", "their"}:
            org = get_recent("orgs")
            return org if org else token
        elif token in {"there", "that place"}:
            loc = get_recent("places")
            return loc if loc else token
        return token

    q = re.sub(r"\b(he|his|him|she|her|they|their|there|that place)\b", replace_pron, q, flags=re.I)

    # ----- (3) Time reference: after that / then -----
    if "after that" in q_lower:
        last_date = get_recent("dates")
        if last_date:
            q = q_lower.replace("after that", f"after {last_date}")

    # ----- (4) Update last_subject -----
    ents = extract_named_entities(q)
    if ents:
        dialog_state["last_subject"] = ents[0][0]

    return q


# ------------------- Main Function: Retrieval & Answer Generation --------------------------
def generate_answer(question, history=None):
    if history is None:
        history = []

    # --- Rewrite question ---
    rewritten_q = rewrite_question(question, history)
    print("[DEBUG] Rewritten question: ", rewritten_q)

    # --- Construct context-enhanced query ---
    ctx = " ".join(dialog_state["persons"] + dialog_state["orgs"] + dialog_state["places"])
    combined_query = f"{ctx} {rewritten_q}".strip()

    # --- Retrieve ---
    top_k = config["retrieval"]["top_k"]
    retrieved_ids = retriever.retrieve(combined_query, top_k)
    retrieved_texts = [collection[doc_id] for doc_id, _ in retrieved_ids]

    # --- Generate answer ---
    results = generator.generate(rewritten_q, retrieved_texts)
    print("[DEBUG] results =", results)
    print("[DEBUG] type =", type(results))

    answer = results["answer"]
    sub_queries = results["sub_queries"]

    # --- Update entity queues ---
    update_entity_queues(extract_named_entities(question + " " + answer))

    # --- Construct response (summarized documents) ---
    retrieved_docs = []
    for doc_id, score in retrieved_ids:
        summary = generator.summarize(collection[doc_id])
        retrieved_docs.append({
            "id": doc_id,
            "score": round(float(score), 2),
            "text": summary
        })

    print({
        "id": f"q{len(history) + 1}",
        "question": question,
        "answer": answer,
        "sub_queries": sub_queries,
        "retrieved_docs": retrieved_docs
    })

    return {
        "id": f"q{len(history) + 1}",
        "question": question,
        "answer": answer,
        "sub_queries": sub_queries,
        "retrieved_docs": retrieved_docs
    }
