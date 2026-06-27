"""
rag_retriever.py
----------------
Deterministic RAG Retriever for SpectroCough.

Responsibilities:
- Load appropriate KB
- Retrieve relevant sections
- Return structured knowledge chunks
- No embeddings required
- No vector DB
- No LLM usage

Designed for:
- Hybrid RAG + Session Injection
- Deterministic and controllable behavior
"""

import json
from pathlib import Path
from typing import List, Dict, Any


# ==========================================================
# KB PATH RESOLUTION
# ==========================================================

def get_kb_root():
    """
    Returns path to chatbot_kb directory.
    """
    from runtime.base_paths import CHATBOT_KB_DIR
    current_dir = CHATBOT_KB_DIR
    return current_dir


# ==========================================================
# LOADERS
# ==========================================================

def load_json_file(path: Path) -> Any:

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"[KB Warning] Failed loading {path}: {e}"
        )

        return {}

def load_kb_folder(folder_name: str, audio_type: str = None) -> List[Dict]:
    """
    Load all JSON files from a folder.
    """
    kb_root = get_kb_root()

    folder_path = kb_root / folder_name

    # --------------------------------------------------
    # Optional modality-specific folder
    # --------------------------------------------------

    if audio_type:

        modality_path = folder_path / audio_type

        if modality_path.exists():
            folder_path = modality_path

    if not folder_path.exists():

        print(
            f"[KB Warning] "
            f"Missing KB folder: {folder_path}"
        )

        return []

    documents = []

    for file in sorted(folder_path.glob("*.json")):
        documents.append(load_json_file(file))

    return documents


# ==========================================================
# SEARCH UTILITIES
# ==========================================================

def keyword_match_score(text: str, query: str) -> int:
    """
    Simple keyword overlap scoring.
    """

    text = text.lower()
    query_words = query.lower().split()

    score = 0
    for word in query_words:
        if word in text:
            score += 1

    return score


def rank_documents(documents: List[Dict], query: str) -> List[Dict]:
    """
    Rank documents by keyword overlap.
    """

    scored_docs = []

    for doc in documents:
        doc_text = json.dumps(doc).lower()
        score = keyword_match_score(doc_text, query)

        scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Return only documents with positive score
    return [doc for score, doc in scored_docs if score > 0]


# ==========================================================
# RETRIEVAL FUNCTIONS
# ==========================================================

def retrieve_from_disease_kb(
    query: str,
    audio_type: str = None
) -> List[Dict]:

    documents = load_kb_folder(
        "diseases",
        audio_type
    )

    return rank_documents(
        documents,
        query
    )


def retrieve_from_acoustic_kb(
    query: str,
    audio_type: str = None
) -> List[Dict]:

    documents = load_kb_folder(
        "acoustic",
        audio_type
    )

    return rank_documents(
        documents,
        query
    )


def retrieve_from_system_kb(
    query: str,
    audio_type: str = None
) -> List[Dict]:

    documents = load_kb_folder(
        "system",
        audio_type
    )

    return rank_documents(
        documents,
        query
    )


def retrieve_from_faq_kb(query: str) -> List[Dict]:
    kb_root = get_kb_root()
    faq_path = kb_root / "faq" / "faq.json"

    if not faq_path.exists():
        raise FileNotFoundError("faq.json not found")

    faq_data = load_json_file(faq_path)

    scored = []

    for entry in faq_data:
        combined_text = (entry["question"] + " " + entry["answer"]).lower()
        score = keyword_match_score(combined_text, query)

        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [entry for score, entry in scored if score > 0]


# ==========================================================
# MAIN RETRIEVER API
# ==========================================================

def retrieve_knowledge(
    query: str,
    target_kb: str,
    audio_type: str = None
) -> List[Dict]:
    """
    Central retrieval interface.

    target_kb options:
    - disease_kb
    - acoustic_kb
    - system_kb
    - faq_kb
    """

    if target_kb == "disease_kb":
        return retrieve_from_disease_kb(query, audio_type)

    elif target_kb == "acoustic_kb":
        return retrieve_from_acoustic_kb(query, audio_type)

    elif target_kb == "system_kb":
        return retrieve_from_system_kb(query, audio_type)

    elif target_kb == "faq_kb":
        return retrieve_from_faq_kb(query)

    elif target_kb == "explanation_kb":
        return retrieve_from_explanation_kb(query, audio_type)

    else:
        return []

# ==========================================================
# EXPLANATION KB RETRIEVAL
# ==========================================================

def retrieve_from_explanation_kb(
    query: str,
    audio_type: str = None
) -> List[Dict]:
    """
    Retrieve combined knowledge for explanation generation.

    This merges:
    - disease definitions
    - acoustic feature explanations
    """

    disease_docs = retrieve_from_disease_kb(
        query,
        audio_type
    )

    acoustic_docs = retrieve_from_acoustic_kb(
        query,
        audio_type
    )

    combined = []

    combined.extend(disease_docs)
    combined.extend(acoustic_docs)

    return combined