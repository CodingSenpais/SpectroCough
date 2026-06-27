"""
intent_router.py
----------------
SpectroCough Intent Routing Engine.

Responsibilities:
- Classify user query intent
- Route to correct knowledge base
- Decide whether session context must be injected
- Maintain strict separation:
    - Disease KB
    - Acoustic KB
    - System KB
    - FAQ KB
    - Session-based explanation

STRICT DESIGN:
- No LLM usage here
- Deterministic routing logic
- No memory retention
- Lightweight and fast
"""

import re
from typing import Dict
import json
from runtime.base_paths import WEB_KB_DIR

# ==========================================================
# INTENT CATEGORIES
# ==========================================================

INTENT_EXPLANATION = "explanation"
INTENT_DISEASE_INFO = "disease_information"
INTENT_FEATURE_MEANING = "feature_explanation"
INTENT_SYSTEM_INFO = "system_information"
INTENT_FAQ = "faq"
INTENT_UNKNOWN = "unknown"


# ==========================================================
# KEYWORD MAPS
# ==========================================================

EXPLANATION_KEYWORDS = [
    "why",
    "predicted",
    "prediction",
    "result",
    "classified",
    "detected",
    "confidence",
    "explain",
    "how did you"
]

FEATURE_KEYWORDS = [
    "rms",
    "zcr",
    "spectral",
    "centroid",
    "bandwidth",
    "rolloff",
    "contrast",
    "mfcc",
    "feature",
    "acoustic",
    "energy",
    "frequency",
    "acoustic profile",
    "feature deviation"
]

MICROPHONE_KEYWORDS = [
    "microphone",
    "mic",
    "cough sound",
    "voice cough",
    "recorded cough",
    "phone recording",
    "respiratory sound"
]

SYSTEM_KEYWORDS = [
    "how does this work",
    "about system",
    "privacy",
    "store",
    "data",
    "offline",
    "real time",
    "accuracy",
    "model",
    "trained"
]

DISEASE_KEYWORDS = [

    # General disease queries

    "disease",
    "condition",
    "symptoms",
    "causes",
    "treatment",
    "risk",
    "diagnosis",
    "infection",
    "respiratory disease",
    "lung disease",

    # Common aliases

    "covid",
    "covid19",
    "sneeze",
    "healthy cough"
]


# ==========================================================
# DYNAMIC DISEASE CLASS DISCOVERY
# ==========================================================

def get_all_known_classes():

    classes = []

    for folder in [
        "stethoscope_profiles",
        "microphone_profiles"
    ]:

        stats_path = (
            WEB_KB_DIR /
            folder /
            "class_statistics.json"
        )

        try:

            with open(
                stats_path,
                "r",
                encoding="utf-8"
            ) as f:

                stats = json.load(f)

            classes.extend(
                stats.keys()
            )

        except Exception:
            pass

    aliases = []

    for c in classes:

        aliases.append(
            c.lower()
        )

        aliases.append(
            c.lower().replace("_", " ")
        )

    return list(
        set(aliases)
    )

KNOWN_CLASSES = get_all_known_classes()
# ==========================================================
# HELPER
# ==========================================================

def contains_any(text: str, keywords: list) -> bool:
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


# ==========================================================
# CORE ROUTING LOGIC
# ==========================================================

def route_intent(user_query: str) -> Dict[str, object]:
    """
    Determine user intent and routing configuration.

    Returns:
    {
        "intent": str,
        "use_session_context": bool,
        "target_kb": str
    }
    """

    query = user_query.lower().strip()

    dynamic_classes = KNOWN_CLASSES

    # 1️. Prediction Explanation (highest priority)
    if contains_any(query, EXPLANATION_KEYWORDS):
        return {
            "intent": INTENT_EXPLANATION,
            "use_session_context": True,
            "target_kb": None
        }

    # 2️. Feature Meaning
    if contains_any(query, FEATURE_KEYWORDS):
        return {
            "intent": INTENT_FEATURE_MEANING,
            "use_session_context": True,
            "target_kb": "acoustic_kb"
        }

    # 3. Microphone respiratory queries
    if contains_any(query, MICROPHONE_KEYWORDS):
        return {
            "intent": INTENT_FEATURE_MEANING,
            "use_session_context": True,
            "target_kb": "acoustic_kb"
        }

    if contains_any(
        query,
        dynamic_classes
    ):
        return {
            "intent": INTENT_DISEASE_INFO,
            "use_session_context": False,
            "target_kb": "disease_kb"
        }

    # 4. Disease Info
    if contains_any(query, DISEASE_KEYWORDS):
        return {
            "intent": INTENT_DISEASE_INFO,
            "use_session_context": False,
            "target_kb": "disease_kb"
        }

    # 5. System Info
    if contains_any(query, SYSTEM_KEYWORDS):
        return {
            "intent": INTENT_SYSTEM_INFO,
            "use_session_context": False,
            "target_kb": "system_kb"
        }

    # 6. Default → FAQ
    return {
        "intent": INTENT_FAQ,
        "use_session_context": False,
        "target_kb": "faq_kb"
    }


# ==========================================================
# VALIDATION (Optional Safety)
# ==========================================================

def validate_intent_output(intent_output: Dict[str, object]) -> bool:
    required_keys = ["intent", "use_session_context", "target_kb"]
    return all(key in intent_output for key in required_keys)