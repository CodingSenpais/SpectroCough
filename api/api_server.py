"""
api_server.py
-------------
Flask API server for SpectroCough frontend ↔ ML backend integration.

Responsibilities:
- Receive audio from frontend
- Save temporary audio file
- Run inference using infer.py
- Return prediction JSON
- Save screening report
- Provide report history
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from werkzeug.exceptions import RequestEntityTooLarge
import logging
import re

from api.infer import run_inference
from api.file_handler import save_uploaded_file, delete_file

from runtime.counterfactual_engine import compute_counterfactual
from runtime.fingerprint_engine import generate_fingerprint
from runtime.spectrogram_engine import build_spectrogram_analysis
from runtime.rag_runtime_orchestrator import run_full_runtime_pipeline

from runtime.base_paths import API_DIR

from runtime.explanation_engine import (
    generate_explanation
)

# ============================================================
# ROOT PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

# Maximum upload size (15 MB)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# Restrict CORS to your frontend
if os.getenv("FLASK_ENV") == "production":
    CORS(app)
else:
    CORS(
        app,
        resources={
            r"/*": {
                "origins": [
                    "http://127.0.0.1:5500",
                    "http://localhost:5500"
                ]
            }
        }
    )

REPORT_FILE = API_DIR / "reports.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SpectroCough")


# ============================================================
# REPORT STORAGE
# ============================================================

def load_reports():
    """Load report history from JSON file."""
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w") as f:
            json.dump([], f)
        return []

    with open(REPORT_FILE, "r") as f:
        return json.load(f)


def save_report(entry):
    """Append new report entry."""
    reports = load_reports()
    reports.append(entry)

    with open(REPORT_FILE, "w") as f:
        json.dump(reports, f, indent=4)



@app.after_request
def add_security_headers(response):

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(), camera=()"

    return response

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):

    return jsonify({
        "error": "Maximum upload size is 15 MB."
    }), 413
# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "SpectroCough API running"
    })


# ============================================================
# AUDIO INFERENCE ROUTE
# ============================================================

@app.route("/infer", methods=["POST"])
def infer_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    analysis_type = request.form.get("analysis_type", "AI Analysis")
    user_email = request.form.get("user_email","")

    audio_type = request.form.get(
        "audio_type",
        "stethoscope"
    ).lower()

    if audio_type not in (
        "stethoscope",
        "microphone"
    ):
        return jsonify({
            "error": "Invalid audio type."
        }), 400

    temp_path = None

    try:
        temp_path = save_uploaded_file(audio_file)
        prediction = run_inference(
            temp_path,
            audio_type=audio_type
        )

        acoustic_vector = prediction["acoustic_vector"]
        predicted_class = prediction["predicted_class"]
        confidence = prediction["confidence"]

        # Generate explanation
        

        runtime_output = run_full_runtime_pipeline(
            acoustic_vector=np.array(acoustic_vector),
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prediction.get(
                "probabilities"
            ),
            audio_type=audio_type
        )

        audio = prediction.pop(
            "audio",
            None
        )

        sr = prediction.pop(
            "sr",
            None
        )

        # ---------------------------------------------------
        # Spectrogram analysis requires raw audio
        # ---------------------------------------------------

        spectrogram_analysis = None

        if audio is not None and sr is not None:

            spectrogram_analysis = (
                build_spectrogram_analysis(
                    predicted_class,
                    audio,
                    sr,
                    audio_type=audio_type
                )
            )

        prediction_session = {
            **prediction,

            "explanation":
                runtime_output["explanation"],

            "counterfactual":
                runtime_output["counterfactual"],

            "class_comparison":
                runtime_output["class_comparison"],

            "decision_boundaries":
                runtime_output["decision_boundaries"],

            "visualization":
                runtime_output["visualization"],

            "spectrogram_analysis":
                spectrogram_analysis
        }

#=====================================================================

        report_entry = {
            "email": user_email,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_type": analysis_type,
            "audio_type": audio_type,
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"]
        }

        save_report(report_entry)

        prediction["analysis_type"] = analysis_type
        prediction["audio_type"] = audio_type
        prediction["timestamp"] = report_entry["timestamp"]

        return jsonify(prediction_session)

    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     return jsonify({"error": str(e)}), 500

    except Exception as e:

        logger.exception("Inference pipeline failed.")

        return jsonify({
            "error": "Internal server error."
        }), 500

    finally:
        if temp_path:
            delete_file(temp_path)


# ============================================================
# CHATBOT ROUTE (RAG + OpenRouter)
# ============================================================

from runtime.chatbot_core import handle_user_query

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        if not request.is_json:

            return jsonify({
                "error": "JSON request required."
            }), 400

        data = request.get_json(silent=True)

        if data is None:

            return jsonify({
                "error": "Invalid JSON."
            }), 400

        message = data.get("message")
        if len(message) > 2000:

            return jsonify({
                "error": "Message too long."
            }), 400

        message = re.sub(
            r"\s+",
            " ",
            message
        ).strip()
        session = data.get("session")

        if not message:
            return jsonify({"error": "Message required"}), 400

        reply_payload = handle_user_query(
            user_query=message,

            acoustic_vector=
                session.get("acoustic_vector")
                if session else None,

            predicted_class=
                session.get("predicted_class")
                if session else None,

            confidence=
                session.get("confidence")
                if session else None,

            explanation_payload=None,

            audio_type=
                session.get("audio_type", "stethoscope")
                if session else "stethoscope",

            direct_llm=True
        )

        reply = reply_payload["response"]

        return jsonify({"reply": reply})

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    except Exception:

        logger.exception("Chat endpoint failed.")

        return jsonify({
            "error": "Internal server error."
        }), 500

# ============================================================
# REPORT HISTORY ROUTE
# ============================================================

# @app.route("/reports", methods=["GET"])
# def get_reports():
#     """Return screening history."""
#     return jsonify(load_reports())

@app.route("/reports", methods=["POST"])

def get_reports():

    data = request.json or {}

    email = data.get("email", "")

    reports = load_reports()

    reports = [

        r

        for r in reports

        if r.get("email") == email

    ]

    return jsonify(reports)



# ============================================================
# EXPLANATORY LAB ROUTE
# ============================================================

@app.route("/explanation_session", methods=["POST"])
def explanation_session():
    """
    Return explanation payload for explanatory lab page.
    """

    try:
        if not request.is_json:

            return jsonify({
                "error": "JSON request required."
            }), 400

        data = request.get_json(silent=True)

        if data is None:

            return jsonify({
                "error": "Invalid JSON."
            }), 400

        acoustic_vector = np.array(
            data["acoustic_vector"]
        )

        predicted_class = data[
            "predicted_class"
        ]

        confidence = float(
            data["confidence"]
        )

        audio_type = data.get(
            "audio_type",
            "stethoscope"
        )

        explanation = generate_explanation(
            acoustic_vector=acoustic_vector,
            predicted_class=predicted_class,
            confidence=confidence,
            audio_type=audio_type
        )

        return jsonify(explanation)

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    except Exception:

        logger.exception("Explanation generation failed.")

        return jsonify({
            "error": "Internal server error."
        }), 500

# ============================================================
# RUN SERVER
# ============================================================


# COMMITTED CHANGE FOR "FAILED TO FETCH ERROR" CAUSED DUE TO DEBUG WAS SET TO TRUE
"""COMMITTMENT DATE : 1/03/2026 23:10 PM"""
# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True
#     )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False
    )