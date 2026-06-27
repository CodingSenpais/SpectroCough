"""
base_paths.py
-------------
Unified centralized path configuration
for SpectroCough Unified Architecture.

Purpose
------------------------------------------------
Single source of truth for:
- API paths
- Runtime paths
- ML pipeline paths
- KB paths
- Model paths
- Frontend paths

Deployment-safe:
------------------------------------------------
- No hardcoded absolute paths
- Portable across systems
- Docker/cloud friendly
"""

from pathlib import Path

# ==========================================================
# ROOT PROJECT DIRECTORY
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# ==========================================================
# CORE DIRECTORIES
# ==========================================================

API_DIR = ROOT_DIR / "api"

RUNTIME_DIR = ROOT_DIR / "runtime"

ML_PIPELINE_DIR = ROOT_DIR / "ml_pipeline"

FRONTEND_DIR = ROOT_DIR / "frontend"

WEB_KB_DIR = ROOT_DIR / "web_kb"

CHATBOT_KB_DIR = ROOT_DIR / "chatbot_kb"

TOOLS_DIR = ROOT_DIR / "tools"

LOGS_DIR = ROOT_DIR / "logs"

CACHE_DIR = ROOT_DIR / "cache"

# ==========================================================
# DATASET DIRECTORIES
# ==========================================================

DATASETS_DIR = (
    ROOT_DIR / "datasets"
)

PANEL1_DATASET_DIR = (
    DATASETS_DIR /
    "panel1_stethoscope"
)

PANEL2_DATASET_DIR = (
    DATASETS_DIR /
    "panel2_microphone"
)
# ==========================================================
# PANEL 1 — STETHOSCOPE PIPELINE
# ==========================================================

PANEL1_DIR = (
    ML_PIPELINE_DIR /
    "panel1_stethoscope"
)

PANEL1_MODEL_DIR = (
    PANEL1_DIR / "models"
)

PANEL1_SCALER_DIR = (
    PANEL1_DIR / "scalers"
)

PANEL1_MODEL_PATH = (
    PANEL1_MODEL_DIR /
    "spectrocough_v1_baseline.keras"
)

PANEL1_SCALER_PATH = (
    PANEL1_SCALER_DIR /
    "acoustic_scaler.pkl"
)

# ==========================================================
# PANEL 2 — MICROPHONE PIPELINE
# ==========================================================

PANEL2_DIR = (
    ML_PIPELINE_DIR /
    "panel2_microphone"
)

PANEL2_MODEL_DIR = (
    PANEL2_DIR / "models"
)

PANEL2_SCALER_DIR = (
    PANEL2_DIR / "scalers"
)

PANEL2_MODEL_PATH = (
    PANEL2_MODEL_DIR /
    "spectrocough_yamnet_fusion.keras"
)

PANEL2_SCALER_PATH = (
    PANEL2_SCALER_DIR /
    "scaler_yamnet.pkl"
)

# ==========================================================
# PANEL 1 WEB KB
# ==========================================================

PANEL1_KB_DIR = (
    WEB_KB_DIR /
    "stethoscope_profiles"
)

PANEL1_REFERENCE_SPEC_DIR = (
    PANEL1_KB_DIR /
    "reference_spectrograms"
)

# ==========================================================
# PANEL 2 WEB KB
# ==========================================================

PANEL2_KB_DIR = (
    WEB_KB_DIR /
    "microphone_profiles"
)

PANEL2_REFERENCE_SPEC_DIR = (
    PANEL2_KB_DIR /
    "reference_spectrograms"
)

# ==========================================================
# FRONTEND ASSETS
# ==========================================================

ASSETS_DIR = FRONTEND_DIR / "assets"

GLB_MODEL_DIR = (
    ASSETS_DIR / "3d_model"
)

VIDEO_ASSET_DIR = (
    ASSETS_DIR / "video_animation"
)

ABOUT_US_ASSET_DIR = (
    ASSETS_DIR / "about_us"
)

HOME_GLASSY_ASSET_DIR = (
    ASSETS_DIR / "home_glassy"
)

# ==========================================================
# REPORTS + TEMP STORAGE
# ==========================================================

REPORT_FILE = API_DIR / "reports.json"

TEMP_UPLOAD_DIR = (
    FRONTEND_DIR / "temp_uploads"
)