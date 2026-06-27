"""
file_handler.py
---------------
Temporary file upload handling for SpectroCough API.

Responsibilities:
- Save uploaded audio file safely
- Generate unique file names
- Validate file type
- Delete file after inference
"""

import os
import uuid
from werkzeug.utils import secure_filename

from runtime.base_paths import (
    TEMP_UPLOAD_DIR
)

# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = str(
    TEMP_UPLOAD_DIR
)

# Allowed audio formats
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "m4a"}

MAX_AUDIO_SIZE = 15 * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a"
}

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has valid extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# SAVE FILE
# ============================================================

def save_uploaded_file(file):
    """
    Save uploaded file with unique name.

    Parameters
    ----------
    file : werkzeug FileStorage object

    Returns
    -------
    str : path to saved file
    """

    if (
        file is None
        or not hasattr(file, "filename")
        or file.filename == ""
    ):
        raise ValueError(
            "No file uploaded"
        )

    if not allowed_file(file.filename):
        raise ValueError("Unsupported audio format")

    if file.mimetype not in ALLOWED_MIME_TYPES:

        raise ValueError(
            "Unsupported MIME type."
        )

    # Generate unique file name
    filename = secure_filename(file.filename)

    extension = filename.rsplit(".",1)[1].lower()

    unique_name = (
        f"{uuid.uuid4()}."
        f"{extension}"
    )
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.seek(0, os.SEEK_END)

        size = file.tell()

        file.seek(0)

        if size > MAX_AUDIO_SIZE:

            raise ValueError(
                "Audio exceeds maximum size."
            )

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        file.save(file_path)

    except Exception as e:

        raise RuntimeError(
            f"Failed to save upload: {e}"
        )

    return file_path


# ============================================================
# DELETE FILE
# ============================================================

def delete_file(file_path: str):
    """
    Delete temporary file safely.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass