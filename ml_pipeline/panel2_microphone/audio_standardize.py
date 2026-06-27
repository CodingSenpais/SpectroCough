"""
audio_standardize.py
--------------------
Microphone-optimized audio standardization for SpectroCough

Design:
- Preserve cough/sneeze characteristics
- Remove low-frequency noise
- Use ENERGY-FOCUSED smart cropping
- Standardize to 5 seconds
"""

import numpy as np
import librosa
import scipy.signal as signal

from ml_pipeline.panel2_microphone.config import (
    TARGET_SAMPLE_RATE,
    TARGET_NUM_SAMPLES,
    FORCE_MONO,
    USE_RMS_NORMALIZATION,
    TARGET_RMS,
)

# ============================================================
# 📥 LOAD AUDIO
# ============================================================

def load_audio(file_path: str):
    y, sr = librosa.load(
        file_path,
        sr=None,
        mono=False
    )
    return y.astype(np.float32), sr


# ============================================================
# 🎧 MONO CONVERSION
# ============================================================

def convert_to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0)


# ============================================================
# 🔥 PRE-EMPHASIS
# ============================================================

def apply_preemphasis(y: np.ndarray) -> np.ndarray:
    return librosa.effects.preemphasis(y, coef=0.97)


# ============================================================
# 🔊 HIGH-PASS FILTER
# ============================================================

def highpass_filter(y: np.ndarray, sr: int, cutoff=100):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist

    b, a = signal.butter(2, norm_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, y)


# ============================================================
# 🔄 RESAMPLING
# ============================================================

def resample_audio(y: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SAMPLE_RATE:
        return y

    return librosa.resample(
        y=y,
        orig_sr=orig_sr,
        target_sr=TARGET_SAMPLE_RATE
    )



# ============================================================
# ✂️ SILENCE TRIMMING
# ============================================================

def trim_silence(y: np.ndarray) -> np.ndarray:
    """
    Remove leading/trailing silence.

    Important for:
    - cough localization
    - cleaner embeddings
    - better temporal consistency
    """

    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=25
    )

    # Safety fallback
    if len(y_trimmed) < 1000:
        return y

    return y_trimmed

def enforce_fixed_length(y: np.ndarray) -> np.ndarray:

    current_len = len(y)

    # -------------------------------
    # Case 1: Exact length
    # -------------------------------
    if current_len == TARGET_NUM_SAMPLES:
        return y

    # -------------------------------
    # Case 2: Short → pad
    # -------------------------------
    if current_len < TARGET_NUM_SAMPLES:
        return np.pad(y, (0, TARGET_NUM_SAMPLES - current_len))

    # -------------------------------
    # Case 3: Long → SMART CROPPING
    # -------------------------------

    # ALWAYS energy crop
    try:
        # Compute energy
        energy = librosa.feature.rms(y=y)[0]

        # Get top 3 peaks (burst candidates)
        top_k = min(3, len(energy))
        peak_indices = np.argsort(energy)[-top_k:]

        # Choose one peak randomly
        chosen_peak = np.random.choice(peak_indices)

        # Convert frame index to sample index
        hop_length = 512  # librosa default
        peak_sample = chosen_peak * hop_length

        # Center crop around peak
        start = peak_sample - TARGET_NUM_SAMPLES // 2

        # Boundary correction
        start = max(0, start)
        end = start + TARGET_NUM_SAMPLES

        if end > current_len:
            end = current_len
            start = end - TARGET_NUM_SAMPLES

        return y[start:end]

    except Exception:
        pass  # fallback to random crop

    # --------------------------------------------------------
    # Fallback:
    # center crop if energy extraction fails
    # --------------------------------------------------------

    start = max(
        0,
        (current_len - TARGET_NUM_SAMPLES) // 2
    )

    end = start + TARGET_NUM_SAMPLES

    return y[start:end]



# ============================================================
# 🔥 MULTI-WINDOW EVENT EXTRACTION
# ============================================================

def extract_multi_windows(
    y: np.ndarray,
    num_windows: int = 3
):
    """
    Extract multiple energetic cough-event windows.

    Purpose:
    ------------------------------------------------
    Instead of relying on a single crop,
    capture multiple cough regions from the audio.

    Returns:
    ------------------------------------------------
    List[np.ndarray]
        multiple standardized windows
    """

    current_len = len(y)

    # --------------------------------------------------------
    # Short audio fallback
    # --------------------------------------------------------

    if current_len <= TARGET_NUM_SAMPLES:

        padded = np.pad(
            y,
            (0, max(0, TARGET_NUM_SAMPLES - current_len))
        )

        return [padded.astype(np.float32)]

    # --------------------------------------------------------
    # Compute RMS energy
    # --------------------------------------------------------

    energy = librosa.feature.rms(
        y=y,
        hop_length=512
    )[0]

    # --------------------------------------------------------
    # Find strongest peaks
    # --------------------------------------------------------

    top_k = min(num_windows, len(energy))

    peak_indices = np.argsort(
        energy
    )[-top_k:]

    # Sort chronologically
    peak_indices = np.sort(
        peak_indices
    )

    windows = []

    # --------------------------------------------------------
    # Extract windows around peaks
    # --------------------------------------------------------

    for peak in peak_indices:

        peak_sample = peak * 512

        start = peak_sample - TARGET_NUM_SAMPLES // 2

        start = max(0, start)

        end = start + TARGET_NUM_SAMPLES

        if end > current_len:
            end = current_len
            start = end - TARGET_NUM_SAMPLES

        segment = y[start:end]

        # ----------------------------------------------------
        # Final safety padding
        # ----------------------------------------------------

        if len(segment) < TARGET_NUM_SAMPLES:

            segment = np.pad(
                segment,
                (
                    0,
                    TARGET_NUM_SAMPLES - len(segment)
                )
            )

        # ----------------------------------------------------
        # RMS normalize each segment
        # ----------------------------------------------------

        if USE_RMS_NORMALIZATION:
            segment = rms_normalize(segment)

        windows.append(
            segment.astype(np.float32)
        )

    return windows

# ============================================================
# 📊 RMS NORMALIZATION
# ============================================================

def rms_normalize(y: np.ndarray, target_rms: float = TARGET_RMS):

    rms = np.sqrt(np.mean(y ** 2))

    if rms < 1e-8:
        return y

    gain = target_rms / rms
    
    y = y * gain

    # Prevent clipping
    y = np.clip(y, -1.0, 1.0)

    return y


# ============================================================
# 🚀 FULL PIPELINE
# ============================================================

def standardize_audio(file_path: str) -> np.ndarray:

    # 1. Load
    y, sr = load_audio(file_path)

    # 2. Mono
    if FORCE_MONO:
        y = convert_to_mono(y)

    # 3. Pre-emphasis
    y = apply_preemphasis(y)

    # 4. High-pass filter
    y = highpass_filter(y, sr)

    # 5. Resample
    y = resample_audio(y, sr)

    # 6. Trim silence
    y = trim_silence(y)

    # 7. Energy-focused crop
    y = enforce_fixed_length(y)

    # 8. RMS normalize
    if USE_RMS_NORMALIZATION:
        y = rms_normalize(y)

    return y.astype(np.float32)