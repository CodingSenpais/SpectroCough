import numpy as np
import librosa

from ml_pipeline.panel2_microphone.config import (
    TARGET_SAMPLE_RATE,
    FFT_SIZE,
    HOP_LENGTH,
    N_MFCC
)

def extract_acoustic_features(y):

    features = []

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta, axis=1))

    delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(delta2, axis=1))

    rms = librosa.feature.rms(
        y=y,
        frame_length=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.append(np.mean(rms))
    features.append(np.std(rms))
    features.append(np.max(rms))

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.append(np.mean(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.append(np.mean(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))

    contrast = librosa.feature.spectral_contrast(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH
    )

    features.extend(np.mean(contrast, axis=1))

    harmonic, percussive = librosa.effects.hpss(y)

    features.append(np.mean(harmonic))
    features.append(np.mean(percussive))

    flatness = librosa.feature.spectral_flatness(y=y)

    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    S = np.abs(librosa.stft(y))

    S_norm = S / (
        np.sum(S, axis=0, keepdims=True)
        + 1e-8
    )

    entropy = -np.sum(
        S_norm * np.log(S_norm + 1e-8),
        axis=0
    )

    features.append(np.mean(entropy))

    return np.array(
        features,
        dtype=np.float32
    )