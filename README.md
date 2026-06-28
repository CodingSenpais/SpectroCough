# SpectroCough – AI-Powered Mel-Acoustic Fusion For Respiratory Illness Detection & Pre-Screening System

<p align="center">
  <img src="frontend/assets/sc_icon.png" alt="SpectroCough Logo" width="140"/>
</p>

<p align="center">
<b>Hybrid Deep Learning • Explainable AI • Respiratory Acoustic Analysis • Interactive Medical Decision Support</b>
</p>

---

## Overview

**SpectroCough** is a project that performs AI-assisted respiratory disease pre-screening using cough audio recordings. The system supports two independent screening pipelines based on different recording modalities and provides explainable AI outputs to improve transparency and interpretability.

Unlike conventional classification systems that only predict a disease label, SpectroCough combines:

* Hybrid Deep Learning
* Acoustic Feature Engineering
* Explainable AI (XAI)
* Interactive Respiratory Visualizations
* Retrieval-Augmented Generation (RAG)
* AI-powered Clinical Chat Assistant

to generate clinically interpretable respiratory screening reports.

> **Disclaimer:** SpectroCough is intended solely as a pre-screening, research and educational decision-support system. It is **not** a replacement for professional medical diagnosis.

---

# Key Features

## Panel 1 — Digital Stethoscope Analysis

Supported Classes:

* Healthy
* Asthma
* Bronchitis
* COPD
* Pneumonia

Pipeline:

```
Respiratory Audio
        │
Audio Standardization
        │
Acoustic Feature Extraction
        │
Mel Spectrogram Generation
        │
CNN + Dense Neural Network Fusion
        │
Prediction
        │
Explainable AI Runtime
```

---

## Panel 2 — Microphone Analysis

Supported Classes:

* COVID-19 Cough
* Healthy Cough
* Non-Cough (Sneezing)

Pipeline:

```
Respiratory Audio
        │
Audio Standardization
        │
Acoustic Features
        │
YAMNet Embeddings
        │
Hybrid Deep Learning Model
        │
Prediction
        │
Explainable AI Runtime
```

---

# Explainable AI Components

After every prediction, SpectroCough automatically generates:

* Acoustic Biomarker Explanation
* Disease Fingerprint
* Counterfactual Analysis
* Decision Boundary Interpretation
* Class Comparison
* Spectrogram Interpretation
* Feature Importance Analysis
* Visual Acoustic Comparison

These explanations are generated from structured knowledge bases rather than hardcoded text.

---

# AI Chat Assistant

SpectroCough includes **Coughie**, an AI-powered assistant capable of answering questions related to:

* Respiratory diseases
* Acoustic biomarkers
* Spectrogram interpretation
* Prediction reasoning
* System workflow
* Feature engineering
* Model limitations
* Screening guidance

The assistant uses:

* Retrieval-Augmented Generation (RAG)
* Structured Medical Knowledge Bases
* Large Language Model integration

to produce context-aware explanations.

---

# Machine Learning Architecture

## Panel 1

* CNN-based Spectrogram Learning
* Sound Acoustic Features
* CNN+DNN Fusion Network
* Softmax Classification

---

## Panel 2

* CNN-based Spectrogram Learning and YAMNet Embeddings
* Acoustic Feature Engineering
* CNN+DNN Fusion Network
* Softmax Classification

---

# Acoustic Features

The system extracts multiple respiratory acoustic biomarkers, including:

* MFCC
* RMS Energy
* Zero Crossing Rate
* Spectral Centroid
* Spectral Bandwidth
* Spectral Contrast
* Spectral Rolloff
* Chroma Features
* Mel Spectrogram
* Log Mel Spectrogram
* Spectral Flatness
* Spectral Flux
* Short-Time Energy

---

# Technology Stack

### Backend

* Python
* Flask
* TensorFlow
* Keras
* TensorFlow Hub
* Librosa
* NumPy
* Scikit-learn

### Frontend

* HTML5
* CSS3
* JavaScript
* Glassmorphism UI
* Font Awesome
* Google Model Viewer

### AI Components

* CNN+DNN Model
* YAMNet Embeddings
* Explainable AI
* Retrieval-Augmented Generation (RAG)
* Acoustic Feature Engineering

---

# Project Structure

```
SpectroCough/

├── api/
├── ml_pipeline/
│   ├── panel1_stethoscope/
│   └── panel2_microphone/
├── runtime/
├── chatbot_kb/
├── web_kb/
├── frontend/
├── dev_tools/
└── datasets/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/SpectroCough.git
```

Create a virtual environment:

```bash
conda create -n spectrocough python=3.13.5
conda activate spectrocough
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=YOUR_API_KEY
GROQ_MODEL=llama-3.3-70b-versatile
```

Run the application:

```bash
python -m api.api_server
```

Open:

```
http://127.0.0.1:5000
```

---

# Repository Contents

The repository includes:

* Source code
* Trained AI models
* Knowledge bases
* Runtime inference pipeline
* Frontend interface
* Explainability engine
* Report generation
* AI chatbot
* Utility scripts

Datasets, runtime logs, temporary uploads, and generated reports are excluded from version control.

---

# Current Limitations

* Designed for respiratory pre-screening only.
* Not intended for emergency diagnosis.
* Performance depends on recording quality.
* Background noise may affect predictions.
* The system does not currently perform automatic cough/non-cough validation before inference.

---

# Future Enhancements

* Live microphone recording
* Cloud deployment
* Multi-language chatbot
* Additional respiratory disease classes
* Clinical dashboard
* Patient history analytics
* Audio quality assessment

---

# License

This repository is intended for academic and educational purposes.

---

# Authors

**SpectroCough Development Team**

---

## Copyright

© 2026 CodingSenpais. All Rights Reserved.

Developed collaboratively by the SpectroCough Development Team under the CodingSenpais project.

This project, including its source code, machine learning models, documentation, design, and associated assets, is the intellectual property of the SpectroCough Development Team (CodingSenpais). Unauthorized copying, redistribution, modification, or commercial use without prior written permission from the copyright holders is prohibited.


<p align="center">
<b>SpectroCough</b><br>
AI-Assisted Respiratory Disease Pre-Screening through Explainable Acoustic Intelligence
</p>
