<div align="center">

# 🎙️ لهجة · Lahja
### Arabic Dialect AI — Detection · Transcription · Translation · TTS

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Detects Arabic dialect from voice input, transcribes it, translates it to another dialect, and speaks it back — in real time.*

</div>

---

## 📸 Screenshots

> *(Add your screenshots here)*

| Dialect Detection | Feature Visualization |
|:-:|:-:|
| ![Detection UI](screenshots/detection.png) | ![Feature Charts](screenshots/features.png) |

| Full Pipeline | Dialect Mixer |
|:-:|:-:|
| ![Pipeline](screenshots/pipeline.png) | ![Mixer](screenshots/mixer.png) |

---

## 🗂️ Project Structure

```
lahja/
├── backend/
│   ├── app.py                    ← Flask API server (4 endpoints)
│   ├── dialect_model.joblib      ← Trained SVM classifier
│   ├── label_encoder.joblib      ← Maps class indices → dialect names
│   ├── feature_scaler.joblib     ← StandardScaler (must be applied before predict)
│   ├── feature_names.joblib      ← 318 feature names
│   ├── feature_stats.joblib      ← Per-dialect feature stats for visualization
│   └── requirements.txt
├── frontend/
│   └── index.html                ← Single-page app (vanilla JS + CSS)
├── task5DS_modified.ipynb        ← Training notebook (Google Colab)
├── .env.example                  ← API key template — copy to .env
├── .gitignore
└── README.md
```

---

## 🤖 The ML Model

### Dataset
- **Source:** [`ArabicSpeech/ADI17`](https://huggingface.co/datasets/ArabicSpeech/ADI17) on HuggingFace
- **Size:** 800 audio clips × 4 dialects = **3,200 total files**
- **Dialects:** Egyptian (EGY) · Levantine/Syrian (SYR) · Emirati/Gulf (UAE) · Moroccan (MOR)

### Feature Engineering — 318 features total

| Feature Group | Count | What it captures |
|---|---|---|
| MFCC (CMVN-normalised) | 120 | Vocal tract shape / vowel identity |
| Delta MFCC | 40 | Rate of spectral change |
| Delta² MFCC | 40 | Spectral acceleration |
| LPC (order 12) | 24 | Formant proxies (F1/F2) |
| Chroma | 24 | Tonal / pitch-class content |
| Spectral Contrast | 14 | Tonal vs noise energy per band |
| Spectral Shape | 10 | Centroid, bandwidth, rolloff, flatness, ZCR |
| Prosodic | 6 | F0 mean/std, voiced fraction, rhythm |

Each feature is summarised with: **mean, std, p25, p75, skew, kurtosis**.

### Model
- **Algorithm:** SVM with RBF kernel (`C=10, gamma='scale'`)
- **Why SVM over Random Forest:** MFCC features are correlated — SVM handles correlated high-dimensional features better than tree-based splits
- **Preprocessing:** CMVN normalisation on MFCCs + `StandardScaler` on full feature vector
- **Split:** 80/20 stratified train/test

---

## 🔌 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` |
| `GET` | `/health` | Model load status check |
| `POST` | `/detect` | Upload audio → dialect + confidence + features + spectrogram |
| `POST` | `/mix` | Upload 2 audio files + weight → classify the mix, return mixed audio |
| `GET` | `/config` | Returns API keys to frontend (see [Security](#-security)) |

### `/detect` — Request / Response

```bash
curl -X POST http://localhost:5000/detect \
  -F "audio=@sample.wav"
```

```json
{
  "dialect": "egyptian",
  "confidence": 87.3,
  "probabilities": {
    "egyptian": 87.3,
    "levant": 6.1,
    "emarit": 4.2,
    "moroccan": 2.4
  },
  "features": {
    "mfcc": [...],
    "chroma": [...],
    "centroid": 1527.4,
    "f0_mean": 231.1,
    "voiced_frac": 0.654,
    "rhy_max": 0.821,
    ...
  },
  "spectrogram": "<base64-encoded PNG>"
}
```

---

## 🚀 Full Speech Pipeline

```
Voice Input
    ↓  /detect  →  SVM (318 DSP features)
Dialect Detected
    ↓  Whisper base  →  word-level timestamps
Arabic Transcription  (words highlight as audio plays)
    ↓  Gemini 2.5 Flash
Dialect-to-Dialect Translation
    ↓  ElevenLabs eleven_multilingual_v2
TTS Audio in Target Dialect  (translated words reveal as TTS plays)
```

---

## ⚙️ Setup & Run

### 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/lahja.git
cd lahja
```

### 2 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4 — Set up API keys
```bash
cp .env.example .env
# then open .env and fill in your keys:
# ELEVENLABS_API_KEY=...
# GEMINI_API_KEY=...
```

### 5 — Run the backend
```bash
cd backend
python app.py
```

### 6 — Open the app
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🔑 API Keys You Need

| API | Free Tier | Get it |
|---|---|---|
| **ElevenLabs** | 10,000 chars/month | [elevenlabs.io](https://elevenlabs.io/app/settings/api-keys) |
| **Google Gemini** | Free with rate limits | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

> **Note:** Whisper runs locally — no API key needed.

---

## 🔒 Security

API keys are loaded from environment variables, never hardcoded:

```python
# backend/app.py
from dotenv import load_dotenv
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY",     "")
```

The `/config` endpoint exposes keys to the frontend so the browser can call ElevenLabs and Gemini directly. This is acceptable for a local/demo deployment. **For production:**
- Move all ElevenLabs and Gemini calls to the backend
- Remove the `/config` endpoint entirely
- Add rate limiting and authentication

---

## 🏋️ Retrain the Model

Open `task5DS_modified.ipynb` in Google Colab:

1. **Cell 03** — downloads the dataset from HuggingFace (needs Google Drive)
2. **Cell 09** — set config constants
3. **Cell 12** — feature extraction functions (do not modify)
4. **Cell 14** — builds the feature matrix (`X`: 3200×318)
5. **Cell 18** — 80/20 train/test split
6. **Cell 20** — StandardScaler + LabelEncoder
7. **Cell 22** — trains SVM
8. **Cell 24** — evaluates (accuracy + confusion matrix)
9. **Cell 28** — saves 5 `.joblib` artifacts
10. **Cell 30** — downloads artifacts → place in `backend/`

---

## 📊 Model Explainability

The notebook generates:
- **Violin plots** — feature distributions per dialect (Cell 16)
- **Feature importance** — Gini importance via Random Forest (Cell 26)
- **Dialect fingerprint charts** — averaged features per dialect (Cell 06)
- **Best representative files** — top 4 files per dialect on discriminative features (Cell 07)

Key findings:
- **Spectral Contrast** is the most discriminative feature group (7 of top 8 features)
- **Rhythm periodicity** has the highest single-feature F-score (F=211)
- Egyptian and Levantine cluster as "dark" dialects (low centroid); Moroccan and Emirati as "bright"
- Voiced fraction separates dialects clearly: Levantine 70.8% vs Egyptian 58.2%

---

## 🗃️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn SVM, librosa, scipy |
| Backend | Python 3.10+, Flask, Flask-CORS |
| Transcription | OpenAI Whisper (local, `base` model) |
| Translation | Google Gemini 2.5 Flash |
| TTS | ElevenLabs `eleven_multilingual_v2` |
| Frontend | Vanilla JS, CSS custom properties, Web Audio API |
| Training | Google Colab, HuggingFace Datasets |

---

## 👥 Team

> *(Add your team members here)*

| Name | Role |
|---|---|
| — | ML / Data Science |
| — | Backend |
| — | Frontend |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ☕ and librosa
</div>
