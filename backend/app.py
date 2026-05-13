"""
Lahja — Arabic Dialect Detection Backend
Full DSP version — matches optimized notebook (318 features, SVM + scaler)
"""

import os, io, base64, traceback, warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import joblib

try:
    from scipy.stats import skew, kurtosis
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠️  scipy not found — install with: pip install scipy")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Keys ─────────────────────────────────────────────────────────
# open backend/app.py and manually replace the two hardcoded lines with:
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY",     "")

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "dialect_model.joblib")
ENCODER_PATH  = os.path.join(BASE_DIR, "label_encoder.joblib")
SCALER_PATH   = os.path.join(BASE_DIR, "feature_scaler.joblib")
FRONTEND_DIR  = os.path.join(BASE_DIR, "..", "frontend")

SAMPLE_RATE = 16_000
N_MFCC      = 20
N_LPC       = 12
PORT        = 5000

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ── Load artifacts ────────────────────────────────────────────────
print("Loading model artifacts...")
try:
    model         = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"✅ Model loaded — classes: {list(label_encoder.classes_)}")
    print(f"   n_features_in: {getattr(model, 'n_features_in_', '?')}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = label_encoder = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded")
except Exception as e:
    print(f"⚠️  Scaler not found ({e}) — continuing without it")
    scaler = None


# ════════════════════════════════════════════════════════════════
#  DSP HELPERS
# ════════════════════════════════════════════════════════════════

def _lpc_coefficients(audio, order=N_LPC):
    frame_length, hop_length = 512, 256
    try:
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    except Exception:
        return np.zeros((order, 1))
    window = np.hamming(frame_length)
    frames = frames * window[:, np.newaxis]
    lpc_mat = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        r = np.correlate(frame, frame, mode='full')
        r = r[len(r)//2 : len(r)//2 + order + 1]
        if r[0] < 1e-10:
            lpc_mat.append(np.zeros(order))
            continue
        R = np.array([[r[abs(ii-jj)] for jj in range(order)] for ii in range(order)])
        try:
            a = np.linalg.solve(R, r[1:order+1])
        except np.linalg.LinAlgError:
            a = np.zeros(order)
        lpc_mat.append(a)
    return np.array(lpc_mat).T if lpc_mat else np.zeros((order, 1))


def _energy_periodicity(audio, hop=256):
    try:
        rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
        rms = rms / (np.max(rms) + 1e-8)
        acf = np.correlate(rms, rms, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-8)
        lags = acf[10:51] if len(acf) > 51 else acf
        return float(np.max(lags)), float(np.mean(lags)), float(np.std(lags))
    except Exception:
        return 0.0, 0.0, 0.0


def _voiced_fraction(audio, sr):
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=70, fmax=400, sr=sr,
            frame_length=1024, hop_length=256)
        if voiced_flag is None or len(voiced_flag) == 0:
            return 0.5, 0.0, 0.0
        voiced_bool = np.asarray(voiced_flag, dtype=bool)
        vf = float(np.mean(voiced_bool))
        if f0 is None:
            return vf, 0.0, 0.0
        f0_arr = np.asarray(f0, dtype=float)
        f0v    = f0_arr[voiced_bool]
        f0v    = f0v[~np.isnan(f0v)]
        f0_mean = float(np.mean(f0v)) if len(f0v) > 0 else 0.0
        f0_std  = float(np.std(f0v))  if len(f0v) > 0 else 0.0
        return vf, f0_mean, f0_std
    except Exception:
        traceback.print_exc()
        return 0.5, 0.0, 0.0


def extract_features(audio, sr=SAMPLE_RATE):
    """Returns (vector, vis) or (None, None) on failure."""
    try:
        if len(audio) < sr * 0.5:
            print(f"  [skip] audio too short: {len(audio)/sr:.2f}s")
            return None, None

        # 1. CMVN MFCCs
        mfccs         = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mu       = np.mean(mfccs, axis=1, keepdims=True)
        mfcc_sig      = np.std( mfccs, axis=1, keepdims=True) + 1e-8
        mfccs_n       = (mfccs - mfcc_mu) / mfcc_sig

        mfcc_mean = np.mean(mfccs_n, axis=1)
        mfcc_std  = np.std( mfccs_n, axis=1)
        mfcc_p25  = np.percentile(mfccs_n, 25, axis=1)
        mfcc_p75  = np.percentile(mfccs_n, 75, axis=1)

        if SCIPY_OK:
            mfcc_skew = skew(mfccs_n, axis=1)
            mfcc_kurt = kurtosis(mfccs_n, axis=1)
        else:
            mfcc_skew = np.zeros(N_MFCC)
            mfcc_kurt = np.zeros(N_MFCC)

        # 2. Delta & Delta²
        delta    = librosa.feature.delta(mfccs_n)
        delta2   = librosa.feature.delta(mfccs_n, order=2)
        delta_n  = delta  / (np.std(delta,  axis=1, keepdims=True) + 1e-8)
        delta2_n = delta2 / (np.std(delta2, axis=1, keepdims=True) + 1e-8)

        # 3. LPC
        lpc_mat  = _lpc_coefficients(audio)
        lpc_mean = np.mean(lpc_mat, axis=1)
        lpc_std  = np.std( lpc_mat, axis=1)

        # 4. Chroma
        chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std( chroma, axis=1)

        # 5. Spectral contrast
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std  = np.std( contrast, axis=1)

        # 6. Spectral shape
        centroid  = librosa.feature.spectral_centroid( y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        rolloff   = librosa.feature.spectral_rolloff(  y=audio, sr=sr)[0]
        flatness  = librosa.feature.spectral_flatness( y=audio)[0]
        zcr       = librosa.feature.zero_crossing_rate(audio)[0]

        spectral = np.array([
            np.mean(centroid),  np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff),   np.std(rolloff),
            np.mean(flatness),  np.std(flatness),
            np.mean(zcr),       np.std(zcr),
        ])

        # 7. Prosodic
        vf, f0_mean, f0_std        = _voiced_fraction(audio, sr)
        rhy_max, rhy_mean, rhy_std = _energy_periodicity(audio)
        prosodic = np.array([vf, f0_mean, f0_std, rhy_max, rhy_mean, rhy_std])

        vector = np.concatenate([
            mfcc_mean, mfcc_std,
            mfcc_p25,  mfcc_p75,
            mfcc_skew, mfcc_kurt,
            np.mean(delta_n,  axis=1), np.std(delta_n,  axis=1),
            np.mean(delta2_n, axis=1), np.std(delta2_n, axis=1),
            lpc_mean, lpc_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            spectral,
            prosodic,
        ]).astype(np.float32)

        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  [features] vector shape: {vector.shape}")

        vis = {
            "mfcc":           [round(float(v), 2) for v in np.mean(mfccs, axis=1)],   # raw — CMVN mean is always ~0
            "delta":          [round(float(v), 2) for v in np.mean(delta,  axis=1)],  # raw delta
            "chroma":         [round(float(v), 3) for v in chroma_mean],
            "contrast":       [round(float(v), 2) for v in contrast_mean],
            "centroid":       round(float(np.mean(centroid)),  1),
            "bandwidth":      round(float(np.mean(bandwidth)), 1),
            "rolloff":        round(float(np.mean(rolloff)),   1),
            "zcr":            round(float(np.mean(zcr)),       5),
            "f0_mean":        round(f0_mean, 1),
            "f0_std":         round(f0_std,  1),
            # ── Prosodic (previously missing from vis) ──────────
            "voiced_frac":    round(vf,       3),
            "rhy_max":        round(rhy_max,  3),
            "rhy_mean":       round(rhy_mean, 3),
            "rhy_std":        round(rhy_std,  3),
        }

        return vector, vis

    except Exception:
        print("[extract_features] EXCEPTION:")
        traceback.print_exc()
        return None, None


def classify(vector):
    v = vector.reshape(1, -1)
    if scaler is not None:
        v = scaler.transform(v)
    pred  = str(model.predict(v)[0])
    probs, conf = {}, None
    if hasattr(model, "predict_proba"):
        p     = model.predict_proba(v)[0]
        probs = {str(cls): round(float(pv)*100, 1)
                 for cls, pv in zip(label_encoder.classes_, p)}
        conf  = round(float(max(p))*100, 1)
    return pred, conf, probs


def make_spectrogram_b64(audio, sr, title=""):
    try:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor='#0a0a0f')
        ax.set_facecolor('#111118')
        S    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img  = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                        ax=ax, cmap='magma')
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='#7a7a9a', labelcolor='#7a7a9a')
        ax.set_title(title, color='#c8a96e', fontsize=10, pad=6)
        ax.tick_params(colors='#7a7a9a', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a3a')
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        traceback.print_exc()
        return ""


def audio_to_wav_b64(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV', subtype='PCM_16')
    return base64.b64encode(buf.getvalue()).decode()


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/config")
def config():
    return jsonify({
        "elevenlabs_key": ELEVENLABS_API_KEY,
        "gemini_key":     GEMINI_API_KEY,
    })

@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "scaler_loaded": scaler is not None,
        "scipy_ok":      SCIPY_OK,
        "classes":       list(label_encoder.classes_) if label_encoder else [],
        "n_features":    getattr(model, "n_features_in_", None),
    })

@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded — check server logs"}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' field in request"}), 400

    audio_bytes = request.files["audio"].read()
    if not audio_bytes:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        print(f"[detect] loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    except Exception as e:
        return jsonify({"error": f"Cannot load audio: {e}"}), 422

    vector, vis = extract_features(audio, sr)
    if vector is None:
        return jsonify({"error": "Feature extraction failed — check Flask terminal for traceback"}), 422

    # Dimension check — helpful error if model/features mismatch
    expected = getattr(model, "n_features_in_", None)
    if expected and vector.shape[0] != expected:
        return jsonify({
            "error": f"Feature dimension mismatch: got {vector.shape[0]}, model expects {expected}. "
                     f"Make sure dialect_model.joblib was trained with the optimized notebook."
        }), 422

    try:
        dialect, confidence, probabilities = classify(vector)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Classification failed: {e}"}), 500

    spec_b64 = make_spectrogram_b64(audio, sr, title=f"Mel Spectrogram · {dialect}")

    return jsonify({
        "dialect":       dialect,
        "confidence":    confidence,
        "probabilities": probabilities,
        "features":      vis,
        "spectrogram":   spec_b64,
    })


@app.route("/mix", methods=["POST"])
def mix():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "audio1" not in request.files or "audio2" not in request.files:
        return jsonify({"error": "Need audio1 and audio2 fields"}), 400

    weight = float(request.form.get("weight", 0.5))
    weight = max(0.0, min(1.0, weight))

    try:
        audio1, _ = librosa.load(
            io.BytesIO(request.files["audio1"].read()), sr=SAMPLE_RATE, mono=True)
        audio2, _ = librosa.load(
            io.BytesIO(request.files["audio2"].read()), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return jsonify({"error": f"Cannot load audio: {e}"}), 422

    n     = max(len(audio1), len(audio2))
    a1    = np.pad(audio1, (0, n - len(audio1)))
    a2    = np.pad(audio2, (0, n - len(audio2)))
    mixed = (1 - weight) * a1 + weight * a2

    v1, _ = extract_features(audio1, SAMPLE_RATE)
    v2, _ = extract_features(audio2, SAMPLE_RATE)
    vm, _ = extract_features(mixed,  SAMPLE_RATE)

    if v1 is None or v2 is None or vm is None:
        return jsonify({"error": "Feature extraction failed on one or more files"}), 422

    try:
        d1, c1, p1 = classify(v1)
        d2, c2, p2 = classify(v2)
        dm, cm, pm = classify(vm)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Classification failed: {e}"}), 500

    return jsonify({
        "file1":       {"dialect": d1, "confidence": c1, "probabilities": p1},
        "file2":       {"dialect": d2, "confidence": c2, "probabilities": p2},
        "mixed":       {"dialect": dm, "confidence": cm, "probabilities": pm},
        "weight":      weight,
        "spectrogram": make_spectrogram_b64(
            mixed, SAMPLE_RATE,
            title=f"Mixed — {int((1-weight)*100)}% {d1} + {int(weight*100)}% {d2}"
        ),
        "audio": audio_to_wav_b64(mixed, SAMPLE_RATE),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)