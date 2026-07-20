"""
Flask deployment app for a pre-trained scikit-learn SVM (SVC) model.
Designed for deployment on Render (or any WSGI host via gunicorn).

The model expects 9 features, in this order:
gender, age, study_hours_per_week, attendance_rate, parent_education,
internet_access, extracurricular, previous_score, final_score

NOTE ON CATEGORICAL ENCODING
-----------------------------
The pickle only stores the fitted SVC estimator, not the encoders used
on categorical columns during training. The ENCODINGS map below is a
reasonable default — update it to match your original training
pipeline exactly if predictions look off.
"""

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "svm_model.pkl"

FEATURE_ORDER = [
    "gender",
    "age",
    "study_hours_per_week",
    "attendance_rate",
    "parent_education",
    "internet_access",
    "extracurricular",
    "previous_score",
    "final_score",
]

ENCODINGS = {
    "gender": {"Female": 0, "Male": 1},
    "internet_access": {"No": 0, "Yes": 1},
    "extracurricular": {"No": 0, "Yes": 1},
    "parent_education": {
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
    },
}

# Load the model once at startup rather than per-request.
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)


def encode_payload(payload: dict) -> pd.DataFrame:
    """Turn raw form values into the numeric row the model expects."""
    row = {}
    for key in FEATURE_ORDER:
        value = payload.get(key)
        if value is None:
            raise ValueError(f"Missing field: {key}")
        if key in ENCODINGS:
            mapping = ENCODINGS[key]
            if value not in mapping:
                raise ValueError(f"Invalid value for {key}: {value}")
            row[key] = mapping[value]
        else:
            row[key] = float(value)
    return pd.DataFrame([row])[FEATURE_ORDER]


@app.route("/")
def index():
    return render_template("index.html", encodings=ENCODINGS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        input_df = encode_payload(payload)
        prediction = MODEL.predict(input_df)[0]
        return jsonify({"ok": True, "prediction": str(prediction)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local dev only — Render runs this via gunicorn (see Procfile).
    app.run(host="0.0.0.0", port=5000, debug=True)
