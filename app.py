import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained SVM model once, at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svm_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature order MUST match the order the model was trained on
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

# Encoding maps used to turn friendly form inputs into the numbers the model expects.
# NOTE: These must match the encoding used when the model was originally trained.
# Adjust them here if your training pipeline used a different mapping.
ENCODINGS = {
    "gender": {"Male": 0, "Female": 1},
    "internet_access": {"No": 0, "Yes": 1},
    "extracurricular": {"No": 0, "Yes": 1},
    "parent_education": {
        "High School": 0,
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3,
    },
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build the feature vector in the correct order
        row = []
        for feature in FEATURE_ORDER:
            value = data.get(feature)
            if value is None or value == "":
                return jsonify({"error": f"Missing value for '{feature}'"}), 400

            if feature in ENCODINGS:
                mapping = ENCODINGS[feature]
                if value not in mapping:
                    return jsonify({"error": f"Invalid value '{value}' for '{feature}'"}), 400
                row.append(mapping[value])
            else:
                row.append(float(value))

        X = pd.DataFrame([row], columns=FEATURE_ORDER)

        prediction = model.predict(X)[0]

        # probability=False on this model, so use decision_function as a
        # confidence proxy instead of predict_proba
        try:
            score = float(model.decision_function(X)[0])
        except Exception:
            score = None

        return jsonify({
            "prediction": str(prediction),
            "decision_score": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
