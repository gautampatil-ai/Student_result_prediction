import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load SVM Model
MODEL_PATH = "svm_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Feature list dynamically extracted from model or fallback
FEATURE_NAMES = [
    "gender",
    "age",
    "study_hours_per_week",
    "attendance_rate",
    "parent_education",
    "internet_access",
    "extracurricular",
    "previous_score",
    "final_score"
]

if model and hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)

# HTML Template with Glassmorphic UI & Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Student Outcome Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #090d16 0%, #111827 50%, #1f1435 100%);
            --glass-bg: rgba(255, 255, 255, 0.04);
            --glass-border: rgba(255, 255, 255, 0.1);
            --accent-glow: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --accent-hover: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Ambient Glowing Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            pointer-events: none;
            animation: float 14s infinite alternate ease-in-out;
        }
        .orb-1 { width: 380px; height: 380px; background: rgba(59, 130, 246, 0.2); top: -10%; left: -5%; }
        .orb-2 { width: 420px; height: 420px; background: rgba(139, 92, 246, 0.2); bottom: -10%; right: -5%; animation-delay: -7s; }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(30px, 40px) scale(1.08); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 880px;
            background: var(--glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Result Animation Box */
        .result-box {
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 2rem;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .result-box.yes {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .result-box.no {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        @keyframes popIn {
            0% { transform: scale(0.85); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .result-box.yes .result-title { color: #34d399; }
        .result-box.no .result-title { color: #f87171; }

        .result-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.82rem;
            color: var(--text-muted);
            text-transform: capitalize;
            font-weight: 500;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: #8b5cf6;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
            background: rgba(15, 23, 42, 0.85);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: var(--accent-glow);
            color: white;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        .btn-submit:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Spinner Loading Effect */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 640px) {
            .container { padding: 1.5rem; }
            .header h1 { font-size: 1.75rem; }
            .form-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <div class="header">
            <h1>SVM Classification Predictor</h1>
            <p>Input student attributes below to run the classification model</p>
        </div>

        {% if prediction %}
        <div class="result-box {{ 'yes' if prediction == 'Yes' else 'no' }}">
            <div class="result-title">Prediction Result</div>
            <div class="result-value">{{ prediction }}</div>
        </div>
        {% endif %}

        <form method="POST" action="/predict" id="predForm">
            <div class="form-grid">
                {% for feature in features %}
                <div class="input-group">
                    <label for="{{ loop.index0 }}">{{ feature.replace('_', ' ') }}</label>
                    <input 
                        type="number" 
                        step="any" 
                        id="{{ loop.index0 }}" 
                        name="{{ feature }}" 
                        placeholder="Enter value" 
                        value="{{ form_data.get(feature, '') }}"
                        required
                    >
                </div>
                {% endfor %}

                <button type="submit" class="btn-submit" id="submitBtn">
                    <span id="btnText">Run Prediction</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </div>
        </form>
    </div>

    <script>
        document.getElementById('predForm').addEventListener('submit', function() {
            document.getElementById('btnText').innerText = "Analyzing...";
            document.getElementById('btnSpinner').style.display = "inline-block";
            document.getElementById('submitBtn').style.opacity = "0.85";
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURE_NAMES, 
        prediction=None, 
        form_data={}
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model file missing or failed to load.", 500

    try:
        input_data = []
        form_data = {}
        for feature in FEATURE_NAMES:
            val = float(request.form.get(feature, 0))
            form_data[feature] = request.form.get(feature, '')
            input_data.append(val)

        # Convert to 2D numpy array for prediction
        final_input = np.array([input_data])
        raw_prediction = model.predict(final_input)[0]

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            prediction=str(raw_prediction),
            form_data=form_data
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
