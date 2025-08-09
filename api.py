from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

# -----------------------------
# Load the OVERSAMPLING pipeline (TF-IDF -> RandomOverSampler -> LogisticRegression)
# Train & save this once in your notebook as:
#   joblib.dump(pipe_over, "Models/sentiment_oversampled_pipeline.pkl")
# -----------------------------
MODEL_PATH = "Models/sentiment_oversampled_pipeline.pkl"
pipe_over = joblib.load(MODEL_PATH)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return render_template("landing.html")

@app.get("/health")
def health():
    return {"ok": True, "model": "oversampling-pipeline"}

@app.post("/predict")
def predict():
    """
    POST /predict
    - JSON: {"text": "..."}
    - multipart/form-data: file=<CSV with 'Sentence' column>
    Returns:
      - For text: {ok, label, proba}
      - For CSV: CSV file + X-Graph header (base64 PNG of class distribution)
    """
    try:
        # ---- CSV branch ----
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file)
            if "Sentence" not in df.columns:
                return jsonify({"ok": False, "error": "CSV must include a 'Sentence' column."}), 400

            # Predict using the pipeline directly (handles TF-IDF internally)
            preds = pipe_over.predict(df["Sentence"].astype(str).tolist())
            labels = ["Positive" if p == 1 else "Negative" for p in preds]
            df["Predicted"] = labels

            # Pie chart of distribution
            plt.figure(figsize=(5,5))
            ax = pd.Series(labels).value_counts().plot(kind="pie", autopct="%1.1f%%")
            ax.set_ylabel("")
            img = BytesIO()
            plt.tight_layout()
            plt.savefig(img, format="png")
            plt.close()
            img.seek(0)

            # Return CSV + base64 graph header
            out = BytesIO()
            df.to_csv(out, index=False)
            out.seek(0)
            resp = send_file(out, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")
            resp.headers["X-Graph"] = base64.b64encode(img.getvalue()).decode("utf-8")
            return resp

        # ---- single text branch ----
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Provide non-empty 'text'."}), 400

        proba = None
        if hasattr(pipe_over, "predict_proba"):
            probs = pipe_over.predict_proba([text])[0]
            pred = int(probs.argmax())
            proba = float(probs[pred])
        else:
            pred = int(pipe_over.predict([text])[0])

        label = "Positive" if pred == 1 else "Negative"
        return jsonify({"ok": True, "label": label, "proba": proba})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
