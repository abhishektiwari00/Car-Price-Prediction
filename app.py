from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np

app = Flask(__name__)

FEATURE_COLUMNS = [
    "symboling",
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
]

# Load the .keras model (must exist in same folder as app.py)
model = keras.models.load_model("model.keras", compile=False)


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        columns=FEATURE_COLUMNS,
        prediction=None,
        error=None,
        form_values={},
        api_status="ok",
        feature_count=len(FEATURE_COLUMNS),
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    try:
        values = []
        for col in FEATURE_COLUMNS:
            val = form_values.get(col, "")
            if val is None or val == "":
                val = 0
            values.append(float(val))

        arr = np.array([values], dtype=float)
        pred = model.predict(arr)
        predicted_price = float(pred[0][0])

        return render_template(
            "index.html",
            columns=FEATURE_COLUMNS,
            prediction=round(predicted_price, 2),
            error=None,
            form_values=form_values,
            api_status="ok",
            feature_count=len(FEATURE_COLUMNS),
        )
    except Exception as e:
        return render_template(
            "index.html",
            columns=FEATURE_COLUMNS,
            prediction=None,
            error=str(e),
            form_values=form_values,
            api_status="error",
            feature_count=len(FEATURE_COLUMNS),
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_file": "model.keras",
        "feature_count": len(FEATURE_COLUMNS)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

