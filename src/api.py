from flask import Flask, jsonify, request
import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.data_utils import add_engineered_features

app = Flask(__name__)

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
feature_columns = model_package["feature_columns"]


def generate_alert(probability: float) -> str:
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Predictive maintenance API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()

    required_fields = [
        "type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]

    missing = [field for field in required_fields if field not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = pd.DataFrame([payload])
    input_df = add_engineered_features(input_df)
    X_input = input_df[feature_columns]

    probability = float(model.predict_proba(X_input)[0, 1])
    prediction = int(model.predict(X_input)[0])

    return jsonify({
        "prediction": prediction,
        "failure_probability": round(probability, 4),
        "alert_level": generate_alert(probability),
        "message": "Failure predicted" if prediction == 1 else "Machine operating normally"
    })


if __name__ == "__main__":
    app.run(debug=True)