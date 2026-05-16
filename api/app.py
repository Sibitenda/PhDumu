from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained models
regression_model = joblib.load(
    "models/covid_regression_model.pkl"
)

classification_model = joblib.load(
    "models/covid_classification_model.pkl"
)

# Create Flask app
app = Flask(__name__)

# Risk labels
risk_labels = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

# Home route
@app.route("/")
def home():

    return {
        "message": "COVID Prediction API Running"
    }

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON data
        data = request.get_json()

        # Extract features
        previous_day_cases = data[
            "previous_day_cases"
        ]

        rolling_avg_cases = data[
            "rolling_avg_cases"
        ]

        growth_rate = data[
            "growth_rate"
        ]

        # Prepare model input
        features = np.array([[
            previous_day_cases,
            rolling_avg_cases,
            growth_rate
        ]])

        # Predict future cases
        predicted_cases = (
            regression_model.predict(features)[0]
        )

        # Predict risk category
        risk_prediction = (
            classification_model.predict(features)[0]
        )

        # Convert class to label
        risk_name = risk_labels[
            risk_prediction
        ]

        # API response
        response = {

            "predicted_cases": round(
                float(predicted_cases),
                2
            ),

            "risk_level": risk_name
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# Run Flask application
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )