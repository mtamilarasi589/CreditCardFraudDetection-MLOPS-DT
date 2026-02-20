from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Fraud Detection Decision Tree API 🌳"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    features = np.array([list(data.values())])
    prediction = model.predict(features)

    result = "Fraud" if prediction[0] == 1 else "Normal"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
