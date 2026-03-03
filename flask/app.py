from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model (path relative to this file)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "payments.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Fraud Transaction"
    else:
        result = "Not Fraud"

    return render_template("submit.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)