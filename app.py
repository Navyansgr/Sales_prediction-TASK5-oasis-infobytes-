from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load model and scaler
model = joblib.load('app/model.pkl')
scaler = joblib.load('app/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    
    # Prediction
    prediction = model.predict(features_scaled)[0]
    
    return render_template('index.html', prediction_text=f'💰 Predicted Sales: ${prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
