from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

pipeline = joblib.load('customer prediction/Project_Customer_Category_Classification.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame({
        'Fresh': [float(data['Fresh'])],
        'Milk': [float(data['milk'])],
        'Grocery': [float(data['grocery'])],
        'Frozen': [float(data['frozen'])],
        'Detergents_Paper': [float(data['Detergents_Paper'])]
    })

    preds = pipeline.predict(input_data)
    pred = int(preds[0])

    if pred == 2:
        customer = 'Customer is from Retail Store'
    elif pred == 1:
        customer = 'Customer is from Hotel/Restaurant/Cafe'
    else:
        customer = 'error'

    return jsonify({'prediction': customer})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
