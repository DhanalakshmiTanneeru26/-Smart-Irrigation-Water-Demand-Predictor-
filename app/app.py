from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'supersecretkey'

@app.context_processor
def inject_year():
    return {'year': datetime.now().year}

# Load the trained model - use path relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'irrigation_model.pkl')
model = joblib.load(MODEL_PATH)

# Templates are stored in app/templates/ (home.html, results.html)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        features = {
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'rainfall': float(request.form['rainfall']),
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K'])
        }

        # Make prediction
        input_array = np.array([[
            features['temperature'],
            features['humidity'],
            features['rainfall'],
            features['N'],
            features['P'],
            features['K']
        ]])

        prediction = model.predict(input_array)[0]
        prediction_formatted = f"{prediction:.2f}"

        # Find recommended crop
        import pandas as pd
        try:
            crop_data = pd.read_csv(os.path.join(SCRIPT_DIR, '../../Crop_recommendation.csv'))
            similarities = []
            for _, row in crop_data.iterrows():
                distance = sum((features[col] - row[col])**2 for col in ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K'])
                similarity = 1 / (1 + distance)
                similarities.append(similarity)
            best_idx = similarities.index(max(similarities))
            recommended_crop = crop_data.iloc[best_idx]['label']
        except:
            recommended_crop = "Unable to determine"

    except Exception as e:
        prediction_formatted = f"Error: {str(e)}"
        recommended_crop = "N/A"

    return render_template('results.html', prediction=prediction_formatted, recommended_crop=recommended_crop)

if __name__ == '__main__':
    app.run(debug=True)