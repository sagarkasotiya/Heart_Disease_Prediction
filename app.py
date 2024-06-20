from flask import Flask, request, render_template, jsonify
import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(message)s')

sys.path.insert(0, 'src')  # Correcting the path

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__, template_folder='templates')

app = application

# Home Page Route
@app.route('/')
def home_page():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Extracting form data and converting to appropriate data types
            form_data = {
                'Age': float(request.form.get('Age')),
                'Gender': request.form.get('Gender'),
                'Cholesterol': float(request.form.get('Cholesterol')),
                'BloodPressure': float(request.form.get('BloodPressure')),
                'HeartRate': float(request.form.get('HeartRate')),
                'Smoking': request.form.get('Smoking'),
                'AlcoholIntake': request.form.get('AlcoholIntake'),
                'ExerciseHours': float(request.form.get('ExerciseHours')),
                'FamilyHistory': request.form.get('FamilyHistory'),
                'Diabetes': request.form.get('Diabetes'),
                'Obesity': request.form.get('Obesity'),
                'StressLevel': float(request.form.get('StressLevel')),
                'BloodSugar': float(request.form.get('BloodSugar')),
                'ExerciseInducedAngina': request.form.get('ExerciseInducedAngina'),
                'ChestPainType': request.form.get('ChestPainType')
            }

            # Validate form data
            missing_fields = [key for key, value in form_data.items() if value in (None, '')]
            if missing_fields:
                raise ValueError(f"Missing fields: {', '.join(missing_fields)}")

            # Create CustomData instance
            data = CustomData(**form_data)
            final_new_data = data.get_data_as_dataframe()

            # Predict using the prediction pipeline
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            # Generate prediction result
            result = pred[0]
            if result == 0:
                final_result = "Prediction: You have No Heart Disease"
            else:
                final_result = "Prediction: You have Heart Disease"
            
            return render_template('form.html', final_result=final_result)

        except Exception as e:
            # Log the error
            logging.error(f"Error in prediction: {e}")
            return "Error in processing your request. Please try again."

# Run the application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
