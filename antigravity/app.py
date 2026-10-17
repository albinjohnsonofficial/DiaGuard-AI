"""
DiaGuard: Flask Web Application Backend for Diabetes Prediction
Supports REST API endpoints for model inference, patient presets, model telemetry, and web dashboard.
Model: Support Vector Machine (SVM with RBF Kernel & Platt Probability Calibration)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'svm_diabetes_model.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.json')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Load trained model pipeline and metadata
model_pipeline = joblib.load(MODEL_PATH)

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

# Clinical feature schema with normal physiological ranges and units
FEATURE_INFO = {
    'Pregnancies': {
        'name': 'Pregnancies',
        'unit': 'count',
        'min': 0, 'max': 17, 'default': 2,
        'description': 'Number of times pregnant'
    },
    'Glucose': {
        'name': 'Plasma Glucose',
        'unit': 'mg/dL',
        'min': 40, 'max': 250, 'default': 115,
        'description': '2-hour oral glucose tolerance test plasma glucose concentration'
    },
    'BloodPressure': {
        'name': 'Diastolic Blood Pressure',
        'unit': 'mm Hg',
        'min': 40, 'max': 130, 'default': 72,
        'description': 'Diastolic blood pressure'
    },
    'SkinThickness': {
        'name': 'Triceps Skin Fold Thickness',
        'unit': 'mm',
        'min': 5, 'max': 99, 'default': 23,
        'description': 'Triceps skin fold thickness'
    },
    'Insulin': {
        'name': '2-Hour Serum Insulin',
        'unit': 'μU/mL',
        'min': 10, 'max': 850, 'default': 80,
        'description': '2-Hour serum insulin'
    },
    'BMI': {
        'name': 'Body Mass Index (BMI)',
        'unit': 'kg/m²',
        'min': 15.0, 'max': 65.0, 'default': 28.1,
        'description': 'Weight in kg/(height in m)²'
    },
    'DiabetesPedigreeFunction': {
        'name': 'Diabetes Pedigree Function',
        'unit': 'score',
        'min': 0.05, 'max': 2.5, 'default': 0.38,
        'description': 'Genetic family history diabetes risk score'
    },
    'Age': {
        'name': 'Patient Age',
        'unit': 'years',
        'min': 21, 'max': 85, 'default': 33,
        'description': 'Age in years'
    }
}

# Clinical Preset Profiles for easy demonstration
PRESET_PATIENTS = {
    'healthy': {
        'title': 'Low Risk (Healthy Adult)',
        'description': 'Normal glucose, optimal BMI, young age, low genetic risk',
        'data': {
            'Pregnancies': 1,
            'Glucose': 88,
            'BloodPressure': 68,
            'SkinThickness': 20,
            'Insulin': 65,
            'BMI': 22.4,
            'DiabetesPedigreeFunction': 0.21,
            'Age': 24
        }
    },
    'borderline': {
        'title': 'Moderate Risk (Pre-Diabetic Tendency)',
        'description': 'Impaired fasting glucose, slightly elevated BMI and BP',
        'data': {
            'Pregnancies': 3,
            'Glucose': 132,
            'BloodPressure': 78,
            'SkinThickness': 28,
            'Insulin': 140,
            'BMI': 29.8,
            'DiabetesPedigreeFunction': 0.45,
            'Age': 41
        }
    },
    'high_risk': {
        'title': 'High Risk (Diabetic Profile)',
        'description': 'Markedly elevated glucose, high BMI, maternal history',
        'data': {
            'Pregnancies': 6,
            'Glucose': 175,
            'BloodPressure': 88,
            'SkinThickness': 35,
            'Insulin': 210,
            'BMI': 36.5,
            'DiabetesPedigreeFunction': 0.85,
            'Age': 52
        }
    }
}

def analyze_risk_factors(data):
    """Generates clinical observations based on standard medical benchmarks."""
    factors = []
    
    glucose = data.get('Glucose', 0)
    if glucose >= 140:
        factors.append({'severity': 'high', 'text': f'Elevated Glucose ({glucose} mg/dL): Exceeds the normal postprandial threshold of 140 mg/dL.'})
    elif glucose >= 110:
        factors.append({'severity': 'moderate', 'text': f'Borderline Glucose ({glucose} mg/dL): Falls within the impaired glucose tolerance range.'})
    
    bmi = data.get('BMI', 0)
    if bmi >= 30.0:
        factors.append({'severity': 'high', 'text': f'High BMI ({bmi} kg/m²): Classified in the Obese range (BMI ≥ 30), a major risk factor for insulin resistance.'})
    elif bmi >= 25.0:
        factors.append({'severity': 'moderate', 'text': f'Overweight BMI ({bmi} kg/m²): Classified in the Overweight range (25.0 - 29.9).'})
        
    age = data.get('Age', 0)
    if age >= 45:
        factors.append({'severity': 'moderate', 'text': f'Age {age}: Individuals aged 45+ exhibit naturally increased metabolic vulnerability.'})
        
    dpf = data.get('DiabetesPedigreeFunction', 0)
    if dpf >= 0.6:
        factors.append({'severity': 'high', 'text': f'Elevated Genetic Risk ({dpf:.2f}): Indicates strong familial hereditary predisposition.'})
        
    bp = data.get('BloodPressure', 0)
    if bp >= 85:
        factors.append({'severity': 'moderate', 'text': f'Elevated Diastolic BP ({bp} mm Hg): Stage 1/2 hypertension tendency.'})

    if not factors:
        factors.append({'severity': 'low', 'text': 'All recorded clinical markers fall within standard normal ranges.'})
        
    return factors

# Routes
@app.route('/')
def index():
    return render_template('index.html', features=FEATURE_INFO, presets=PRESET_PATIENTS, metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'success': False, 'error': 'No input payload provided.'}), 400

        # Build feature DataFrame matching model columns exactly
        feature_order = metadata['feature_names']
        input_data = {}
        for feat in feature_order:
            if feat not in payload:
                return jsonify({'success': False, 'error': f'Missing feature: {feat}'}), 400
            try:
                val = float(payload[feat])
                # In Pima dataset, 0 in physiological metrics means missing
                if feat in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and val == 0:
                    val = np.nan
                input_data[feat] = [val]
            except ValueError:
                return jsonify({'success': False, 'error': f'Invalid numeric value for {feat}'}), 400

        input_df = pd.DataFrame(input_data)

        # Run inference through pipeline
        prediction = int(model_pipeline.predict(input_df)[0])
        probabilities = model_pipeline.predict_proba(input_df)[0]
        prob_non_diabetic = float(probabilities[0])
        prob_diabetic = float(probabilities[1])

        # Risk categorization
        if prob_diabetic < 0.30:
            risk_category = 'Low Risk'
            risk_color = '#28a745'
            risk_level = 'low'
        elif prob_diabetic < 0.60:
            risk_category = 'Moderate Risk'
            risk_color = '#ffc107'
            risk_level = 'moderate'
        else:
            risk_category = 'High Risk'
            risk_color = '#dc3545'
            risk_level = 'high'

        # Clinical recommendations
        if risk_level == 'high':
            recommendations = [
                'Schedule a formal Laboratory Fasting Plasma Glucose (FPG) and HbA1c test with a certified physician.',
                'Adopt a strict low-glycemic Mediterranean or DASH dietary pattern.',
                'Engage in 150+ minutes of moderate-intensity aerobic exercise per week.',
                'Monitor blood pressure and lipid profile to minimize cardiovascular co-morbidity risks.'
            ]
        elif risk_level == 'moderate':
            recommendations = [
                'Perform an annual diagnostic diabetes screening (HbA1c test).',
                'Reduce refined carbohydrates and sugar-sweetened beverages.',
                'Implement lifestyle modifications targeting 5-7% weight reduction if BMI > 25.',
                'Increase daily physical activity (e.g. 30-minute brisk walk daily).'
            ]
        else:
            recommendations = [
                'Maintain current healthy diet rich in dietary fiber, whole grains, and lean proteins.',
                'Continue regular physical activity and wellness routines.',
                'Undergo routine preventive health checkups every 1-2 years.'
            ]

        risk_factors = analyze_risk_factors(payload)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_diabetic': round(prob_diabetic * 100, 2),
            'probability_non_diabetic': round(prob_non_diabetic * 100, 2),
            'risk_category': risk_category,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'model_info': {
                'algorithm': 'Support Vector Machine (RBF Kernel)',
                'cv_accuracy': f"{metadata['cv_metrics']['Support Vector Machine (SVM)']['CV Accuracy']['mean']*100:.2f}%",
                'test_accuracy': f"{metadata['test_metrics']['Support Vector Machine (SVM)']['Accuracy']*100:.2f}%"
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify(PRESET_PATIENTS)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    return jsonify(metadata)

@app.route('/figures/<path:filename>')
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)

if __name__ == '__main__':
    print("DiaGuard Diabetes Prediction Server starting on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
