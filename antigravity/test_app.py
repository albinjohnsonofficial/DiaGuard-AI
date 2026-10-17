"""
Unit and Integration Tests for DiaGuard Diabetes Prediction Application
"""

import unittest
import json
from app import app

class DiaGuardAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        """Test GET / renders dashboard"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'DiaGuard AI', response.data)
        self.assertIn(b'Support Vector Machine', response.data)

    def test_prediction_healthy_patient(self):
        """Test POST /api/predict with low-risk profile"""
        payload = {
            'Pregnancies': 1,
            'Glucose': 88,
            'BloodPressure': 68,
            'SkinThickness': 20,
            'Insulin': 65,
            'BMI': 22.4,
            'DiabetesPedigreeFunction': 0.21,
            'Age': 24
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 0)
        self.assertEqual(data['prediction_label'], 'Non-Diabetic')
        self.assertLess(data['probability_diabetic'], 40.0)

    def test_prediction_diabetic_patient(self):
        """Test POST /api/predict with high-risk profile"""
        payload = {
            'Pregnancies': 6,
            'Glucose': 180,
            'BloodPressure': 90,
            'SkinThickness': 35,
            'Insulin': 210,
            'BMI': 38.0,
            'DiabetesPedigreeFunction': 0.95,
            'Age': 54
        }
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction'], 1)
        self.assertEqual(data['prediction_label'], 'Diabetic')
        self.assertGreater(data['probability_diabetic'], 50.0)

    def test_missing_feature_validation(self):
        """Test POST /api/predict returns error on missing parameter"""
        payload = {'Pregnancies': 2, 'Glucose': 120}  # Missing other 6 features
        response = self.app.post('/api/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])

    def test_figures_endpoint(self):
        """Test GET /figures/... serves generated PNGs"""
        for fig in ['01_class_distribution.png', '02_correlation_heatmap.png',
                    '03_confusion_matrix.png', '04_roc_curve.png', '05_model_comparison.png']:
            response = self.app.get(f'/figures/{fig}')
            self.assertEqual(response.status_code, 200, f'Figure {fig} failed to load')

if __name__ == '__main__':
    unittest.main()
