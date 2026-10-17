# DiaGuard: Diabetes Prediction Using Machine Learning (Support Vector Machine)

An end-to-end, clinical-grade Machine Learning system designed for early Type-2 Diabetes detection using **Support Vector Machine (SVM)**, complete with leak-free preprocessing pipelines, 5-fold cross-validation, an interactive Full-Stack Web Application (Flask REST API + Modern Responsive Frontend), an interactive Streamlit application, and a comprehensive academic report.

> **Course Outcome Alignment:**  
> **CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures.**

---

## 🌟 Key Project Highlights

- **Machine Learning Core:**
  - Evaluated **Support Vector Machine (SVM with RBF Kernel & Platt Calibration)** against **Logistic Regression** (linear baseline) and **Random Forest** (ensemble comparator).
  - Addressed physiological missing data (0 values in Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) via **Median Imputation** inside a `Pipeline` combined with `StandardScaler` to strictly prevent data leakage.
  - Validated using **5-Fold Stratified Cross-Validation** and evaluated on an 80/20 holdout test set across **Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC**.
  - Generated and saved **5 high-resolution diagnostic visualizations** with detailed clinical captions.

- **Dual Application Deployments:**
  1. **Full-Stack Web App (`app.py`):**
     - **Backend:** Flask REST API (`/api/predict`, `/api/model-info`, `/api/presets`, `/figures/<filename>`).
     - **Frontend:** Modern healthcare UI with synchronized dual slider/number controls, 1-click clinical patient presets, live probability spectrum bar, risk severity badges, and actionable clinical recommendations.
  2. **Streamlit App (`streamlit_app.py`):**
     - Step 5 implementation with intuitive sliders, "Predict Diabetes" button, real-time risk gauges, model comparison table, and embedded figures.

---

## 📁 Repository Structure

```text
├── diabetes.csv                # Pima Indians Diabetes Dataset (768 patients, 9 attributes)
├── train_model.py             # ML pipeline: EDA, preprocessing, 5-fold CV, model training, evaluation & plotting
├── app.py                     # Flask REST API backend server
├── streamlit_app.py           # Streamlit interactive application
├── test_app.py                # Automated unit & integration tests
├── requirements.txt           # Project dependencies
├── REPORT.md                  # Comprehensive 4–6 page academic project report
├── templates/
│   └── index.html             # Full-Stack frontend dashboard template
├── static/
│   ├── style.css              # Custom responsive stylesheet
│   └── app.js                 # Interactive client controller and API connector
├── model/
│   ├── svm_diabetes_model.joblib # Serialized best-performing SVM Pipeline
│   └── model_metadata.json    # Model evaluation metrics and feature schema
└── figures/
    ├── 01_class_distribution.png    # Class balance (Outcome 0 vs 1)
    ├── 02_correlation_heatmap.png   # Clinical feature correlation matrix
    ├── 03_confusion_matrix.png      # Multi-model confusion matrices
    ├── 04_roc_curve.png             # Receiver Operating Characteristic curves
    └── 05_model_comparison.png      # Multi-metric comparative bar chart
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Ensure Python 3.9+ is installed:

```bash
# Clone or navigate to the project directory
cd "c:\Users\TECHNO\Documents\BTECH\machine learning\antigravity"

# Install required packages
pip install -r requirements.txt
```

---

### 2. Train and Evaluate the Models

To execute the data pipeline, perform 5-fold cross-validation, save the trained SVM model, and generate all evaluation figures:

```bash
python train_model.py
```

**Output Summary:**
- **5-Fold CV Accuracy (SVM):** `77.36% (±1.58%)` | **CV ROC-AUC:** `0.8311`
- **Holdout Test Accuracy (SVM):** `74.68%` | **F1-Score:** `0.5979` | **Recall:** `0.5370`
- Minimizes False Negatives (FN = 25) compared to Random Forest (26) and Logistic Regression (27).

---

### 3. Option A: Run the Full-Stack Web Application (Flask Backend + Modern Frontend)

Launch the Flask server:

```bash
python app.py
```

Then open your browser at:
👉 **`http://127.0.0.1:5000`**

**Features:**
- Real-time prediction with dynamic sliders and synchronized numeric inputs.
- 1-click clinical presets (**Low Risk / Healthy**, **Moderate Risk / Pre-diabetic**, **High Risk / Diabetic**).
- Probability progress gauge, contributing risk flags, and clinical recommendations.
- Interactive model performance benchmarks and embedded figure lightbox viewer.
- Full Course Outcome (CO5) alignment summary.

---

### 4. Option B: Run the Streamlit Application

As specified in **Step 5**, you can also launch the dedicated Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then open your browser at:
👉 **`http://localhost:8501`**

---

### 5. Run Automated Tests

To verify that the Flask backend, prediction endpoints, input validation, and asset delivery function properly:

```bash
python test_app.py
```

---

## 📊 Summary of Model Evaluation (Holdout Test Set)

| Algorithm | 5-Fold CV Accuracy | 5-Fold CV ROC-AUC | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM - RBF)** 🏆 | **77.36%** | **0.8311** | **74.68%** | **0.6744** | **0.5370** | **0.5979** | **0.8019** |
| Random Forest (Ensemble) | 77.03% | 0.8344 | 74.03% | 0.6667 | 0.5185 | 0.5833 | 0.8117 |
| Logistic Regression (Linear) | 78.82% | 0.8434 | 70.78% | 0.6000 | 0.5000 | 0.5455 | 0.8130 |

### 🩺 Why Recall Matters Most in Clinical Diagnosis:
In diabetes screening, **False Negatives (Type II Error)** carry catastrophic health risks. If an undiagnosed diabetic patient is falsely classified as non-diabetic, they do not receive timely interventions, accelerating irreversible microvascular deterioration (retinopathy, renal failure). In contrast, a **False Positive (Type I Error)** merely leads to confirmatory laboratory testing (HbA1c/FPG). **Support Vector Machine achieved the lowest False Negatives (25) and highest Recall (0.5370), making it the optimal clinical screening choice.**

---

## 📄 Academic Project Report

The complete 4–6 page academic project report is available in **[`REPORT.md`](REPORT.md)**.
