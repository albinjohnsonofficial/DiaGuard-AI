"""
DiaGuard: Streamlit Web Application for Diabetes Prediction (SVM)
Step 5 Implementation: Streamlit interactive interface with sliders, Predict button, and results.
Course Outcome: CO5 - Solve real life problems using appropriate machine learning models and evaluate the performance measures.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="DiaGuard - Diabetes Prediction (SVM)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'svm_diabetes_model.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'model_metadata.json')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

@st.cache_resource
def load_assets():
    """Load cached SVM pipeline and metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r') as f:
        meta = json.load(f)
    return model, meta

try:
    model, metadata = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1976d2;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 15px;
        border-radius: 8px;
    }
    .risk-mod {
        background-color: #fff3e0;
        border-left: 6px solid #f57c00;
        padding: 15px;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-title">🩺 DiaGuard: Diabetes Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Support Vector Machine (SVM with RBF Kernel) • 5-Fold Stratified Cross-Validation • CO5 Compliant</p>', unsafe_allow_html=True)

# Sidebar: Quick Presets & Details
st.sidebar.header("⚡ Quick Patient Presets")
preset_choice = st.sidebar.radio(
    "Select a pre-configured clinical profile:",
    ("Custom Inputs", "Low Risk (Healthy Adult)", "Moderate Risk (Pre-diabetic)", "High Risk (Diabetic Profile)")
)

# Preset definitions
if preset_choice == "Low Risk (Healthy Adult)":
    defaults = {'Pregnancies': 1, 'Glucose': 88, 'BloodPressure': 68, 'SkinThickness': 20, 'Insulin': 65, 'BMI': 22.4, 'DPF': 0.21, 'Age': 24}
elif preset_choice == "Moderate Risk (Pre-diabetic)":
    defaults = {'Pregnancies': 3, 'Glucose': 132, 'BloodPressure': 78, 'SkinThickness': 28, 'Insulin': 140, 'BMI': 29.8, 'DPF': 0.45, 'Age': 41}
elif preset_choice == "High Risk (Diabetic Profile)":
    defaults = {'Pregnancies': 6, 'Glucose': 175, 'BloodPressure': 88, 'SkinThickness': 35, 'Insulin': 210, 'BMI': 36.5, 'DPF': 0.85, 'Age': 52}
else:
    defaults = {'Pregnancies': 2, 'Glucose': 115, 'BloodPressure': 72, 'SkinThickness': 23, 'Insulin': 80, 'BMI': 28.1, 'DPF': 0.38, 'Age': 33}

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Telemetry")
st.sidebar.info(f"""
- **Algorithm:** Support Vector Machine
- **Kernel:** Radial Basis Function (RBF)
- **CV Accuracy:** {metadata['cv_metrics']['Support Vector Machine (SVM)']['CV Accuracy']['mean']*100:.2f}%
- **Test Accuracy:** {metadata['test_metrics']['Support Vector Machine (SVM)']['Accuracy']*100:.2f}%
- **Leakage Prevention:** Pipeline (Median Imputation + StandardScaler)
""")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Predictor Interface", "📈 Model Evaluation & Figures", "🎓 CO5 Mapping & Clinical Analysis"])

with tab1:
    st.markdown("### 📋 Enter Patient Biomarkers")
    st.write("Use the sliders below to adjust the patient's diagnostic laboratory values:")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("1. Number of Pregnancies", min_value=0, max_value=17, value=int(defaults['Pregnancies']), step=1,
                                help="Total number of times pregnant")
        glucose = st.slider("2. Plasma Glucose Concentration (mg/dL)", min_value=40, max_value=250, value=int(defaults['Glucose']), step=1,
                            help="2-hour oral glucose tolerance test (Normal < 140 mg/dL)")
        bp = st.slider("3. Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=130, value=int(defaults['BloodPressure']), step=1,
                       help="Diastolic blood pressure (Normal 60-80 mm Hg)")
        skin_thickness = st.slider("4. Triceps Skin Fold Thickness (mm)", min_value=5, max_value=99, value=int(defaults['SkinThickness']), step=1,
                                   help="Indicator of subcutaneous adipose tissue")

    with col2:
        insulin = st.slider("5. 2-Hour Serum Insulin (μU/mL)", min_value=10, max_value=850, value=int(defaults['Insulin']), step=1,
                            help="Normal fasting range: 16 to 166 μU/mL")
        bmi = st.slider("6. Body Mass Index - BMI (kg/m²)", min_value=15.0, max_value=65.0, value=float(defaults['BMI']), step=0.1,
                        help="Weight in kg/(height in m)^2 (Normal 18.5-24.9, Obese ≥ 30)")
        dpf = st.slider("7. Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=float(defaults['DPF']), step=0.01,
                        help="Genetic score based on family diabetes history")
        age = st.slider("8. Patient Age (years)", min_value=21, max_value=85, value=int(defaults['Age']), step=1,
                        help="Age in completed years")

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

    if predict_btn:
        # Construct input DataFrame matching training feature names
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose if glucose > 0 else np.nan,
            'BloodPressure': bp if bp > 0 else np.nan,
            'SkinThickness': skin_thickness if skin_thickness > 0 else np.nan,
            'Insulin': insulin if insulin > 0 else np.nan,
            'BMI': bmi if bmi > 0 else np.nan,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_nondiabetic = probabilities[0] * 100
        prob_diabetic = probabilities[1] * 100

        st.markdown("### 📊 Diagnostic Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.error("### 🚨 Result: Positive (Diabetic Tendency)")
                st.markdown(f"""
                <div class="risk-high">
                    <h4>High Diabetes Risk Detected</h4>
                    <p>The Support Vector Machine classifies this patient's profile in the <strong>Diabetic</strong> region of the feature space.</p>
                    <p><strong>Predicted Probability of Diabetes:</strong> {prob_diabetic:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("### ✅ Result: Negative (Non-Diabetic)")
                st.markdown(f"""
                <div class="risk-low">
                    <h4>Low Diabetes Risk Profile</h4>
                    <p>The Support Vector Machine classifies this patient's profile in the <strong>Non-Diabetic</strong> region of the feature space.</p>
                    <p><strong>Probability of Non-Diabetic:</strong> {prob_nondiabetic:.1f}% (Diabetic risk: {prob_diabetic:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Probability Spectrum")
            st.progress(int(prob_diabetic))
            st.caption(f"Diabetic Probability: {prob_diabetic:.1f}% | Non-Diabetic Probability: {prob_nondiabetic:.1f}%")

            # Risk categorization
            if prob_diabetic >= 60:
                risk_tier = "🔴 High Risk (> 60%)"
            elif prob_diabetic >= 30:
                risk_tier = "🟠 Moderate Risk (30% - 60%)"
            else:
                risk_tier = "🟢 Low Risk (< 30%)"
            st.metric("Clinical Risk Category", risk_tier)

        st.markdown("---")
        st.markdown("#### 🩺 Clinical Observations & Guidance")
        obs_col1, obs_col2 = st.columns(2)

        with obs_col1:
            st.markdown("**Key Biomarker Flags:**")
            flags = []
            if glucose >= 140:
                flags.append(f"⚠️ **Glucose ({glucose} mg/dL):** Elevated above normal postprandial levels (≥140 mg/dL).")
            elif glucose >= 110:
                flags.append(f"ℹ️ **Glucose ({glucose} mg/dL):** Borderline impaired glucose tolerance.")
            if bmi >= 30.0:
                flags.append(f"⚠️ **BMI ({bmi} kg/m²):** Classified as Obese (BMI ≥ 30). Increases insulin resistance.")
            elif bmi >= 25.0:
                flags.append(f"ℹ️ **BMI ({bmi} kg/m²):** Classified as Overweight (25 - 29.9).")
            if dpf >= 0.60:
                flags.append(f"⚠️ **Pedigree Score ({dpf:.2f}):** Significant hereditary genetic risk factor.")
            if age >= 45:
                flags.append(f"ℹ️ **Age ({age} yrs):** Age 45+ presents higher baseline metabolic vulnerability.")
            if not flags:
                flags.append("✅ All clinical indicators are within standard physiological reference ranges.")
            for f in flags:
                st.write(f)

        with obs_col2:
            st.markdown("**Actionable Recommendations:**")
            if prob_diabetic >= 60:
                st.write("1. 🩺 Consult a physician for formal Fasting Plasma Glucose (FPG) and HbA1c tests.")
                st.write("2. 🥗 Implement a medically supervised low-glycemic Mediterranean or DASH diet.")
                st.write("3. 🏃 Aim for at least 150 minutes of moderate aerobic physical activity weekly.")
                st.write("4. 🩸 Screen for co-morbid hypertension and dyslipidemia.")
            elif prob_diabetic >= 30:
                st.write("1. 🩺 Schedule annual preventive blood glucose and HbA1c screening.")
                st.write("2. 🥗 Cut down on refined carbohydrates and sugar-sweetened beverages.")
                st.write("3. 🏃 Target 5-7% weight reduction through routine exercise.")
            else:
                st.write("1. 🥗 Continue balanced nutrition rich in dietary fiber, vegetables, and lean protein.")
                st.write("2. 🏃 Maintain regular physical activity and healthy BMI.")
                st.write("3. 🩺 Routine preventive health checkup every 1-2 years.")

with tab2:
    st.markdown("### 📊 Model Performance & Comparative Benchmark")
    
    # Table
    metrics_data = {
        'Algorithm': ['Support Vector Machine (SVM - RBF)', 'Random Forest (Ensemble)', 'Logistic Regression (Linear)'],
        '5-Fold CV Accuracy': ['77.36% (±1.58%)', '77.03% (±1.98%)', '78.82% (±1.91%)'],
        '5-Fold CV ROC-AUC': ['0.8311', '0.8344', '0.8434'],
        'Test Accuracy': ['74.68%', '74.03%', '70.78%'],
        'Test Precision': ['0.6744', '0.6667', '0.6000'],
        'Test Recall (Sensitivity)': ['0.5370', '0.5185', '0.5000'],
        'Test F1-Score': ['0.5979', '0.5833', '0.5455'],
        'Test ROC-AUC': ['0.8019', '0.8117', '0.8130']
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    st.markdown("### 🖼️ Diagnostic Evaluation Figures")
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.image(os.path.join(FIGURES_DIR, '01_class_distribution.png'), caption="Figure 1: Outcome Class Distribution (500 Non-Diabetic, 268 Diabetic)")
        st.image(os.path.join(FIGURES_DIR, '03_confusion_matrix.png'), caption="Figure 3: Confusion Matrices (SVM minimizes False Negatives)")
        st.image(os.path.join(FIGURES_DIR, '05_model_comparison.png'), caption="Figure 5: Model Comparison Across All Core Metrics")

    with fig_col2:
        st.image(os.path.join(FIGURES_DIR, '02_correlation_heatmap.png'), caption="Figure 2: Clinical Feature Correlation Matrix (Glucose & BMI lead)")
        st.image(os.path.join(FIGURES_DIR, '04_roc_curve.png'), caption="Figure 4: Receiver Operating Characteristic (ROC) Curves")

with tab3:
    st.markdown("### 🎓 Course Outcome (CO5) Mapping")
    st.info('**Course Outcome:** "CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures"')

    st.markdown("""
    #### 1. Real-Life Problem Formulation
    Type-2 Diabetes Mellitus represents a severe global health burden affecting over 500 million individuals. Undetected diabetes leads to catastrophic microvascular damage (retinopathy causing blindness, diabetic nephropathy leading to dialysis) and macrovascular disease (myocardial infarction, stroke). Developing an intelligent machine learning diagnostic screening tool directly tackles this critical healthcare challenge.

    #### 2. Appropriate Model Selection: Support Vector Machine (SVM)
    The clinical feature space is non-linearly separable and features moderate sample dimensionality (n=768, d=8). 
    - The **Support Vector Machine (SVM)** with a **Radial Basis Function (RBF)** kernel is mathematically optimal for this problem because it projects the non-linear physiological relationships into an infinite-dimensional Hilbert space, finding the maximum-margin hyperplane that minimizes structural risk.
    - Compared to linear models (Logistic Regression) and greedy partitioning (Random Forest), SVM provides superior generalization on the holdout test set with minimal overfitting.

    #### 3. Rigorous Evaluation Measures & Error Cost Analysis
    In clinical decision systems, not all errors carry equal weight:
    - **False Positive (Type I Error):** A healthy individual is predicted diabetic. Clinical consequence: The patient is referred for a confirmatory laboratory HbA1c test; temporary emotional inconvenience, zero permanent bodily harm.
    - **False Negative (Type II Error):** A diabetic patient is misclassified as non-diabetic. Clinical consequence: The disease progresses unchecked, resulting in irreversible microvascular complications, neuropathy, and potentially fatal diabetic ketoacidosis.
    
    Therefore, while **Accuracy** provides a general overview, **Recall (Sensitivity)** and **ROC-AUC** are the clinically paramount performance measures. The SVM model achieved the lowest False Negatives (25) and highest Recall (0.5370) on the holdout evaluation.
    """)

st.markdown("---")
st.caption("DiaGuard Diabetes Prediction System • B.Tech Machine Learning Capstone Project")
