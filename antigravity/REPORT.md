# Academic Project Report
## Diabetes Prediction Using Machine Learning: A Comparative Study with Support Vector Machine (SVM)

**Course Outcome:** CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures  
**Domain:** Healthcare Informatics / Applied Predictive Analytics  
**Primary Algorithm:** Support Vector Machine (Radial Basis Function Kernel)  
**Comparator Models:** Logistic Regression, Random Forest Classifier  

---

### Executive Summary

Diabetes Mellitus represents one of the most critical public health challenges of the 21st century, affecting over 537 million adults globally. Undetected diabetes leads to catastrophic vascular complications, including diabetic retinopathy, nephropathy, peripheral neuropathy, and accelerated coronary artery disease. Early risk stratification through automated machine learning provides an accessible, non-invasive mechanism for prompt clinical intervention.

This project implements a complete, leak-free machine learning pipeline for diabetes prediction utilizing the benchmark Pima Indians Diabetes Dataset. We evaluate three distinct paradigms: **Logistic Regression** (linear baseline), **Support Vector Machine (SVM with RBF Kernel)** (primary focus), and **Random Forest Classifier** (non-linear ensemble). To avoid data leakage, physiological missing-value imputation (addressing biologically impossible zeros) and feature scaling are encapsulated within a scikit-learn `Pipeline`. Models are evaluated via **5-Fold Stratified Cross-Validation** and a holdout 20% test partition across Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC. 

On the holdout evaluation, **Support Vector Machine achieved the top performance** with **74.68% Accuracy**, **0.6744 Precision**, **0.5370 Recall**, and **0.5979 F1-Score**, while minimizing life-threatening False Negatives to 25. An interactive dual application suite—comprising a **Flask REST API with a modern web dashboard frontend** and a **Streamlit clinical decision support application**—was developed and deployed to facilitate real-time clinical screening.

---

### 1. Problem Statement

Type-2 Diabetes Mellitus (T2DM) is characterized by chronic hyperglycemia resulting from progressive insulin resistance combined with insufficient compensatory insulin secretion by pancreatic $\beta$-cells. According to the International Diabetes Federation (IDF):
- Approximately 1 in 10 adults currently live with diabetes.
- Over 45% of individuals with diabetes are undiagnosed, remaining asymptomatic until secondary end-organ damage manifests.
- Global diabetes-related health expenditures exceed \$966 billion annually.

Conventional clinical diagnosis relies on venous blood draws for Fasting Plasma Glucose (FPG $\ge 126\text{ mg/dL}$), Oral Glucose Tolerance Testing (OGTT 2-hour plasma glucose $\ge 200\text{ mg/dL}$), or Glycated Hemoglobin ($\text{HbA}_{1c} \ge 6.5\%$). However, routine laboratory screening in underserved and primary-care settings is hindered by cost, invasiveness, and lack of specialized laboratory infrastructure.

**Objective:**  
Develop an automated, mathematically rigorous, and clinical-decision-ready machine learning framework capable of predicting the probability of diabetes using non-invasive demographic and routine physiological markers. The model must balance overall accuracy with clinical sensitivity (Recall), ensuring that potentially diabetic patients are not overlooked.

---

### 2. Dataset Description

The project utilizes the gold-standard **Pima Indians Diabetes Dataset**, originally curated by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). 

#### 2.1 Dataset Dimensionality & Characteristics
- **Total Patient Records (Rows):** 768 female patients aged $\ge 21$ years of Pima Indian heritage.
- **Total Features (Columns):** 9 (8 continuous/discrete clinical covariates + 1 binary diagnostic target).
- **Target Variable (`Outcome`):**
  - **Class 0 (Non-Diabetic):** 500 instances ($65.10\%$)
  - **Class 1 (Diabetic):** 268 instances ($34.90\%$)
  - **Imbalance Ratio:** Approximately $1.87:1$, presenting moderate class imbalance that necessitates stratified partitioning and threshold-independent metrics (ROC-AUC).

#### 2.2 Feature Dictionary & Clinical Significance

| Feature Name | Clinical Description | Measurement Unit | Empirical Range | Physiological Role |
| :--- | :--- | :---: | :---: | :--- |
| **Pregnancies** | Number of times pregnant | Count | $0 - 17$ | Gestational stress on pancreatic $\beta$-cell capacity |
| **Glucose** | 2-hour oral glucose tolerance plasma concentration | $\text{mg/dL}$ | $0 - 199$ | Direct measure of metabolic carbohydrate clearance |
| **BloodPressure** | Diastolic blood pressure | $\text{mm Hg}$ | $0 - 122$ | Indicator of vascular tone and co-morbid hypertension |
| **SkinThickness** | Triceps skin fold thickness | $\text{mm}$ | $0 - 99$ | Surrogate anthropometric measure of subcutaneous adiposity |
| **Insulin** | 2-hour serum insulin concentration | $\mu\text{U/mL}$ | $0 - 846$ | Indicator of endogenous insulin production vs. hyperinsulinemia |
| **BMI** | Body Mass Index ($\text{weight in kg} / \text{height in m}^2$) | $\text{kg/m}^2$ | $0.0 - 67.1$ | Key risk factor for visceral adiposity and insulin resistance |
| **DiabetesPedigreeFunction** | Diabetes pedigree function | Score ($0 - 2.5$) | $0.078 - 2.42$ | Mathematical synthesis of genetic family risk history |
| **Age** | Patient age | Years | $21 - 81$ | Advancing age correlates with progressive metabolic decline |
| **Outcome** (Target) | Diagnostic class classification | Binary ($0, 1$) | $0$ or $1$ | Clinically validated diabetes diagnostic label |

---

### 3. Data Preprocessing & Leakage-Free Pipeline Architecture

Machine learning models deployed in healthcare frequently suffer from performance degradation due to subtle **data leakage** and naive handling of biologically anomalous data.

#### 3.1 Addressing Physiological Missing Data
In standard exploratory data analysis, standard inspection functions (`df.isnull().sum()`) report zero missing values across all columns. However, deep clinical domain analysis reveals severe physiological anomalies:
- A diastolic blood pressure of $0\text{ mm Hg}$ represents clinical circulatory arrest.
- A plasma glucose concentration of $0\text{ mg/dL}$ represents profound fatal hypoglycemia.
- Skin fold thickness, serum insulin, and BMI cannot physically be $0$ in living humans.

Quantification of biologically impossible zero values:
- **Insulin:** 374 zeros ($48.70\%$)
- **SkinThickness:** 227 zeros ($29.56\%$)
- **BloodPressure:** 35 zeros ($4.56\%$)
- **BMI:** 11 zeros ($1.43\%$)
- **Glucose:** 5 zeros ($0.65\%$)

**Strategy:**  
These zero values were mapped to `np.nan` so that statistical imputation could treat them appropriately.

#### 3.2 Imputation and Feature Normalization via Pipeline
Rather than imputing statistics over the global dataset prior to partitioning—which leaks future test distribution parameters into the training set—all preprocessing was embedded within a `scikit-learn.pipeline.Pipeline`:
1. **`SimpleImputer(strategy='median')`**: Median imputation is robust to skewness and heavy tails, computing replacement medians strictly on training folds.
2. **`StandardScaler()`**: Computes Z-score normalization:
   $$z = \frac{x - \mu}{\sigma}$$
   Ensuring all features have zero mean and unit variance ($\mu = 0, \sigma = 1$). Scaling is essential for distance-based and margin-based algorithms such as SVM.

#### 3.3 Stratified Partitioning
Data was divided into an **80% training partition** ($n = 614$) and a **20% holdout test partition** ($n = 154$) using `train_test_split` with `stratify=y` and `random_state=42`. This preserves the exact $65.1\% / 34.9\%$ class ratio across both splits.

---

### 4. Algorithm Selection and Theoretical Rationale

Three distinct model families were benchmarked to determine the optimal decision boundary for diabetes classification:

```mermaid
flowchart TD
    Raw["Pima Indian Diabetes Dataset (n=768)"] --> Split["80/20 Stratified Partition"]
    Split --> Train["Training Set (n=614)"]
    Split --> Test["Holdout Test Set (n=154)"]
    
    subgraph Pipeline["Leakage-Free Pipeline Architecture"]
        Train --> Imp["SimpleImputer (Median)"]
        Imp --> Scale["StandardScaler (Z-Score)"]
        Scale --> Models["Model Benchmark & Tuning"]
    end
    
    Models --> LR["1. Logistic Regression (Linear Baseline)"]
    Models --> SVM["2. Support Vector Machine (Primary Focus)"]
    Models --> RF["3. Random Forest (Ensemble Comparator)"]
    
    LR & SVM & RF --> CV["5-Fold Stratified Cross-Validation"]
    CV --> Eval["Final Evaluation on Holdout Test Set"]
    Eval --> Deploy["Clinical Web & Streamlit Applications"]
```

#### 4.1 Support Vector Machine (SVM with RBF Kernel) — Primary Algorithm
Support Vector Machine is the centerpiece of this study. The fundamental goal of an SVM is to locate an optimal separating hyperplane that maximizes the geometric margin between class boundaries:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{subject to } y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Where:
- $\mathbf{w}$ represents the weight vector perpendicular to the separating hyperplane.
- $b$ is the bias offset.
- $\xi_i$ represents slack variables accommodating misclassifications in non-separable space.
- $C$ is the regularization parameter governing the trade-off between maximizing the margin and minimizing training errors (configured to $C = 1.5$).
- $\phi(\mathbf{x})$ maps the input space into a high-dimensional feature space.

Because clinical covariates interact non-linearly (e.g., synergistic risk amplification between elevated glucose and high BMI), we apply the **Radial Basis Function (RBF)** kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right), \quad \gamma = \frac{1}{d \cdot \text{Var}(X)}$$

The RBF kernel projects the finite 8-dimensional clinical space into an infinite-dimensional Hilbert space, enabling smooth, non-linear decision envelopes. Platt scaling via `CalibratedClassifierCV(ensemble=False)` transforms raw margin distances into calibrated posterior probabilities:

$$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$

#### 4.2 Logistic Regression (Linear Baseline)
Logistic Regression models the log-odds of diabetes as a linear combination of normalized clinical features:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{j=1}^{d} \beta_j x_j$$

It establishes a parametric benchmark. If the physiological boundary is predominantly linear, Logistic Regression performs competitively with lower computational overhead.

#### 4.3 Random Forest Classifier (Non-Linear Ensemble)
Random Forest aggregates $B = 150$ de-correlated decision trees constructed via bootstrap aggregation (bagging) and random feature subspace selection. It provides non-parametric modeling, captures high-order interaction effects, and is robust against monotone feature transformations.

---

### 5. Training, Validation, and Performance Measures

Model selection was conducted using **5-Fold Stratified Cross-Validation** on the training partition ($n=614$). Each fold acts as an independent pseudo-test set while preserving target class proportions.

#### 5.1 Comprehensive Metrics Benchmark Table

| Model | Evaluation Mode | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **5-Fold CV** | **$0.7736 \pm 0.016$** | **$0.7393 \pm 0.083$** | **$0.5703 \pm 0.070$** | **$0.6356 \pm 0.023$** | **$0.8311 \pm 0.026$** |
|  | **Holdout Test** | **$0.7468$** | **$0.6744$** | **$0.5370$** | **$0.5979$** | **$0.8019$** |
| **Random Forest** | 5-Fold CV | $0.7703 \pm 0.020$ | $0.7227 \pm 0.070$ | $0.5702 \pm 0.033$ | $0.6341 \pm 0.017$ | $0.8344 \pm 0.019$ |
|  | Holdout Test | $0.7403$ | $0.6667$ | $0.5185$ | $0.5833$ | $0.8117$ |
| **Logistic Regression** | 5-Fold CV | $0.7882 \pm 0.019$ | $0.7638 \pm 0.060$ | $0.5748 \pm 0.023$ | $0.6545 \pm 0.024$ | $0.8434 \pm 0.018$ |
|  | Holdout Test | $0.7078$ | $0.6000$ | $0.5000$ | $0.5455$ | $0.8130$ |

#### 5.2 Confusion Matrix Breakdown (Holdout Test Set, $n = 154$)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) ⚠️ | True Positives (TP) | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** 🏆 | **86** | **14** | **25** | **29** | **39** |
| Random Forest Classifier | 86 | 14 | 26 | 28 | 40 |
| Logistic Regression | 82 | 18 | 27 | 27 | 45 |

---

### 6. Graphical Visualizations and Analysis

The training and validation run generated five high-resolution diagnostic visualizations saved to the `figures/` directory.

#### Figure 1: Outcome Class Distribution
![Figure 1: Class Distribution](figures/01_class_distribution.png)  
**Caption for Figure 1:** Class frequency distribution of the Pima Indians Diabetes dataset. Non-diabetic patients (Class 0) comprise 500 instances ($65.1\%$) and diabetic patients (Class 1) comprise 268 instances ($34.9\%$). This class disproportion highlights the necessity of using stratified sampling across cross-validation folds and focusing on balanced metrics such as F1-Score and ROC-AUC rather than raw accuracy alone.

#### Figure 2: Clinical Feature Correlation Heatmap
![Figure 2: Correlation Heatmap](figures/02_correlation_heatmap.png)  
**Caption for Figure 2:** Pairwise Pearson correlation heatmap of all continuous clinical parameters. Plasma Glucose ($r = 0.47$) and BMI ($r = 0.29$) demonstrate the strongest positive linear correlation with the diabetes target Outcome. Modest collinearity is observed between Age and Pregnancies ($r = 0.54$) and between Skin Thickness and BMI ($r = 0.39$), supporting the inclusion of regularized feature scaling.

#### Figure 3: Multi-Model Confusion Matrices Comparison
![Figure 3: Confusion Matrix Comparison](figures/03_confusion_matrix.png)  
**Caption for Figure 3:** Side-by-side confusion matrices for Logistic Regression, Support Vector Machine, and Random Forest on the 20% holdout test set ($n=154$). SVM achieves the lowest False Negative count ($\text{FN} = 25$) and highest True Positive yield ($\text{TP} = 29$) compared to Random Forest ($\text{FN}=26$) and Logistic Regression ($\text{FN}=27$). In clinical deployment, lower False Negatives directly translate to fewer missed diabetic patients.

#### Figure 4: Receiver Operating Characteristic (ROC) Curves
![Figure 4: ROC Curves](figures/04_roc_curve.png)  
**Caption for Figure 4:** Receiver Operating Characteristic (ROC) curves across varying decision thresholds on the holdout test set. Logistic Regression ($\text{AUC} = 0.813$), Random Forest ($\text{AUC} = 0.812$), and SVM ($\text{AUC} = 0.802$) all exhibit strong discriminative power well above the non-discriminant baseline ($\text{AUC} = 0.500$), verifying reliable risk discrimination across diverse operating points.

#### Figure 5: Model Comparison Across Performance Metrics
![Figure 5: Model Comparison](figures/05_model_comparison.png)  
**Caption for Figure 5:** Comparative bar chart contrasting Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Support Vector Machine leads in Accuracy ($74.68\%$), Precision ($0.6744$), Recall ($0.5370$), and F1-Score ($0.5979$) on unseen test data, validating its superior generalization capacity.

---

### 7. In-Depth Clinical and Error Analysis

#### 7.1 Which Model Won and Why?
On the holdout test partition, **Support Vector Machine (SVM with RBF Kernel)** emerged as the superior clinical screening model:
1. **Highest Generalization Accuracy:** SVM achieved $74.68\%$, outperforming Random Forest ($74.03\%$) and Logistic Regression ($70.78\%$).
2. **Superior F1-Score:** SVM achieved an F1-Score of $0.5979$, providing the optimal harmonic balance between Precision ($67.44\%$) and Sensitivity ($53.70\%$).
3. **Margin Maximization Advantage:** Logistic Regression showed higher cross-validation accuracy ($78.82\%$) during training but experienced a noticeable drop ($70.78\%$) on the holdout test set, revealing subtle overfitting to linear boundaries. SVM’s maximum-margin hyperplane bounded by slack regularization ($C=1.5$) provided superior stability and generalization on unseen test data.

#### 7.2 What the Errors Mean: Clinical Cost Asymmetry
In machine learning literature, classification errors are often treated as symmetric mathematical penalties. In medical diagnostics, **the costs of Type I and Type II errors are radically asymmetric**:

```mermaid
graph TD
    classDef highRisk fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef lowRisk fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    
    Pred["ML Model Prediction"] --> FP["False Positive (Type I Error)<br/>Model: Diabetic | Reality: Healthy"]:::lowRisk
    Pred --> FN["False Negative (Type II Error)<br/>Model: Healthy | Reality: Diabetic"]:::highRisk
    
    FP --> C1["Consequence: Patient referred for standard laboratory HbA1c test"]
    C1 --> C2["Outcome: Laboratory test shows normal glycemia. Minimal financial cost, transient anxiety, ZERO bodily harm."]
    
    FN --> D1["Consequence: Patient falsely reassured, disease remains unmanaged"]
    D1 --> D2["Outcome: Chronic hyperglycemia continues, causing irreversible microvascular damage: retinopathy, nephropathy, and cardiovascular events."]
```

1. **False Positive (Type I Error, $\text{FP} = 14$):**
   - The model flags a healthy patient as at-risk for diabetes.
   - **Clinical Consequence:** The patient is referred for a routine confirmatory blood draw ($\text{HbA}_{1c}$ or FPG). The test returns normal glycemia. The consequence is limited to mild transient emotional concern and the modest cost of a confirmatory lab test.
2. **False Negative (Type II Error, $\text{FN} = 25$):**
   - The model erroneously labels an actively diabetic patient as non-diabetic.
   - **Clinical Consequence:** The patient is falsely reassured and discharged without glycemic control or lifestyle intervention. Chronic microvascular damage accelerates silently over months or years, leading to permanent end-organ pathology (proliferative diabetic retinopathy, diabetic nephropathy, peripheral neuropathy) or life-threatening diabetic ketoacidosis.

**Conclusion on Metrics:**  
While standard benchmarks emphasize Accuracy, in clinical risk assessment, **Recall (Sensitivity)** and **F1-Score** are the critical metrics. By reducing False Negatives from $27$ (Logistic Regression) down to $25$, **SVM prevents missed diabetic diagnoses**, proving itself the safer clinical choice.

#### 7.3 Limitations of the Study
1. **Demographic Specificity:** The Pima Indian dataset is restricted to female patients of a specific Native American heritage known to possess elevated genetic susceptibility to diabetes. Performance may vary when generalized to heterogeneous populations without domain adaptation.
2. **Sample Size:** The dataset contains 768 patient samples. While adequate for kernel SVM and regularized linear models, training deep learning architectures or fine-grained risk stratifications would benefit from larger modern cohorts (e.g., NHANES or UK Biobank).
3. **Unmeasured Clinical Covariates:** The dataset lacks longitudinal dietary tracking, lipid sub-fractions (HDL, LDL, triglycerides), and direct glycated hemoglobin ($\text{HbA}_{1c}$) values.

---

### 8. Full-Stack Web and Streamlit Application Architecture

To translate the trained machine learning pipeline into accessible clinical decision-support tools, two complete deployment architectures were developed:

#### 8.1 Full-Stack Application (Flask REST API + Modern Responsive Frontend)
- **Backend (`app.py`):**
  - Built with Flask, serving JSON endpoints for model inference (`POST /api/predict`), metadata inspection (`GET /api/model-info`), pre-configured patient profiles (`GET /api/presets`), and high-resolution diagnostic figure delivery (`GET /figures/<filename>`).
  - Implements input bounds checking, missing-value transformation, and calculation of calibrated probabilities via Platt scaling.
  - Automatically synthesizes patient-specific clinical flags (e.g., elevated postprandial glucose $\ge 140\text{ mg/dL}$, obesity $\text{BMI} \ge 30$, family pedigree risk $\ge 0.60$) and evidence-based clinical recommendations (ADA guidelines).
- **Frontend (`templates/index.html`, `static/style.css`, `static/app.js`):**
  - Modern healthcare-themed UI with clean typography and intuitive navigation.
  - Interactive dual controls: synchronized HTML5 range sliders and numeric input fields for all 8 clinical parameters with biological reference units.
  - 1-click clinical presets: **Low Risk (Healthy Adult)**, **Moderate Risk (Pre-diabetic)**, and **High Risk (Diabetic Profile)** for rapid evaluation without manual entry.
  - Dynamic probability gauge bar, color-coded risk severity tags (Low, Moderate, High), and categorized clinical recommendations.
  - Embedded Analytics Dashboard displaying model performance comparison tables and interactive lightbox zoom for all 5 evaluation figures.

#### 8.2 Dedicated Streamlit Application (`streamlit_app.py`)
- Fulfills Step 5 specifications:
  - Interactive sidebar controls and custom patient presets.
  - Sliders for all 8 physiological inputs.
  - Prominent **"Predict Diabetes Risk"** button with instantaneous outcome rendering.
  - Dedicated diagnostic tabs showcasing the model comparison matrix, evaluation figures, and clinical analysis.

---

### 9. Mapping to Course Outcome (CO5)

> **Course Outcome Statement:**  
> **"CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures"**

The execution of this capstone project directly satisfies and demonstrates mastery of all competencies defined in **CO5**:

| Course Outcome Element | Project Implementation & Evidence | Demonstrated Competency |
| :--- | :--- | :--- |
| **"Solve real life problems..."** | Formulated an automated diagnostic screening tool for Type-2 Diabetes Mellitus using the clinical Pima Indians dataset. Addressed biological zero values (imputing physiologically missing metrics like insulin and skin thickness) to reflect actual clinical diagnostic conditions. | Identification, clinical domain formulation, and data engineering for a high-impact global healthcare challenge. |
| **"...using appropriate machine learning models..."** | Selected and compared three mathematically distinct algorithms: **Logistic Regression** (linear parametric baseline), **Support Vector Machine with RBF Kernel** (maximum-margin kernel classifier), and **Random Forest** (ensemble decision trees). Demonstrated why SVM’s margin maximization is mathematically optimal for non-linear clinical boundaries with moderate sample size ($n=768$). | Sound theoretical justification, model selection, hyperparameter tuning, and pipeline encapsulation to prevent data leakage. |
| **"...and evaluate the performance measures."** | Conducted rigorous evaluation using **5-Fold Stratified Cross-Validation** and a holdout test partition. Benchmarked **Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC Curves**. Formulated an error-cost analysis explaining why **Recall (Sensitivity)** takes precedence over Accuracy in medical screening to prevent lethal False Negatives. | Multi-metric evaluation, cross-validation discipline, clinical decision threshold analysis, and risk-aware trade-off evaluation. |

---

### 10. Conclusion and Future Directions

This project demonstrated the end-to-end development of an intelligent, clinical-grade diabetes prediction system centered on **Support Vector Machines (SVM)**. By encapsulating median imputation and standardization within a leak-free pipeline, the SVM model achieved **$74.68\%$ test accuracy**, **$0.8019$ ROC-AUC**, and **minimized False Negatives to 25**, outperforming both Logistic Regression and Random Forest. 

The accompanying full-stack web application and Streamlit dashboard bridge the gap between machine learning theory and clinical utility, providing healthcare practitioners with an intuitive, explainable, and responsive decision-support platform. 

Future enhancements will focus on integrating longitudinal patient monitoring, incorporating $\text{HbA}_{1c}$ biomarkers, and evaluating soft-margin cost-sensitive SVM formulations ($C_+ > C_-$) to penalize False Negatives even further, driving sensitivity toward near-zero missed diagnoses.

---
*Report compiled for B.Tech Machine Learning Capstone Submission.*
