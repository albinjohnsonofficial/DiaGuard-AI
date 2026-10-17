"""
Diabetes Prediction Using Machine Learning (Support Vector Machine - SVM)
Course Outcome: CO5 - Solve real life problems using appropriate machine learning models and evaluate the performance measures.

Author: B.Tech Machine Learning Project
Topic: Diabetes Prediction Using Machine Learning (Support Vector Machine)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Set plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.autolayout'] = True

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('model', exist_ok=True)

print("=" * 80)
print(" STEP 3: DIABETES PREDICTION MACHINE LEARNING PIPELINE")
print("=" * 80)

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
data_path = 'diabetes.csv'
if not os.path.exists(data_path):
    import urllib.request
    print(f"Downloading dataset to {data_path}...")
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
    urllib.request.urlretrieve(url, data_path)

df = pd.read_csv(data_path)

print(f"\n[1] DATASET OVERVIEW:")
print(f"  - Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  - Column Names: {list(df.columns)}")
print(f"\n[Class Counts - Outcome]:")
class_counts = df['Outcome'].value_counts()
class_percentages = df['Outcome'].value_counts(normalize=True) * 100
for outcome_val, count in class_counts.items():
    label = "Non-Diabetic (Class 0)" if outcome_val == 0 else "Diabetic (Class 1)"
    pct = class_percentages[outcome_val]
    print(f"  - {label}: {count} samples ({pct:.2f}%)")

# ---------------------------------------------------------
# 2. Preprocessing & Physiological Missing Value Handling
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("[2] PREPROCESSING & EXPLORATION:")
print(f"  - Checking explicit null values:\n{df.isnull().sum()}")
duplicates_count = df.duplicated().sum()
print(f"  - Duplicate rows found: {duplicates_count}")

# Check data types and encoding requirement
text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if len(text_cols) == 0:
    print("  - Text / Categorical columns: None found (all features are numeric continuous/discrete).")
else:
    print(f"  - Encoding required for: {text_cols}")

# Clinical domain nuance: Zero values in physiological measurements represent missing values
zero_impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n  - Physiological Zero Value Analysis (physically impossible in living patients):")
for col in zero_impute_cols:
    zeros = (df[col] == 0).sum()
    pct_zeros = (zeros / len(df)) * 100
    print(f"    * {col}: {zeros} zero values ({pct_zeros:.2f}%) -> treated as missing data")

# Prepare feature matrix X and target y
X = df.drop(columns=['Outcome']).copy()
y = df['Outcome'].copy()

# Replace physiologically invalid zeros with NaN so SimpleImputer handles them inside Pipeline
for col in zero_impute_cols:
    X[col] = X[col].replace(0, np.nan)

# ---------------------------------------------------------
# 3. Train-Test Split (80/20 Stratified to avoid data leakage)
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("[3] TRAIN-TEST SPLIT:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"  - Training Set: {X_train.shape[0]} samples ({y_train.value_counts().to_dict()})")
print(f"  - Testing Set : {X_test.shape[0]} samples ({y_test.value_counts().to_dict()})")

# ---------------------------------------------------------
# 4. Model Selection Rationale & Pipeline Setup
# ---------------------------------------------------------
"""
Algorithm Selection Rationale:
1. Logistic Regression:
   - Serves as the classic linear baseline.
   - Provides easily interpretable log-odds coefficients and well-calibrated baseline probabilities.
2. Support Vector Machine (SVM):
   - Chosen as the primary topic algorithm.
   - Effective in high-dimensional feature spaces with moderate sample size (n=768).
   - Utilizes the Radial Basis Function (RBF) kernel to map non-linearly separable clinical
     boundaries into higher dimensional space where optimal maximum-margin hyperplanes can be formed.
   - Regularization parameter C and kernel coefficient gamma effectively control bias-variance trade-off.
3. Random Forest Classifier:
   - Serves as a non-linear ensemble tree-based comparator.
   - Captures complex non-linear feature interactions without assumption of linearity and provides
     inherent resilience to outliers.
"""
from sklearn.calibration import CalibratedClassifierCV

print("\n" + "-" * 60)
print("[4] ALGORITHM SELECTION:")
print("  1. Logistic Regression  : Standard linear baseline with probabilistic log-odds calibration.")
print("  2. Support Vector Machine: Primary focus model; non-linear RBF kernel with Platt probability calibration.")
print("  3. Random Forest        : Non-linear ensemble baseline leveraging bootstrap aggregation.")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (SVM)': CalibratedClassifierCV(
        estimator=SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42),
        ensemble=False
    ),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
}

pipelines = {}
for name, clf in models.items():
    # Pipeline contains SimpleImputer and StandardScaler to prevent ANY data leakage
    pipelines[name] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

# ---------------------------------------------------------
# 5. 5-Fold Stratified Cross-Validation (Model Selection)
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("[5] 5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (TRAINING DATA):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_summary = {}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

for name, pipe in pipelines.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_results_summary[name] = {
        'CV Accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
        'CV Precision': (scores['test_precision'].mean(), scores['test_precision'].std()),
        'CV Recall': (scores['test_recall'].mean(), scores['test_recall'].std()),
        'CV F1': (scores['test_f1'].mean(), scores['test_f1'].std()),
        'CV ROC-AUC': (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std()),
    }
    print(f"\n  * {name}:")
    for metric, (mean, std) in cv_results_summary[name].items():
        print(f"      {metric:14s}: {mean:.4f} (+/- {std:.4f})")

# ---------------------------------------------------------
# 6. Fit Models on Train Set & Evaluate on Holdout Test Set
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("[6] HOLDOUT TEST SET EVALUATION (20% Split):")

test_results = {}
trained_models = {}
y_preds = {}
y_probs = {}

for name, pipe in pipelines.items():
    # Train
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    
    # Predict
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    y_preds[name] = y_pred
    y_probs[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    test_results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'Confusion Matrix': cm.tolist()
    }
    
    print(f"\n  === {name} ===")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}  (Critical metric: False Negatives = {cm[1, 0]})")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Confusion Matrix (TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}):")
    print(f"    [[{cm[0,0]:3d}, {cm[0,1]:3d}],\n     [{cm[1,0]:3d}, {cm[1,1]:3d}]]")

# Save primary SVM model
svm_pipe = trained_models['Support Vector Machine (SVM)']
model_save_path = os.path.join('model', 'svm_diabetes_model.joblib')
joblib.dump(svm_pipe, model_save_path)
print(f"\n[SAVED] SVM Pipeline saved successfully to: {model_save_path}")

# Also save all test results & metadata for app & reporting
metadata = {
    'feature_names': list(X.columns),
    'feature_ranges': {col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'median': float(df[col].median())} for col in X.columns},
    'test_metrics': test_results,
    'cv_metrics': {k: {m: {'mean': float(v[0]), 'std': float(v[1])} for m, v in metrics.items()} for k, metrics in cv_results_summary.items()},
    'model_name': 'Support Vector Machine (SVM - RBF Kernel)',
    'course_outcome': 'CO5 - Solve real life problems using appropriate machine learning models and evaluate the performance measures'
}
with open(os.path.join('model', 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"[SAVED] Metadata saved to: model/model_metadata.json")

# ---------------------------------------------------------
# 7. Generate All 5 Required Visualizations
# ---------------------------------------------------------
print("\n" + "-" * 60)
print("[7] GENERATING COMPREHENSIVE VISUALIZATIONS...")

# Plot 1: Class Distribution
plt.figure(figsize=(7, 5))
palette = ['#2b5c8f', '#d9534f']
ax = sns.countplot(x='Outcome', data=df, palette=palette, hue='Outcome', legend=False)
plt.title('Figure 1: Outcome Class Distribution (Pima Diabetes Dataset)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Diagnostic Outcome (0 = Non-Diabetic, 1 = Diabetic)', fontsize=11)
plt.ylabel('Number of Patients', fontsize=11)
plt.xticks([0, 1], ['Non-Diabetic (Class 0)', 'Diabetic (Class 1)'])
for p in ax.patches:
    h = p.get_height()
    pct = (h / len(df)) * 100
    ax.annotate(f"{int(h)} ({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('figures/01_class_distribution.png', dpi=300)
plt.close()
print("  [+] Saved: figures/01_class_distribution.png")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', vmin=-0.1, vmax=1.0,
            square=True, linewidths=.8, cbar_kws={"shrink": .8})
plt.title('Figure 2: Clinical Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/02_correlation_heatmap.png', dpi=300)
plt.close()
print("  [+] Saved: figures/02_correlation_heatmap.png")

# Plot 3: Confusion Matrices (Multi-model Comparison)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for idx, (name, y_pred) in enumerate(y_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[idx].set_title(f'{name}\n(Recall: {recall_score(y_test, y_pred):.2f})', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xticklabels(['Non-Diabetic', 'Diabetic'])
    axes[idx].set_yticklabels(['Non-Diabetic', 'Diabetic'])
plt.suptitle('Figure 3: Confusion Matrices Comparison (Holdout Test Set)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [+] Saved: figures/03_confusion_matrix.png")

# Plot 4: ROC Curves
plt.figure(figsize=(8, 6))
colors = {'Logistic Regression': '#1f77b4', 'Support Vector Machine (SVM)': '#d62728', 'Random Forest': '#2ca02c'}
for name, y_prob in y_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=colors[name], linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', linewidth=1.5)
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('Figure 4: Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig('figures/04_roc_curve.png', dpi=300)
plt.close()
print("  [+] Saved: figures/04_roc_curve.png")

# Plot 5: Model Comparison Bar Chart
metrics_df = pd.DataFrame({
    name: [
        test_results[name]['Accuracy'],
        test_results[name]['Precision'],
        test_results[name]['Recall'],
        test_results[name]['F1-Score'],
        test_results[name]['ROC-AUC']
    ] for name in models.keys()
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

plt.figure(figsize=(10, 6))
ax = metrics_df.plot(kind='bar', figsize=(10, 6), width=0.75, colormap='viridis')
plt.title('Figure 5: Model Performance Comparison Across Evaluation Metrics', fontsize=13, fontweight='bold', pad=15)
plt.ylabel('Score (0.0 to 1.0)', fontsize=11)
plt.xlabel('Evaluation Metric', fontsize=11)
plt.ylim(0, 1.08)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title="Model", frameon=True)
for p in ax.patches:
    val = p.get_height()
    if val > 0:
        ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val + 0.015),
                    ha='center', va='bottom', fontsize=8, rotation=90)
plt.tight_layout()
plt.savefig('figures/05_model_comparison.png', dpi=300)
plt.close()
print("  [+] Saved: figures/05_model_comparison.png")

print("\n" + "=" * 80)
print(" MACHINE LEARNING PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
