# 🩺 Pima Diabetes Clinical Risk Predictor & Personalized Care Advisory

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-15B064?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end clinical machine learning solution designed to assess patient diabetes risk, mitigate class imbalance with anti-leakage resampling, benchmark 9 supervised classification algorithms, and provide personalized post-prediction medical next steps and dietary meal plans.

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Dataset Architecture](#-dataset-architecture)
- [Two-Stage Engineering Workflow](#-two-stage-engineering-workflow)
  - [Stage 1: Dataset Understanding & Clinical Preprocessing](#stage-1-dataset-understanding--clinical-preprocessing)
  - [Stage 2: Supervised ML Benchmarking & Imbalance Handling](#stage-2-supervised-ml-benchmarking--imbalance-handling)
- [Model Performance Benchmark Leaderboard](#-model-performance-benchmark-leaderboard)
- [Interactive Clinical Application (`app.py`)](#-interactive-clinical-application-apppy)
  - [What To Do Next (Clinical & Diagnostic Workup)](#what-to-do-next-clinical--diagnostic-workup)
  - [What To Eat (Personalized Medical Nutrition Plan)](#what-to-eat-personalized-medical-nutrition-plan)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & How to Run](#-installation--how-to-run)
- [Future Scope & Next-Gen Innovations](#-future-scope--next-gen-innovations)

---

## 🚀 Executive Summary
Early detection of type 2 diabetes and prediabetes significantly reduces the incidence of long-term microvascular and macrovascular complications. This system delivers:
- **High-Performance Diagnostic Modeling**: Optimized **Gradient Boosting** achieving **88.96% Accuracy**, **83.64% Precision**, **85.19% Recall**, and **0.9481 ROC-AUC** (with 5-Fold Cross-Validation AUC of **0.9579**).
- **Leak-Free Imbalance Handling**: Utilizes **BorderlineSMOTE** exclusively on the training split to synthesize minority samples along decision boundaries without test set contamination.
- **Automated Patient Report Ingestion**: Native parser for pathology and lab reports (**PDF**, **CSV**, **TXT**, **Excel**, **JSON**) that automatically extracts clinical biomarkers.
- **3-Tier Severity Triage & Conditional Action Plans**: Categorizes positive diabetic predictions into **Low**, **Moderate**, and **High** severity levels, prescribing medical next steps, clinical urgency timelines, and culturally tailored Indian dietary plans (Vegetarian & Non-Vegetarian) **exclusively when diabetes is predicted**.

### 🔬 Clinical Feature Ranges & Diabetes Prediction Levels
Below are the physiological biomarker ranges and clinical thresholds utilized by the system to evaluate patient status and classify diabetes severity levels:

| Input Feature | Clinical Unit | Normal / Healthy Range | Low Severity Diabetes (Early / Borderline) | Moderate Severity Diabetes (Established Glycemic Strain) | High Severity Diabetes (Critical / Advanced) | System Input Boundaries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plasma Glucose** | mg/dL | **70 – 99** (Fasting) / **< 140** (Random) | **100 – 144** mg/dL *(Mild elevation / Impaired glucose)* | **145 – 179** mg/dL *(Marked hyperglycemia)* | **$\ge 180$** mg/dL *(Severe hyperglycemia)* | `0 – 250` mg/dL |
| **Diastolic Blood Pressure** | mmHg | **60 – 79** mmHg *(Optimal)* | **70 – 79** mmHg *(Normal)* | **80 – 89** mmHg *(Prehypertension)* | **$\ge 90$** mmHg *(Stage 2 Hypertension)* | `0 – 140` mmHg |
| **Body Mass Index (BMI)** | kg/m² | **18.5 – 24.9** kg/m² *(Healthy)* | **25.0 – 29.9** kg/m² *(Overweight)* | **30.0 – 34.9** kg/m² *(Class I Obesity)* | **$\ge 35.0$** kg/m² *(Class II+ Severe Obesity)* | `0.0 – 60.0` kg/m² |
| **2-Hour Serum Insulin** | $\mu\text{U/ml}$ | **2.6 – 24.9** $\mu\text{U/ml}$ *(Normal fasting)* | **25 – 99** $\mu\text{U/ml}$ *(Mild hyperinsulinemia)* | **100 – 159** $\mu\text{U/ml}$ *(Elevated insulin resistance)* | **$\ge 160$** $\mu\text{U/ml}$ *(Severe insulin resistance)* | `0 – 900` $\mu\text{U/ml}$ |
| **Triceps Skinfold Thickness**| mm | **10 – 25** mm *(Normal subcutaneous fat)* | **20 – 25** mm *(Baseline)* | **26 – 34** mm *(Elevated adiposity)* | **$\ge 35$** mm *(High subcutaneous adiposity)* | `0 – 100` mm |
| **Patient Age** | Years | **21 – 34** Years *(Low metabolic baseline)* | **35 – 44** Years *(Emerging demographic risk)* | **45 – 54** Years *(Substantial demographic risk)* | **$\ge 55$** Years *(High cardiovascular & glycemic risk)*| `21 – 90` Years |
| **Number of Pregnancies** | Count | **0 – 2** Pregnancies | **2 – 3** Pregnancies | **4 – 5** Pregnancies | **$\ge 6$** Pregnancies *(Gestational stress history)* | `0 – 20` |
| **Diabetes Pedigree Function**| Index | **0.05 – 0.39** *(Low genetic predisposition)* | **0.40 – 0.54** *(Moderate genetic risk)* | **0.55 – 0.74** *(Strong genetic risk)* | **$\ge 0.75$** *(Critical hereditary cluster)* | `0.05 – 2.50` |

#### 🎯 Severity Stratification Logic Summary:
- **Negative (Normal / Non-Diabetic)**: Model probability $< 50\%$ and key biomarkers within physiological thresholds. Suggestion and action plan tabs are intentionally hidden to confirm healthy baseline.
- **🟡 Low Severity Diabetes**: Predicted positive ($50\% \le \text{Prob} < 70\%$) with early-stage glycemic elevation ($\text{Glucose} < 145$ mg/dL); high clinical potential for complete reversal/stabilization via 30–40 min daily exercise and nutrition reset.
- **🟠 Moderate Severity Diabetes**: Model probability $\ge 70\%$ or marked glucose ($145 - 179$ mg/dL) or obesity strain ($\text{BMI} \ge 30$ with elevated glucose); requires physician visit within 7–10 days and oral medication evaluation.
- **🔴 High Severity Diabetes**: Severe hyperglycemia ($\text{Glucose} \ge 180$ mg/dL), critical probability ($\ge 92\%$), severe obesity strain ($\text{BMI} \ge 35$ & Glucose $\ge 165$), or severe insulin resistance ($\text{HOMA-IR} \ge 75$); requires urgent specialist consultation within 24–48 hours.

---

## 📊 Dataset Architecture
The project utilizes the **Pima Indians Diabetes Database** sourced from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK):
- **Cohort**: 768 female patients of Pima Indian heritage aged $\ge 21$ years.
- **Target (`Outcome`)**: Binary classification (0 = Non-Diabetic, 1 = Diabetic).
- **Class Distribution**: 500 Non-Diabetic ($\sim 65.1\%$) vs 268 Diabetic ($\sim 34.9\%$) — exhibiting a $1.87 : 1$ class imbalance.

### Physiological Attributes & Biological Zeroes
In the raw dataset, missing biological measurements were encoded as `0`. A plasma glucose, diastolic blood pressure, triceps skin fold, serum insulin, or BMI of zero is physiologically incompatible with life.

| Feature | Raw Invalid 0s | Missing % | Biological Justification |
| :--- | :---: | :---: | :--- |
| `Glucose` | 5 | 0.7% | Fasting blood glucose is vital for cellular respiration; 0 is fatal. |
| `BloodPressure` | 35 | 4.6% | Diastolic pressure below 30–40 mmHg indicates cardiogenic shock. |
| `SkinThickness` | 227 | 29.6% | Represents triceps subcutaneous fat caliper measurement. |
| `Insulin` | 374 | 48.7% | 2-hour serum insulin after oral glucose challenge; frequently unrecorded. |
| `BMI` | 11 | 1.4% | Body mass index must be positive. |
| `Pregnancies` | 111 | 14.5% | Biologically valid (nulliparous women). |
| `DiabetesPedigreeFunction`| 0 | 0.0% | Hereditary family history genetic risk score. |
| `Age` | 0 | 0.0% | Patient age in years. |

---

## 🛠️ Two-Stage Engineering Workflow

```
                                    +-----------------------+
                                    |     diabetes.csv      |
                                    +-----------------------+
                                                |
                                                v
                              +-----------------------------------+
                              |           STAGE 1                 |
                              |   stage1.ipynb (Notebook)         |
                              | - Biological Zero Identification  |
                              | - Cohort-Conditioned Imputation   |
                              | - IQR Outlier Treatment           |
                              | - Clinical Feature Engineering    |
                              +-----------------------------------+
                                                |
                                                v
                              +-----------------------------------+
                              |     diabetes_preprocessed.csv     |
                              +-----------------------------------+
                                                |
                                                v
                              +-----------------------------------+
                              |           STAGE 2                 |
                              |   stage2.ipynb (Notebook)         |
                              | - Stratified 80/20 Train-Test     |
                              | - BorderlineSMOTE Resampling      |
                              | - StandardScaler Normalization    |
                              | - 5-Fold Stratified CV GridSearch |
                              | - 9 Supervised ML Classifiers     |
                              +-----------------------------------+
                                                |
                                                v
                              +-----------------------------------+
                              |         DEPLOYMENT                |
                              |   app.py (Streamlit Web App)      |
                              | - Real-time Risk Prediction       |
                              | - Diagnostic Action Roadmap       |
                              | - Personalized Diet & Meal Plan   |
                              +-----------------------------------+
```

### Stage 1: Dataset Understanding & Clinical Preprocessing
**Notebook**: [`stage1.ipynb`](stage1.ipynb)

1. **Biomarker Manifold Imputation**:
   - Invalid biological zeros are converted to `NaN`.
   - Missing values are imputed using **cohort-conditioned median imputation** (preserving distinct distributions between diabetic and non-diabetic cohorts), with a serialized global median fallback for unseen clinical inference.
2. **IQR Outlier Treatment**:
   - Outliers are capped using conservative Interquartile Range boundaries ($Q_1 - 2.5 \times \text{IQR}$, $Q_3 + 2.5 \times \text{IQR}$) to avoid skewing distance-based algorithms.
3. **High-Impact Clinical Feature Engineering**:
   - `Glucose_Risk`: Fasting blood glucose $\ge 140\text{ mg/dL}$ (ADA clinical criterion).
   - `Glucose_High_Borderline`: Impaired fasting glucose indicator ($\ge 120\text{ mg/dL}$).
   - `BMI_Obese`: WHO Class I+ obesity threshold ($\text{BMI} \ge 30\text{ kg/m}^2$).
   - `Age_Risk`: Age risk threshold ($\ge 35\text{ years}$).
   - `Insulin_Glucose_Ratio`: Biomarker representing metabolic insulin resistance.
   - `Glucose_BMI`: Synergistic metabolic syndrome interaction score.
   - `Pedigree_Age`: Family genetic pedigree weighted by chronological age.
   - `HOMA_IR`: Homeostatic Model Assessment surrogate $(\text{Glucose} \times \text{Insulin})/405$.
4. **Data Export**:
   - Outputs finalized clean dataset: [`diabetes_preprocessed.csv`](diabetes_preprocessed.csv) (16 features + 1 target).

---

### Stage 2: Supervised ML Benchmarking & Imbalance Handling
**Notebook**: [`stage2.ipynb`](stage2.ipynb) | **CLI Pipeline**: [`train_models.py`](train_models.py)

1. **Anti-Leakage Data Partitioning**:
   - Stratified train-test split ($80\%$ train, $20\%$ test) preserves class proportions.
   - Feature scaling (`StandardScaler`) is fitted strictly on the training set and applied to the test set.
2. **BorderlineSMOTE Resampling**:
   - Traditional SMOTE generates synthetic points indiscriminately, which degrades precision on noisy clinical datasets.
   - **BorderlineSMOTE** identifies minority class instances located near the decision boundary (borderline points) and generates synthetic samples only where classification ambiguity exists.
   - Resampling is applied **exclusively to the training split**; the test set remains $100\%$ untouched.
3. **Hyperparameter Tuning via 5-Fold Stratified Cross-Validation**:
   - Exhaustive grid search (`GridSearchCV`) evaluates all parameter combinations using `roc_auc` scoring.

---

## 📈 Model Performance Benchmark Leaderboard

Evaluated on the unseen test set ($n=154$, 54 positive, 100 negative):

| Rank | Supervised Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC | 5-Fold CV AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting** ⭐ | **88.96%** | **83.64%** | **85.19%** | **91.00%** | **0.8440** | **0.9481** | **0.9579** |
| 🥈 | **Voting Ensemble** (Soft) | **88.31%** | **82.14%** | **85.19%** | **90.00%** | **0.8364** | **0.9430** | **0.9583** |
| 🥉 | **XGBoost Classifier** | **87.66%** | **81.82%** | **83.33%** | **90.00%** | **0.8257** | **0.9404** | **0.9636** |
| 4 | **Random Forest** | 87.01% | 79.31% | 85.19% | 88.00% | 0.8214 | 0.9431 | 0.9586 |
| 5 | **Decision Tree** | 86.36% | 82.35% | 77.78% | 91.00% | 0.8000 | 0.9098 | 0.9300 |
| 6 | **Support Vector Machine (SVM)** | 83.77% | 75.44% | 79.63% | 86.00% | 0.7748 | 0.9187 | 0.9189 |
| 7 | **Logistic Regression** | 77.92% | 62.82% | 90.74% | 71.00% | 0.7424 | 0.8306 | 0.8623 |
| 8 | **K-Nearest Neighbors (KNN)** | 75.97% | 60.76% | 88.89% | 69.00% | 0.7218 | 0.8689 | 0.9348 |
| 9 | **Naive Bayes (GaussianNB)** | 74.68% | 60.87% | 77.78% | 73.00% | 0.6829 | 0.8140 | 0.8316 |

> **Key Clinical Takeaway**: In clinical diagnostics, avoiding false negatives (missed diabetic patients) while minimizing false alarms is paramount. **Gradient Boosting** achieved the highest overall diagnostic utility with **88.96% accuracy**, **85.19% recall**, and an exceptional **0.9481 ROC-AUC**.

---

## 💻 Interactive Clinical Application (`app.py`)

Run locally with:
```bash
streamlit run app.py
```

### Key Clinical UI Capabilities:
1. **📄 Upload Patient Diagnostic / Lab Report**:
   - Supports uploading real patient lab reports in **PDF**, **CSV**, **TXT**, **Excel (.xlsx, .xls)**, or **JSON** format.
   - Automatically parses and extracts key clinical biomarkers: Glucose, Blood Pressure, BMI, Insulin, Skin Thickness, Age, Pregnancies, Pedigree, and Patient Name.
   - Interactive verification form allows instant review, editing, or auto-imputation before running the diagnostic model.
   - Includes **5 ready-to-test sample reports** (High Severity PDF, Moderate Severity PDF, Low Severity PDF, Normal PDF, and multi-patient CSV) available for direct download within the UI.

2. **⚙️ Manual Clinical Diagnostic Entry**:
   - High-precision clinical sliders and numeric inputs for direct entry.

3. **🎯 Primary Prediction & Severity Classification (Low, Moderate, High)**:
   - Evaluates whether the patient is **Diabetic** (Positive) or **Non-Diabetic** (Negative).
   - When positive for Diabetes, sub-stratifies into 3 clinical severity tiers:
     - 🔴 **High Severity**: Severe hyperglycemia ($\ge 180$ mg/dL), critical model certainty ($\ge 92\%$), severe obesity strain ($\text{BMI} \ge 35$ & Glucose $\ge 165$), or severe insulin resistance ($\text{HOMA-IR} \ge 75$).
     - 🟠 **Moderate Severity**: Marked elevated glucose ($145 - 179$ mg/dL), model probability $\ge 70\%$, or obesity strain ($\text{BMI} \ge 30$ with Glucose $\ge 135$).
     - 🟡 **Low Severity**: Early-stage borderline glycemic dysregulation ($\text{Glucose} < 145$ mg/dL with early metabolic shift) with high potential for complete lifestyle-driven stabilization/reversal.

4. **📋 Targeted Action Plans (Given Exclusively When Diabetes is Predicted)**:
   - As per clinical triage principles, detailed suggestion and action plan tabs are **rendered only when diabetes is predicted**, personalized directly to that severity level:
     - **Urgent Clinical Action Steps**: Specific doctor consultation timeline (Urgent 24-48h for High, 7-10 days for Moderate, 2-4 weeks for Low), necessary diagnostic workups (HbA1c, microalbuminuria, lipid profile, retinal screening), and home monitoring protocols.
     - **Tailored Vegetarian Nutrition Plan**: Severity-specific everyday Indian diet roadmaps (whole grains, pulses, green vegetables, good foods to eat, foods to strictly eliminate, and 1-day meal schedules).
     - **Tailored Non-Vegetarian Nutrition Plan**: Lean proteins (eggs, skinless poultry, omega-3 rich fish), portion guidelines, and sample meal routines.
     - **Daily Rules & Precautions**: The Indian Half-Plate rule, post-meal walking habits, and emergency red-flag warning signs.

5. **📥 Download Diagnostic Assessment Report (PDF)**:
   - Generates an authenticated clinical diagnostic evaluation summary PDF containing patient demographics, evaluated biomarkers, model prediction probability, assigned severity tier, and full actionable medical roadmap.

---

## 📂 Project Directory Structure

```
Diabetics predictor/
│
├── diabetes.csv                       # Raw Pima Indians dataset
├── diabetes_preprocessed.csv          # Cleaned & feature-engineered dataset
│
├── stage1.ipynb                       # [STAGE 1 NOTEBOOK] EDA & Preprocessing Pipeline
├── stage2.ipynb                       # [STAGE 2 NOTEBOOK] Supervised ML Benchmarking
├── train_models.py                    # Production model training & benchmark CLI script
├── app.py                             # Interactive Streamlit Web Application (Predictor + Upload + Severity)
├── report_parser.py                   # Automated Patient Report Parser (PDF, CSV, TXT, Excel, JSON)
├── pdf_generator.py                   # Downloadable Patient Diagnostic Assessment PDF generator
├── generate_sample_reports.py         # Generator for sample clinical test PDF & CSV reports
│
├── sample_reports/                    # Ready-to-test pre-generated clinical patient reports
│   ├── patient_report_high_risk.pdf
│   ├── patient_report_moderate_risk.pdf
│   ├── patient_report_low_risk.pdf
│   ├── patient_report_normal.pdf
│   └── sample_patient_records.csv
│
├── saved_models/                      # Minimal production serialized artifacts
│   ├── best_model.joblib              # Serialized best model (Gradient Boosting)
│   ├── scaler.joblib                  # Standard scaler fitted on 16 features
│   ├── imputer.joblib                 # Median imputer for missing biological zeros
│   ├── best_model_info.json           # Model leaderboard benchmarks & metadata
│   └── feature_metadata.json          # Schema of original and engineered features
│
└── README.md                          # Comprehensive project documentation
```

---

## ⚙️ Installation & How to Run

### Prerequisites
- Python 3.10+
- pip package manager

### 1. Clone or Open the Repository
```bash
cd "c:\Users\temp.admin\Documents\Diabetics predictor"
```

### 2. Install Required Dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn streamlit joblib
```

### 3. Run the Stage Notebooks
Open VS Code / Jupyter Lab and run:
1. Open [`stage1.ipynb`](stage1.ipynb) $\rightarrow$ Run all cells to explore data and generate `diabetes_preprocessed.csv`.
2. Open [`stage2.ipynb`](stage2.ipynb) $\rightarrow$ Run all cells to benchmark models, visualize ROC curves, and export `saved_models/`.

### 4. Train via CLI (Optional)
```bash
python train_models.py
```

### 5. Launch the Streamlit Diagnostic App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔮 Future Scope & Next-Gen Innovations

1. **Continuous Glucose Monitor (CGM) API Integration**:
   - Connect live Bluetooth or cloud streams from Dexcom G7 / FreeStyle Libre sensors to perform continuous, real-time glycemic variability prediction.
2. **Wearable Health Sync**:
   - Integrate Apple HealthKit and Garmin APIs to ingest daily resting heart rate, active calories, sleep hygiene, and VO2 max as live features.
3. **Computer Vision Meal Logging**:
   - Embed a mobile camera module that photographs meals, runs semantic food segmentation, estimates carbohydrate grams, and predicts postprandial glucose spikes.
4. **Conversational LLM Doctor Copilot**:
   - Integrate an empathetic, medically grounded conversational agent (via Google Gemini) that translates risk probabilities into personalized lifestyle coaching and answers patient dietary questions.
5. **SHAP (SHapley Additive exPlanations) Local Interpretability**:
   - Provide interactive waterfall plots explaining exactly which physiological variables contributed to an individual patient's prediction.

---

## 📜 License & Medical Disclaimer
This software is distributed under the **MIT License**.

> **⚠️ Medical Disclaimer**: This system provides automated statistical predictions for educational and clinical decision-support purposes only. It is **not a substitute for professional medical diagnosis, prescription, or clinical judgment**. Always seek the advice of a qualified healthcare provider regarding any medical condition.
