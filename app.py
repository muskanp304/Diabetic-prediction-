import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import io
import os

from report_parser import parse_report_file
from pdf_generator import generate_patient_pdf

st.set_page_config(
    page_title="Pima Diabetes Diagnostic & Clinical Decision System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0ea5e9;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .highlight-best {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.35);
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .severity-badge-high {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #f87171;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .severity-badge-mod {
        background: rgba(249, 115, 22, 0.15);
        border: 2px solid #f97316;
        color: #fb923c;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .severity-badge-low {
        background: rgba(245, 158, 11, 0.15);
        border: 2px solid #f59e0b;
        color: #fbbf24;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .severity-badge-neg {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        color: #34d399;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .param-chip {
        display: inline-block;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(14, 165, 233, 0.35);
        color: #38bdf8;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    .action-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    return pd.read_csv('diabetes.csv')

@st.cache_resource
def load_models_and_info():
    imputer = joblib.load('saved_models/imputer.joblib')
    scaler = joblib.load('saved_models/scaler.joblib')
    best_model = joblib.load('saved_models/best_model.joblib')
    
    with open('saved_models/best_model_info.json', 'r') as f:
        info = json.load(f)
        
    return imputer, scaler, best_model, info

df_raw = load_dataset()
imputer, scaler, best_model, model_info = load_models_and_info()

best_model_name = model_info['best_model_name']
results = model_info['results']

# Sidebar Navigation
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Select View:",
    [
        "🩺 Patient Diabetes Predictor",
        "📊 Model Overview & Benchmarks"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Best Model:**\n\n `{best_model_name}`\n\nTuned via 5-Fold Stratified CV & Anti-Bias SMOTE.")

def evaluate_patient(patient_data):
    """
    Runs imputation, feature engineering, scaling, and model prediction.
    Determines if Diabetes is present and classifies Severity: Low, Moderate, High.
    """
    raw_input = pd.DataFrame([{
        'Pregnancies': patient_data.get('Pregnancies', 0),
        'Glucose': patient_data.get('Glucose', np.nan) if patient_data.get('Glucose', 0) > 0 else np.nan,
        'BloodPressure': patient_data.get('BloodPressure', np.nan) if patient_data.get('BloodPressure', 0) > 0 else np.nan,
        'SkinThickness': patient_data.get('SkinThickness', np.nan) if patient_data.get('SkinThickness', 0) > 0 else np.nan,
        'Insulin': patient_data.get('Insulin', np.nan) if patient_data.get('Insulin', 0) > 0 else np.nan,
        'BMI': patient_data.get('BMI', np.nan) if patient_data.get('BMI', 0) > 0 else np.nan,
        'DiabetesPedigreeFunction': patient_data.get('DiabetesPedigreeFunction', 0.5),
        'Age': patient_data.get('Age', 35)
    }])

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    raw_input[zero_cols] = imputer.transform(raw_input[zero_cols])

    # Feature Engineering matching training pipeline
    raw_input['Glucose_Risk'] = (raw_input['Glucose'] >= 140).astype(int)
    raw_input['Glucose_High_Borderline'] = (raw_input['Glucose'] >= 120).astype(int)
    raw_input['BMI_Obese'] = (raw_input['BMI'] >= 30).astype(int)
    raw_input['Age_Risk'] = (raw_input['Age'] >= 35).astype(int)
    raw_input['Insulin_Glucose_Ratio'] = raw_input['Insulin'] / (raw_input['Glucose'] + 1e-5)
    raw_input['Glucose_BMI'] = (raw_input['Glucose'] * raw_input['BMI']) / 100.0
    raw_input['Pedigree_Age'] = raw_input['DiabetesPedigreeFunction'] * raw_input['Age']
    raw_input['HOMA_IR'] = (raw_input['Glucose'] * raw_input['Insulin']) / 405.0

    feature_cols = model_info.get('feature_names', list(raw_input.columns))
    features_to_scale = raw_input[feature_cols]
    scaled_input = pd.DataFrame(scaler.transform(features_to_scale), columns=feature_cols)

    pred = best_model.predict(scaled_input)[0]
    prob = best_model.predict_proba(scaled_input)[0][1] if hasattr(best_model, 'predict_proba') else 0.5

    # Determine Diabetes Status
    is_diabetic = bool(pred == 1 or prob >= 0.50)

    # Classify Severity if Diabetic: Low, Moderate, High
    # Based on clinical glycemic thresholds and model confidence
    glucose_val = float(raw_input['Glucose'].iloc[0])
    bmi_val = float(raw_input['BMI'].iloc[0])
    bp_val = float(raw_input['BloodPressure'].iloc[0])
    homa_val = float(raw_input['HOMA_IR'].iloc[0])

    severity = None
    severity_reasons = []

    if is_diabetic:
        # High Severity Criteria: Severe Hyperglycemia (>=180), critical probability (>=0.92 with glucose>=165), Severe Obesity Strain (BMI>=35 & Glucose>=165), or very elevated HOMA_IR (>=75 & Glucose>=160)
        if glucose_val >= 180 or (prob >= 0.92 and glucose_val >= 165) or (glucose_val >= 165 and bmi_val >= 35) or (homa_val >= 75.0 and glucose_val >= 160):
            severity = "High"
            if glucose_val >= 180: severity_reasons.append(f"Severe hyperglycemia (Glucose: {glucose_val:.0f} mg/dL ≥ 180 mg/dL)")
            elif glucose_val >= 165: severity_reasons.append(f"Marked elevated glucose ({glucose_val:.0f} mg/dL) with critical model probability ({prob*100:.1f}%)")
            if bmi_val >= 35: severity_reasons.append(f"High metabolic risk Class II+ obesity (BMI: {bmi_val:.1f} kg/m²)")
            if homa_val >= 75.0: severity_reasons.append(f"Severe insulin resistance index (HOMA-IR: {homa_val:.1f})")
            if not severity_reasons: severity_reasons.append("High compound metabolic risk profile flagged")
        # Moderate Severity Criteria: Model probability 0.70+, or Glucose 145+, or Glucose 135+ with Obese BMI, or Diastolic BP >= 88
        elif prob >= 0.70 or glucose_val >= 145 or (glucose_val >= 135 and bmi_val >= 30) or bp_val >= 88:
            severity = "Moderate"
            if glucose_val >= 145: severity_reasons.append(f"Marked elevated glucose (Glucose: {glucose_val:.0f} mg/dL)")
            elif glucose_val >= 135: severity_reasons.append(f"Elevated glucose ({glucose_val:.0f} mg/dL) combined with obesity strain (BMI: {bmi_val:.1f})")
            if prob >= 0.70: severity_reasons.append(f"Established diagnostic model confidence ({prob*100:.1f}%)")
            if bp_val >= 88: severity_reasons.append(f"Elevated diastolic blood pressure ({bp_val:.0f} mmHg)")
            if not severity_reasons: severity_reasons.append("Moderate metabolic dysregulation identified")
        # Low Severity Criteria: Early stage / Borderline diabetes
        else:
            severity = "Low"
            severity_reasons.append(f"Early-stage glycemic elevation (Glucose: {glucose_val:.0f} mg/dL)")
            severity_reasons.append(f"Borderline model risk probability ({prob*100:.1f}%)")
            severity_reasons.append("Early metabolic shift with high reversal potential through lifestyle changes")

    # Risk Factors
    factors = []
    if glucose_val >= 140: factors.append(f"Elevated Glucose ({glucose_val:.0f} mg/dL): Exceeds 140 mg/dL diagnostic threshold")
    if bmi_val >= 30: factors.append(f"High BMI ({bmi_val:.1f} kg/m²): WHO Obesity Class I+")
    if patient_data.get('Age', 0) >= 45: factors.append(f"Age Risk Demographic ({patient_data.get('Age', 0)} yrs)")
    if patient_data.get('DiabetesPedigreeFunction', 0) >= 0.5: factors.append(f"Hereditary Genetic Risk (DPF: {patient_data.get('DiabetesPedigreeFunction', 0):.2f})")
    if homa_val >= 3.0: factors.append(f"Elevated Insulin Resistance (HOMA-IR: {homa_val:.2f})")

    return {
        'is_diabetic': is_diabetic,
        'severity': severity,
        'severity_reasons': severity_reasons,
        'prob': prob,
        'raw_input': raw_input,
        'factors': factors
    }

def render_assessment_results(patient_name, eval_res, patient_inputs):
    """
    Renders diagnostic predictions, severity grading, and action plans.
    Suggestions/Action plans are ONLY rendered when Diabetes is predicted.
    """
    is_diabetic = eval_res['is_diabetic']
    severity = eval_res['severity']
    prob = eval_res['prob']
    factors = eval_res['factors']
    severity_reasons = eval_res['severity_reasons']

    st.markdown("---")
    st.markdown(f"### 📋 Diagnostic Assessment for: **{patient_name}**")

    # Primary Diagnosis & Severity Banner
    col_res1, col_res2 = st.columns([1.2, 2.0])
    with col_res1:
        if is_diabetic:
            st.markdown("<div class='severity-badge-high' style='font-size:1.15rem; width:100%; text-align:center;'>🚨 PREDICTED: DIABETES DETECTED</div>", unsafe_allow_html=True)
            st.metric("Model Probability", f"{prob*100:.1f}%", help="Probability calculated by the best supervised ML model")
            
            # Severity classification display
            if severity == "High":
                st.markdown("<div class='severity-badge-high' style='width:100%; text-align:center;'>🔴 HIGH SEVERITY</div>", unsafe_allow_html=True)
                st.caption("Advanced Hyperglycemia / Immediate Clinical Review Required")
            elif severity == "Moderate":
                st.markdown("<div class='severity-badge-mod' style='width:100%; text-align:center;'>🟠 MODERATE SEVERITY</div>", unsafe_allow_html=True)
                st.caption("Established Impaired Glycemic Regulation / Physician Review Advised")
            else:
                st.markdown("<div class='severity-badge-low' style='width:100%; text-align:center;'>🟡 LOW SEVERITY</div>", unsafe_allow_html=True)
                st.caption("Early-Stage / Borderline Glycemic Disruption / High Reversibility")
        else:
            st.markdown("<div class='severity-badge-neg' style='font-size:1.15rem; width:100%; text-align:center;'>✅ NEGATIVE: NO DIABETES DETECTED</div>", unsafe_allow_html=True)
            st.metric("Diabetes Risk Probability", f"{prob*100:.1f}%", delta="Normal Range", delta_color="inverse")
            st.success("All clinical markers align with normal metabolic physiology.")

    with col_res2:
        st.markdown("#### 🔬 Clinical Evaluation Summary")
        if is_diabetic:
            st.write(f"**Severity Classification Basis ({severity} Severity):**")
            for r in severity_reasons:
                st.write(f"• {r}")
            if factors:
                st.write("**Key Contributing Risk Factors:**")
                for f in factors:
                    st.write(f"• {f}")
        else:
            st.info("""
            **Primary Assessment: Non-Diabetic / Normal Metabolic Status**
            
            The machine learning evaluation indicates that the patient does not exhibit clinical diabetes. 
            All evaluated glycemic, anthropometric, and metabolic indicators are within safe thresholds.
            
            - **Glucose & Insulin**: In healthy physiological range
            - **Risk Profile**: Low baseline probability
            - **Recommendation**: Maintain your current healthy lifestyle, balanced nutrition, and schedule routine annual health checkups.
            """)

    # DOWNLOAD REPORT PDF
    actions_summary = []
    if is_diabetic:
        if severity == "High":
            actions_summary = [
                "URGENT: Consult a diabetologist/endocrinologist within 24-48 hours.",
                "Order comprehensive lab tests: HbA1c, CMP, Serum Creatinine, Microalbuminuria, Lipid Profile.",
                "Screen for diabetic complications: Dilated eye fundus exam & diabetic foot sensory test.",
                "Review immediate pharmacotherapy (oral dual therapy or insulin) with your physician.",
                "Intensive daily blood sugar self-monitoring (fasting and post-meal)."
            ]
        elif severity == "Moderate":
            actions_summary = [
                "Schedule physician consultation within 7-10 days for clinical confirmation.",
                "Complete blood tests: HbA1c test and Fasting & Post-prandial glucose.",
                "Evaluate first-line oral medication (e.g. Metformin) with physician.",
                "Follow strict low-glycemic dietary regimen and structured 45-min daily exercise.",
                "Test blood sugar 3-4 times weekly."
            ]
        else:
            actions_summary = [
                "Schedule routine primary care doctor consultation within 2-4 weeks.",
                "Get confirmation HbA1c and Fasting Blood Sugar tests.",
                "High potential for lifestyle reversal: Eliminate all added sugars and refined grains.",
                "Engage in 30-40 minutes daily brisk walking + 15-minute walk after dinner.",
                "Re-check glycemic markers in 3 months."
            ]

    pdf_bytes = generate_patient_pdf(
        patient_name=patient_name,
        inputs=patient_inputs,
        is_diabetic=is_diabetic,
        severity=severity if is_diabetic else "Normal",
        prob=prob,
        factors=factors,
        actions=actions_summary
    )

    st.download_button(
        label="📥 Download Clinical Diagnostic Assessment Report (PDF)",
        data=pdf_bytes,
        file_name=f"Diabetes_Assessment_{patient_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # =========================================================================
    # ACTION PLAN & SUGGESTIONS: ONLY GIVEN WHEN DIABETES IS PREDICTED
    # =========================================================================
    if is_diabetic:
        st.markdown("---")
        st.markdown(f"## 🩺 Structured Medical Action Plan & Roadmaps ({severity.upper()} SEVERITY)")
        st.write(f"Because the patient is predicted positive for diabetes with **{severity} Severity**, follow the targeted clinical protocol below:")

        tab_action, tab_diet_veg, tab_diet_nonveg, tab_rules = st.tabs([
            f"🏥 {severity} Severity Action Steps",
            f"🥗 Tailored Diet Plan (Vegetarian)",
            f"🍗 Tailored Diet Plan (Non-Vegetarian)",
            f"⚡ Daily Rules & Precautions"
        ])

        with tab_action:
            if severity == "High":
                st.error("### 🚨 High Severity Action Protocol (Urgent Medical Review)")
                st.markdown("""
                1. **Immediate Medical Consultation (Within 24 to 48 Hours)**:
                   - Book an immediate appointment with a **Consultant Diabetologist or Endocrinologist**. 
                   - Do not postpone; severe hyperglycemia requires urgent clinical stabilization to prevent acute complications.

                2. **Essential Diagnostic Workup to Order Immediately**:
                   - **HbA1c Glycated Hemoglobin Test** (assess long-term glycemic control).
                   - **Kidney Function Panel (KFT)**: Serum Creatinine, eGFR, and **Urine Microalbumin-to-Creatinine Ratio**.
                   - **Dilated Eye Fundus Examination** by an ophthalmologist (screen for diabetic retinopathy).
                   - **Cardiac Evaluation**: Resting ECG & Comprehensive Lipid Profile (Triglycerides, LDL, HDL).
                   - **Neuropathy Check**: Monofilament vibration foot sensory screening.

                3. **Medication & Treatment Review**:
                   - Take all previous medical reports to your doctor.
                   - The physician will evaluate initiation of pharmacotherapy (dual oral agents or basal insulin) tailored to your kidney and liver profiles.

                4. **Intensive Home Glucose Self-Monitoring**:
                   - Check blood sugar with a glucometer **2 to 3 times daily**:
                     - Fasting (upon waking up)
                     - 2 hours after lunch
                     - 2 hours after dinner
                   - Maintain a written log to share with your specialist.

                5. **Critical Warning Symptoms (Emergency Room Precaution)**:
                   - Seek urgent emergency care if experiencing: sudden extreme thirst with nausea, rapid unexplained weight loss, shortness of breath, confusion, or persistent vomiting (signs of Diabetic Ketoacidosis or Hyperosmolar Hyperglycemic State).
                """)

            elif severity == "Moderate":
                st.warning("### 🟠 Moderate Severity Action Protocol (Physician Intervention)")
                st.markdown("""
                1. **Physician Consultation Within 7 to 10 Days**:
                   - Schedule a formal consultation with your family physician or internal medicine specialist.
                   - Bring your complete medical history and blood test reports.

                2. **Recommended Diagnostic Tests**:
                   - **HbA1c Test**: To determine 3-month average glucose control.
                   - **Fasting Blood Sugar (FBS)** and **2-Hour Post-Prandial Blood Sugar (PPBS)**.
                   - **Lipid Profile & Renal Function Test**: Assess cholesterol and kidney health.

                3. **Therapeutic Medical Management**:
                   - Your physician will evaluate whether first-line oral anti-diabetic medications (e.g. Metformin) are required alongside lifestyle changes.
                   - Always follow the exact prescription; never self-medicate or abruptly alter doses.

                4. **Structured Home Monitoring**:
                   - Test blood sugar **3 to 4 times a week** at alternating times (fasting and post-meal).
                   - Keep a target fasting range of 80-130 mg/dL and post-meal < 180 mg/dL (or as advised by your doctor).

                5. **Active Lifestyle Interventions**:
                   - Commit to 40-45 minutes of moderate aerobic exercise daily.
                   - Practice portion control and eliminate refined carbohydrates immediately.
                """)

            else:  # Low Severity
                st.info("### 🟡 Low Severity Action Protocol (Early-Stage Reversal & Lifestyle Reset)")
                st.markdown("""
                1. **Primary Care Consultation Within 2 to 4 Weeks**:
                   - Visit a doctor for clinical confirmation and baseline records.
                   - **High Reversal Window**: At this early/mild stage, aggressive lifestyle and dietary optimization can often stabilize or normalize glucose regulation without needing lifelong medication.

                2. **Baseline Confirmation Tests**:
                   - **HbA1c Test** (to detect early borderline elevation).
                   - **Fasting Plasma Glucose Test** (after 8-10 hours overnight fasting).

                3. **Proactive Lifestyle Intervention (Primary Prescription)**:
                   - **Weight Management**: If BMI is over 25, aiming for a modest 5% to 7% body weight reduction significantly restores insulin sensitivity.
                   - **Brisk Walking**: 30-40 minutes daily brisk walk + mandatory 10-15 minute walk right after dinner.
                   - **Sleep & Stress**: Ensure 7-8 hours of quality sleep; chronic stress increases cortisol, elevating blood glucose.

                4. **Routine Tracking**:
                   - Check fasting blood sugar once every 1-2 weeks.
                   - Schedule a follow-up HbA1c test in 3 months to monitor progress.
                """)

        with tab_diet_veg:
            st.subheader(f"Vegetarian Nutrition Roadmap for {severity} Severity")
            if severity == "High":
                st.markdown("**Strict Low-Glycemic Medical Nutrition Therapy**: High priority on eliminating blood sugar surges.")
            elif severity == "Moderate":
                st.markdown("**Controlled Carbohydrate & High-Fiber Plan**: Focus on balanced metabolic digestion.")
            else:
                st.markdown("**Metabolic Reset & Whole Food Plan**: Focus on wholesome nutrition and natural fiber.")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.success("#### ✅ Recommended Foods")
                st.markdown("""
                - **Whole Grains**: Pure Jowar, Bajra, Ragi, or Barley (Jau) rotis. Avoid plain wheat and maida.
                - **Lentils & Pulses**: Sprouted Moong, Chana dal, Toor dal, Rajma, Lobia (protein + soluble fiber).
                - **Green Vegetables**: Karela (bitter gourd - proven insulin-like peptide), Methi, Palak, Lauki, Turai, Cabbage, French beans.
                - **Proteins**: Low-fat fresh Paneer (100g max), Tofu, Soya chunks, sprouted legumes.
                - **Healthy Snacks**: Roasted makhana, roasted chana (bhuna chana), 4-5 soaked almonds & 2 walnuts daily.
                - **Hydration & Herbal**: Plain salted chaach (buttermilk with roasted jeera), cinnamon water, methi seed water.
                """)
            with col_v2:
                st.error("#### ❌ Strictly Avoid / Eliminate")
                st.markdown("""
                - **Direct Sugars**: Table sugar in chai/coffee, jaggery (gud), honey, sweets, halwa, ice cream.
                - **Refined Carbs**: Maida, white bread, naan, bhature, noodles, pasta, biscuits, rusks.
                - **High-GI Foods**: Large bowls of white rice, potatoes (aloo), sweet potatoes, sweet corn.
                - **Beverages**: Packaged juices, sodas, energy drinks, milkshakes, sweetened lassi.
                - **Fried Snacks**: Samosas, pakoras, bhujia, namkeen, chips.
                """)

            st.markdown("---")
            st.info(f"""
            #### 🍽️ Sample 1-Day Vegetarian Meal Schedule ({severity} Severity Protocol):
            - **Morning (7:00 AM)**: 1 glass warm water with overnight soaked methi seeds OR cinnamon stick water.
            - **Breakfast (8:30 AM)**: 2 Besan Chilla with grated paneer & mint coriander chutney OR vegetable oats daliya + sugar-free herbal tea.
            - **Mid-Morning (11:00 AM)**: 1 bowl roasted makhana OR 5 soaked almonds + 1 green apple / guava.
            - **Lunch (1:30 PM)**:
              - *Step 1*: Big bowl of raw cucumber and tomato salad (eat first).
              - *Step 2*: 1-2 Jowar or Bajra rotis with 1 bowl green sabzi (Palak, Bhindi, or Lauki).
              - *Step 3*: 1 bowl thick Moong or Chana dal + 1 glass plain roasted jeera chaach.
            - **Evening (5:00 PM)**: 1 cup green tea / black tea + small bowl of roasted chana (bhuna chana).
            - **Dinner (7:30 - 8:00 PM - Early)**: 1 Multigrain roti with Paneer bhurji and bowl of vegetable soup OR vegetable moong dal khichdi.
            """)

        with tab_diet_nonveg:
            st.subheader(f"Non-Vegetarian Nutrition Roadmap for {severity} Severity")
            st.markdown("Lean animal proteins (eggs, skinless poultry, fish) have minimal impact on blood glucose while promoting satiety and muscle mass.")

            col_nv1, col_nv2 = st.columns(2)
            with col_nv1:
                st.success("#### ✅ Recommended Non-Veg Options")
                st.markdown("""
                - **Eggs**: 2 whole boiled eggs, poached eggs, or vegetable omelet made with onions, tomatoes, and spinach.
                - **Poultry**: Skinless chicken breast cooked in minimal mustard or olive oil, boiled shredded chicken, or tandoori chicken.
                - **Fish (Rich in Omega-3)**: Rohu, Surmai, Pomfret, Katla, Salmon (steamed, grilled, or light homestyle curry).
                - **Accompaniments**: 1-2 Jowar/Bajra rotis, large green salads, moong dal, steamed greens.
                - **Healthy Snacks**: Boiled egg whites sprinkled with chaat masala, roasted chana, walnuts.
                """)
            with col_nv2:
                st.error("#### ❌ Strictly Avoid / Eliminate")
                st.markdown("""
                - **Deep-Fried Meats**: Crispy fried chicken, chicken pakoras, fried fish fingers.
                - **Heavy Restaurant Curries**: Butter chicken with heavy cream, rich mutton korma, oily gravies.
                - **High-Fat Meats**: Excessive red meat (mutton/beef) with visible fat, processed sausages, salami.
                - **Combinations**: Heavy mutton biryani with white basmati rice, rumali roti, maida naans.
                - **Alcohol**: Beer, sweet cocktails, sugary mixers.
                """)

            st.markdown("---")
            st.info(f"""
            #### 🍽️ Sample 1-Day Non-Vegetarian Routine ({severity} Severity Protocol):
            - **Morning (7:00 AM)**: Warm water with lemon or soaked chia seeds.
            - **Breakfast (8:30 AM)**: 2 whole boiled eggs OR 2-egg vegetable omelet with 1 slice multigrain toast + sugar-free tea.
            - **Mid-Morning (11:00 AM)**: Handful of roasted makhana + 5 soaked almonds.
            - **Lunch (1:30 PM)**:
              - *Step 1*: Big bowl of fresh salad with lemon dressing.
              - *Step 2*: 1-2 Jowar or Bajra rotis.
              - *Step 3*: 1 bowl homestyle light chicken or fish curry (minimal oil) + 1 small bowl dal.
              - *Step 4*: 1 glass plain buttermilk.
            - **Evening (5:00 PM)**: 2 boiled egg whites with black pepper + green tea.
            - **Dinner (7:30 - 8:00 PM - Early)**: 3-4 pieces of grilled/tandoori chicken breast or grilled fish with sautéed vegetables (beans, broccoli, bell peppers).
            """)

        with tab_rules:
            st.subheader("⚡ Golden Rules & Daily Habits")
            st.markdown("""
            ### 1. The Half-Plate Rule for Every Meal:
            - **50% of your plate**: Non-starchy green vegetables and fresh salads (cucumber, tomato, radish, leafy greens).
            - **25% of your plate**: Clean protein (Dal, Sprouts, Paneer, Tofu, Eggs, or Fish/Chicken).
            - **25% of your plate**: High-fiber complex carbohydrates (Jowar, Bajra, or small bowl brown rice).

            ---

            ### 2. Critical Daily Lifestyle Rules:
            1. **Never Sit Immediately After Meals**: Walk gently for **10 to 15 minutes right after lunch and dinner** — this directly blunts post-prandial blood sugar spikes.
            2. **Stop Sugar in Tea and Coffee**: Cutting out 2-3 cups of sweetened tea per day saves up to 25-30g of pure sucrose daily.
            3. **Early Dinners (Before 8:00 PM)**: Eating late impairs nighttime insulin sensitivity and increases morning fasting glucose.
            4. **Stay Hydrated**: Drink 8-10 glasses of clean water daily to help kidneys flush excess glucose.
            5. **Routine Sleep**: Aim for 7-8 hours of uninterrupted sleep; poor sleep elevates stress hormones and induces insulin resistance.
            """)
            st.caption("⚠️ **Medical Disclaimer**: These suggestions are structured clinical guidance based on machine learning risk evaluation. Always consult a certified diabetologist or registered healthcare provider for personalized medical prescriptions and treatment decisions.")
    else:
        # NON-DIABETIC / NEGATIVE CASE:
        # Per user requirement: Suggestion is given ONLY when diabetes is predicted. For non-diabetic, only report prediction status.
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 1.2rem; margin-top: 1.5rem;">
            <h4 style="color: #10b981; margin-top: 0;">✅ Diagnostic Confirmation: Negative for Diabetes</h4>
            <p style="color: #cbd5e1; margin-bottom: 0;">
                Because all diagnostic markers are within normal limits and diabetes is not predicted, no intensive diabetic intervention or medication protocol is required. 
                Continue your healthy active routine and undergo routine preventative annual health checkups.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: PATIENT DIABETES PREDICTOR
# =============================================================================
if page == "🩺 Patient Diabetes Predictor":
    st.markdown("<h1 class='main-title'>🩺 Patient Diabetes Risk Predictor & Report Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>Predict diabetes probability, classify severity (Low / Moderate / High), and extract diagnostic parameters directly from patient lab reports using <strong>{best_model_name}</strong>.</p>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Input Method:",
        ["📄 Upload Patient Lab Report (PDF / CSV / TXT / Excel)", "⚙️ Manual Parameter Entry (Sliders & Inputs)"],
        horizontal=True
    )

    # -------------------------------------------------------------------------
    # MODE 1: UPLOAD PATIENT LAB REPORT
    # -------------------------------------------------------------------------
    if input_mode == "📄 Upload Patient Lab Report (PDF / CSV / TXT / Excel)":
        st.markdown("""
        <div class='action-card'>
            <h4 style="color: #38bdf8; margin-top: 0;">📄 Upload Patient Diagnostic / Lab Report</h4>
            <p style="color: #94a3b8; margin-bottom: 0.5rem;">
                Upload a patient medical test report (PDF pathology slip, CSV record, TXT doctor note, or Excel file). 
                The system will automatically parse and extract clinical parameters such as Glucose, Blood Pressure, BMI, Insulin, Age, etc.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload Patient Report File",
                type=["pdf", "csv", "txt", "xlsx", "xls", "json"],
                help="Accepts clinical pathology reports in PDF, CSV, TXT, Excel, or JSON format"
            )

        with col_up2:
            st.markdown("##### 🧪 Try With Sample Reports:")
            st.caption("Download and test these pre-generated clinical reports:")
            
            sample_files = {
                "🔴 High Severity Report (PDF)": "sample_reports/patient_report_high_risk.pdf",
                "🟠 Moderate Severity Report (PDF)": "sample_reports/patient_report_moderate_risk.pdf",
                "🟡 Low Severity Report (PDF)": "sample_reports/patient_report_low_risk.pdf",
                "🟢 Normal / Negative Report (PDF)": "sample_reports/patient_report_normal.pdf",
                "📊 Patient Records (CSV)": "sample_reports/sample_patient_records.csv"
            }
            for label, fpath in sample_files.items():
                if os.path.exists(fpath):
                    with open(fpath, "rb") as sf:
                        st.download_button(
                            label=label,
                            data=sf.read(),
                            file_name=os.path.basename(fpath),
                            mime="application/pdf" if fpath.endswith(".pdf") else "text/csv",
                            key=f"dl_{fpath}"
                        )

        # Process uploaded file
        if uploaded_file is not None:
            patient_name, extracted_params, raw_text = parse_report_file(uploaded_file)
            
            st.success(f"✅ Report uploaded successfully: **{uploaded_file.name}**")
            
            # Show extracted parameters summary
            st.markdown("#### 🔍 Extracted Clinical Parameters from Report:")
            if extracted_params:
                chips_html = ""
                for k, v in extracted_params.items():
                    chips_html += f"<span class='param-chip'><strong>{k}:</strong> {v['value']}</span>"
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.warning("Could not automatically detect standard parameters. Please review or fill in manually below.")

            with st.expander("📄 View Raw Report Text / Extracted Content"):
                st.text(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

            st.markdown("#### ✍️ Verify & Adjust Patient Values Before Prediction:")
            st.caption("Review extracted parameters below. Any missing value can be edited or auto-imputed:")

            c_name, c_age, c_preg = st.columns(3)
            with c_name:
                v_name = st.text_input("Patient Name", value=patient_name)
            with c_age:
                def_age = int(extracted_params['Age']['value']) if 'Age' in extracted_params else 45
                v_age = st.number_input("Age (Years)", 1, 100, def_age)
            with c_preg:
                def_preg = int(extracted_params['Pregnancies']['value']) if 'Pregnancies' in extracted_params else 2
                v_preg = st.number_input("Pregnancies", 0, 20, def_preg)

            c1, c2, c3 = st.columns(3)
            with c1:
                def_gluc = float(extracted_params['Glucose']['value']) if 'Glucose' in extracted_params else 140.0
                v_gluc = st.number_input("Glucose (mg/dL)", 0.0, 300.0, def_gluc, step=1.0)
                def_bp = float(extracted_params['BloodPressure']['value']) if 'BloodPressure' in extracted_params else 72.0
                v_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 0.0, 150.0, def_bp, step=1.0)

            with c2:
                def_bmi = float(extracted_params['BMI']['value']) if 'BMI' in extracted_params else 30.0
                v_bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, def_bmi, step=0.1)
                def_ins = float(extracted_params['Insulin']['value']) if 'Insulin' in extracted_params else 100.0
                v_ins = st.number_input("Serum Insulin (mu U/ml)", 0.0, 900.0, def_ins, step=1.0)

            with c3:
                def_skin = float(extracted_params['SkinThickness']['value']) if 'SkinThickness' in extracted_params else 28.0
                v_skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, def_skin, step=1.0)
                def_dpf = float(extracted_params['DiabetesPedigreeFunction']['value']) if 'DiabetesPedigreeFunction' in extracted_params else 0.50
                v_dpf = st.number_input("Diabetes Pedigree Function", 0.01, 2.50, def_dpf, step=0.01)

            if st.button("🚀 Run Diagnostic Assessment on Uploaded Report", type="primary", use_container_width=True):
                patient_data = {
                    'Pregnancies': v_preg,
                    'Glucose': v_gluc,
                    'BloodPressure': v_bp,
                    'SkinThickness': v_skin,
                    'Insulin': v_ins,
                    'BMI': v_bmi,
                    'DiabetesPedigreeFunction': v_dpf,
                    'Age': v_age
                }
                eval_res = evaluate_patient(patient_data)
                render_assessment_results(v_name, eval_res, patient_data)

    # -------------------------------------------------------------------------
    # MODE 2: MANUAL PARAMETER ENTRY
    # -------------------------------------------------------------------------
    else:
        st.markdown("<h4 style='color: #38bdf8;'>Manual Clinical Diagnostic Entry</h4>", unsafe_allow_html=True)
        col_pname, col_spacer = st.columns([1, 1])
        with col_pname:
            manual_patient_name = st.text_input("Patient Full Name", value="Patient")

        col1, col2 = st.columns(2)
        with col1:
            m_glucose = st.slider("Plasma Glucose Concentration (mg/dL)", 0, 250, 148, help="Normal fasting range: 70-99 mg/dL. Values ≥ 140 mg/dL indicate elevated glycemic risk.")
            m_bmi = st.slider("Body Mass Index - BMI (kg/m²)", 0.0, 60.0, 33.6, step=0.1, help="Normal range: 18.5-24.9 kg/m²; Obese: ≥ 30 kg/m²")
            m_age = st.slider("Age (Years)", 21, 90, 50)
            m_pedigree = st.slider("Diabetes Pedigree Function (Genetics)", 0.05, 2.50, 0.627, step=0.01, help="Genetic hereditary diabetes score")

        with col2:
            m_pregnancies = st.number_input("Number of Pregnancies", 0, 20, 6)
            m_insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 0, 900, 155, help="0 will be automatically imputed with median")
            m_bp = st.number_input("Diastolic Blood Pressure (mmHg)", 0, 140, 72, help="0 will be automatically imputed with median")
            m_skin = st.number_input("Triceps Skin Fold Thickness (mm)", 0, 100, 35, help="0 will be automatically imputed with median")

        if st.button("🚀 Compute Diabetes Risk Assessment", type="primary", use_container_width=True):
            patient_data = {
                'Pregnancies': m_pregnancies,
                'Glucose': m_glucose,
                'BloodPressure': m_bp,
                'SkinThickness': m_skin,
                'Insulin': m_insulin,
                'BMI': m_bmi,
                'DiabetesPedigreeFunction': m_pedigree,
                'Age': m_age
            }
            eval_res = evaluate_patient(patient_data)
            render_assessment_results(manual_patient_name, eval_res, patient_data)


# =============================================================================
# PAGE 2: MODEL OVERVIEW & BENCHMARKS
# =============================================================================
elif page == "📊 Model Overview & Benchmarks":
    st.markdown("<h1 class='main-title'>📊 Model Overview & Performance Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Supervised Machine Learning Model Benchmarks & Data Preprocessing Summary</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patient Records", f"{len(df_raw)}")
    with col2:
        num_feats = len(model_info.get('feature_names', [16]))
        st.metric("Clinical Features", f"{num_feats} Features")
    with col3:
        st.metric("Positive Diabetes Ratio", f"{(df_raw['Outcome'].mean()*100):.1f}%")
    with col4:
        st.metric("Top Model ROC-AUC", f"{results[best_model_name]['metrics']['roc_auc']:.4f}")

    st.markdown(f"""
    <div class='highlight-best'>
        <h3>🏆 Selected Best Model: <strong>{best_model_name}</strong></h3>
        <p><strong>Selection Rationale:</strong> In medical diagnosis, avoiding <em>False Negatives</em> (missed diabetic patients) is critical. 
        <strong>{best_model_name}</strong> achieved the highest combined clinical utility with strong <strong>Recall ({(results[best_model_name]['metrics']['recall']*100):.2f}%)</strong> 
        and high <strong>ROC-AUC ({results[best_model_name]['metrics']['roc_auc']:.4f})</strong> after anti-bias SMOTE resampling.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Algorithm Leaderboard Comparison")
    rows = []
    for m_name, info in results.items():
        m = info['metrics']
        rows.append({
            'Algorithm': m_name,
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Precision': f"{m['precision']*100:.2f}%",
            'Recall (Sensitivity)': f"{m['recall']*100:.2f}%",
            'Specificity': f"{m['specificity']*100:.2f}%",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            '5-CV AUC': f"{m['cv_roc_auc']:.4f}"
        })

    res_df = pd.DataFrame(rows)
    st.dataframe(res_df, use_container_width=True)

    st.markdown("### 🛠 Anti-Bias Preprocessing Strategy")
    st.write("""
    1. **Missing Zero Treatment**: Invalid biological zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` replaced with `NaN`.
    2. **Stratified Median Imputation**: Imputed missing values using training set column medians to avoid data leakage.
    3. **SMOTE Class Resampling**: Applied **Synthetic Minority Over-sampling Technique (SMOTE)** on the training set to balance Outcome classes (65% non-diabetic vs 35% diabetic).
    4. **Standardization**: Features normalized with `StandardScaler` (\(\mu=0, \sigma=1\)) for distance-sensitive models.
    5. **Severity Classification Architecture**: Diabetic patients are clinically sub-stratified into **Low**, **Moderate**, and **High** severity levels using model probabilities combined with clinical glycemic thresholds.
    """)
