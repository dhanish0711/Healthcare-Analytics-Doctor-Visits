"""
Healthcare Analytics for Doctor Visits - Streamlit Web Application
Author: Dhanish Ladwani
Course: VOIS AICTE Batch 1 Data Analytics (DIY Project)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Healthcare Analytics | Doctor Visits",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .author-badge {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to locate files safely
def get_path(rel_path):
    # Try current directory first, then look inside 'diy project'
    if os.path.exists(rel_path):
        return rel_path
    alt_path = os.path.join("diy project", rel_path)
    if os.path.exists(alt_path):
        return alt_path
    base_dir = r"F:\Edunet Data Analytics\diy project"
    abs_path = os.path.join(base_dir, rel_path)
    if os.path.exists(abs_path):
        return abs_path
    return rel_path

@st.cache_data
def load_data():
    csv_path = get_path(os.path.join("data_cleaned", "healthcare_doctor_visits_cleaned.csv"))
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def load_bundle():
    model_path = get_path(os.path.join("models", "model_bundle.joblib"))
    bundle = joblib.load(model_path)
    return bundle

try:
    df = load_data()
    bundle = load_bundle()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    data_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)
    st.markdown("## 🏥 Navigation")
    tab_selection = st.radio(
        "Choose View:",
        [
            "📊 Executive Dashboard",
            "🤖 Patient Consultation Predictor",
            "🏥 Clinic Capacity & Triage Simulator",
            "🔬 Statistical Evidence & ML Insights"
        ]
    )
    st.markdown("---")
    st.markdown("""
    **Project:** Healthcare Analytics (DIY)  
    **Author:** Dhanish Ladwani  
    **Program:** VOIS AICTE Batch 1 (2026-2027)  
    **Dataset:** Australian Health Survey (5,190 records)  
    """)
    st.markdown("---")
    st.info("💡 **Zero-Inflation Notice:** 79.8% of patients record 0 visits in a 2-week period. Advanced weighting and hurdle models are used to handle class imbalance.")

# Main Header
st.markdown('<div class="main-header">Healthcare Analytics for Doctor Visits</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predictive Modeling, Clinical Risk Stratification & Healthcare Resource Allocation</div>', unsafe_allow_html=True)
st.markdown('<span class="author-badge">Student Author: Dhanish Ladwani | AICTE Major Project</span>', unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# -------------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if tab_selection == "📊 Executive Dashboard":
    st.markdown("### 📈 Population Health & Consultation Overview")
    
    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Patients", f"{len(df):,}")
    with kpi2:
        zero_pct = (df['visits'] == 0).mean() * 100
        st.metric("Zero Visits (2-Wk)", f"{zero_pct:.1f}%")
    with kpi3:
        st.metric("Mean Fortnight Visits", f"{df['visits'].mean():.3f}")
    with kpi4:
        female_rate = (df[df['gender'] == 'female']['visits'] > 0).mean() * 100
        st.metric("Female Visit Likelihood", f"{female_rate:.1f}%", "+8.8% vs Male")
    with kpi5:
        repat_mean = df[df['insurance_category'] == 'Repatriation / Veteran']['visits'].mean()
        st.metric("Veteran / Repat Visits", f"{repat_mean:.3f}", "+55% vs Avg")
        
    st.markdown("---")
    
    # Filters
    st.markdown("#### 🔍 Dynamic Cohort Filtering")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_genders = st.multiselect("Gender", options=df['gender'].unique(), default=list(df['gender'].unique()))
    with f_col2:
        selected_cohorts = st.multiselect("Age Cohort", options=df['age_group'].unique(), default=list(df['age_group'].unique()))
    with f_col3:
        selected_insurance = st.multiselect("Insurance Coverage", options=df['insurance_category'].unique(), default=list(df['insurance_category'].unique()))
        
    filtered_df = df[
        (df['gender'].isin(selected_genders)) &
        (df['age_group'].isin(selected_cohorts)) &
        (df['insurance_category'].isin(selected_insurance))
    ]
    
    st.caption(f"Showing **{len(filtered_df):,}** patients matching the selected criteria.")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### 📌 Doctor Visit Distribution")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.countplot(x='visits', hue='visits', data=filtered_df, ax=ax1, palette='Blues_r', edgecolor='black', legend=False)
        ax1.set_title("Consultation Frequency (Past 2 Weeks)", fontweight='bold')
        ax1.set_xlabel("Number of Visits")
        ax1.set_ylabel("Patient Count")
        st.pyplot(fig1)
        
    with chart_col2:
        st.markdown("##### 📌 Consultation Likelihood by Age Cohort & Gender")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rate_df = filtered_df.groupby(['age_group', 'gender'], observed=False)['has_visited'].mean().reset_index()
        rate_df['pct'] = rate_df['has_visited'] * 100
        sns.barplot(x='age_group', y='pct', hue='gender', data=rate_df, ax=ax2, palette=['#e74c3c', '#2980b9'], edgecolor='black')
        ax2.set_title("Doctor Consultation Rate (%)", fontweight='bold')
        ax2.set_xlabel("Age Cohort")
        ax2.set_ylabel("% Patients Seeking Care")
        ax2.set_ylim(0, 50)
        st.pyplot(fig2)
        
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown("##### 📌 Health Impairment vs. Doctor Visits")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='reduced', y='visits', data=filtered_df, ax=ax3, color='#e67e22', marker='s', linewidth=2.5)
        ax3.set_title("Mean Visits vs. Reduced Activity Days", fontweight='bold')
        ax3.set_xlabel("Days with Reduced Normal Activity")
        ax3.set_ylabel("Average Visits")
        st.pyplot(fig3)
        
    with chart_col4:
        st.markdown("##### 📌 Chronic Burden vs. Mean Visits")
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.barplot(x='chronic_status', y='visits', hue='chronic_status', data=filtered_df, ax=ax4, palette='Reds', edgecolor='black', legend=False)
        ax4.set_title("Consultations by Chronic Condition Type", fontweight='bold')
        ax4.set_xlabel("Chronic Status")
        ax4.set_ylabel("Mean Visits")
        ax4.tick_params(axis='x', rotation=15)
        st.pyplot(fig4)

# -------------------------------------------------------------
# TAB 2: PATIENT CONSULTATION PREDICTOR
# -------------------------------------------------------------
elif tab_selection == "🤖 Patient Consultation Predictor":
    st.markdown("### 🔮 Real-Time Patient Consultation & Triage Predictor")
    st.write("Input patient demographic, functional impairment, and healthcare coverage parameters to estimate consultation probability and expected visit count.")
    
    with st.form("patient_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            gender_in = st.selectbox("Biological Gender", ["female", "male"])
            age_in = st.slider("Patient Age (Years)", min_value=18, max_value=85, value=45)
            income_in = st.slider("Annual Income ($10k Scale)", min_value=0.0, max_value=1.5, value=0.55, step=0.05)
            
        with col_p2:
            illness_in = st.slider("Acute Illness Count (Past 2 Wks)", min_value=0, max_value=5, value=1)
            reduced_in = st.slider("Days of Reduced Normal Activity (0-14)", min_value=0, max_value=14, value=2)
            health_in = st.slider("General Health Score (GHQ: 0 = Good, 12 = Poor)", min_value=0, max_value=12, value=1)
            
        with col_p3:
            nchronic_in = st.selectbox("Non-Limiting Chronic Illness?", ["no", "yes"])
            lchronic_in = st.selectbox("Activity-Limiting Chronic Illness?", ["no", "yes"])
            insurance_in = st.selectbox("Healthcare Support Scheme", ["Public / Standard", "Private Insurance", "Low-Income Assistance", "Repatriation / Veteran"])
            
        submit_btn = st.form_submit_button("⚡ Run Clinical Assessment & Predict")
        
    if submit_btn:
        # Prepare feature vector
        gender_male = 1 if gender_in == 'male' else 0
        nchronic_bin = 1 if nchronic_in == 'yes' else 0
        lchronic_bin = 1 if lchronic_in == 'yes' else 0
        total_chronic = nchronic_bin + lchronic_bin
        
        private_bin = 1 if insurance_in == "Private Insurance" else 0
        freepoor_bin = 1 if insurance_in == "Low-Income Assistance" else 0
        freerepat_bin = 1 if insurance_in == "Repatriation / Veteran" else 0
        
        reduced_norm = reduced_in / 14.0
        illness_norm = illness_in / 5.0
        health_norm = health_in / 12.0
        chronic_penalty = 1.0 if lchronic_bin == 1 else (0.5 if nchronic_bin == 1 else 0.0)
        
        risk_score = round((reduced_norm * 40.0) + (illness_norm * 30.0) + (health_norm * 20.0) + (chronic_penalty * 10.0), 2)
        
        feat_dict = {
            'gender_male': [gender_male],
            'age_years': [age_in],
            'income': [income_in],
            'illness': [illness_in],
            'reduced': [reduced_in],
            'health': [health_in],
            'health_risk_score': [risk_score],
            'nchronic_bin': [nchronic_bin],
            'lchronic_bin': [lchronic_bin],
            'total_chronic_count': [total_chronic],
            'private_bin': [private_bin],
            'freepoor_bin': [freepoor_bin],
            'freerepat_bin': [freerepat_bin]
        }
        X_input = pd.DataFrame(feat_dict)[bundle['feature_cols']]
        
        # Predictions
        clf = bundle['best_classifier']
        reg = bundle['best_regressor']
        
        prob_visit = clf.predict_proba(X_input)[0, 1]
        pred_visit_cls = clf.predict(X_input)[0]
        
        if bundle['best_regressor_name'] in ['Ridge Regression', 'Poisson Regressor']:
            X_input_sc = bundle['scaler'].transform(X_input)
            pred_visits_num = float(np.clip(reg.predict(X_input_sc)[0], 0, None))
        else:
            pred_visits_num = float(np.clip(reg.predict(X_input)[0], 0, None))
            
        # Cluster assessment
        X_clust_input = pd.DataFrame({
            'age_years': [age_in],
            'illness': [illness_in],
            'reduced': [reduced_in],
            'health': [health_in],
            'total_chronic_count': [total_chronic],
            'health_risk_score': [risk_score]
        })[bundle['cluster_features']]
        
        X_clust_sc = bundle['scaler_cluster'].transform(X_clust_input)
        cluster_id = bundle['kmeans_model'].predict(X_clust_sc)[0]
        cluster_label = bundle['cluster_names'].get(cluster_id, "Standard Patient")
        
        # Display Results
        st.markdown("#### 📋 Assessment & Triage Output")
        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"**Likelihood of Seeking Consultation:**")
            st.progress(float(prob_visit))
            st.markdown(f"### {prob_visit*100:.1f}%")
            if prob_visit >= 0.50:
                st.error("🚨 High Probability of Doctor Visit")
            else:
                st.success("✅ Low Probability of Immediate Visit")
                
        with res2:
            st.markdown(f"**Forecasted Consultation Frequency:**")
            st.markdown(f"### {pred_visits_num:.2f} visits")
            st.caption("Expected medical consultations over 14-day window")
            
        with res3:
            st.markdown(f"**Patient Risk Stratum:**")
            st.markdown(f"### {cluster_label}")
            st.metric("Health Impairment Index", f"{risk_score} / 100")
            
        st.markdown("---")
        st.markdown("##### 🩺 Clinical Care Recommendations")
        if risk_score > 40 or lchronic_bin == 1 or reduced_in >= 7:
            st.warning("""
            - **Priority Triage:** Patient exhibits elevated functional impairment and/or activity-limiting chronic conditions.
            - **Action:** Recommend direct primary care physician scheduling or home-health checkup within 48 hours.
            - **Chronic Care Management:** Enroll in coordinated care program to reduce acute escalation.
            """)
        elif illness_in >= 2 or prob_visit >= 0.40:
            st.info("""
            - **Moderate Need:** Patient experiencing acute symptom burden with low chronic limitation.
            - **Action:** Telemedicine consultation or outpatient clinic walk-in recommended.
            - **Symptom Review:** Monitor activity restriction over the next 7 days.
            """)
        else:
            st.success("""
            - **Routine Care:** Patient maintains low impairment and baseline stability.
            - **Action:** Annual wellness checkup and preventative screening sufficient.
            """)

# -------------------------------------------------------------
# TAB 3: CLINIC CAPACITY & TRIAGE SIMULATOR
# -------------------------------------------------------------
elif tab_selection == "🏥 Clinic Capacity & Triage Simulator":
    st.markdown("### 🏥 Primary Care Clinic Inflow & Capacity Simulator")
    st.write("Simulate patient intake scenarios to project daily consultation volume, physician hours, and nurse triage load.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        intake_volume = st.slider("Total Enrolled / Emplaced Patient Population", min_value=500, max_value=20000, value=5000, step=500)
        pct_elderly = st.slider("% Seniors (65+)", min_value=5, max_value=60, value=25)
        pct_chronic = st.slider("% Patients with Limiting Chronic Conditions", min_value=2, max_value=40, value=12)
        
    with col_s2:
        pct_veteran = st.slider("% Repatriation / Veteran Coverage", min_value=0, max_value=40, value=20)
        avg_consult_mins = st.slider("Average Consultation Duration (Minutes)", min_value=10, max_value=45, value=20)
        physician_daily_hours = st.slider("Physician Daily Direct Patient Hours", min_value=4, max_value=10, value=6)
        
    # Simulation Math
    base_visit_rate = 0.257
    elderly_factor = 1.0 + (pct_elderly / 100.0) * 0.45
    chronic_factor = 1.0 + (pct_chronic / 100.0) * 1.30
    veteran_factor = 1.0 + (pct_veteran / 100.0) * 0.55
    
    simulated_fortnight_visits = intake_volume * base_visit_rate * (elderly_factor * chronic_factor * veteran_factor) / 3.0
    daily_consultations = simulated_fortnight_visits / 10.0  # 10 clinical days in a fortnight
    total_consult_hours = (daily_consultations * avg_consult_mins) / 60.0
    required_doctors = int(np.ceil(total_consult_hours / physician_daily_hours))
    nurse_triage_count = int(np.ceil(daily_consultations / 40.0))
    
    st.markdown("---")
    st.markdown("#### 📊 Simulation Results & Staffing Recommendations")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Projected Daily Consults", f"{int(daily_consultations):,}")
    with m_col2:
        st.metric("Daily Clinical Hours", f"{total_consult_hours:.1f} hrs")
    with m_col3:
        st.metric("Physicians Needed", f"{required_doctors} MDs")
    with m_col4:
        st.metric("Triage Nurses Needed", f"{nurse_triage_count} RNs")
        
    st.info(f"💡 **Operational Guidance:** For a panel of {intake_volume:,} patients with {pct_chronic}% chronic disease burden, the clinic expects approximately **{int(daily_consultations)}** doctor consultations every weekday. We recommend staffing at least **{required_doctors} full-time doctors** to prevent patient wait-times from exceeding 30 minutes.")

# -------------------------------------------------------------
# TAB 4: STATISTICAL EVIDENCE & ML INSIGHTS
# -------------------------------------------------------------
elif tab_selection == "🔬 Statistical Evidence & ML Insights":
    st.markdown("### 🔬 Rigorous Statistical Tests & Machine Learning Metrics")
    
    st.markdown("#### 1. Inferential Statistical Hypothesis Tests")
    stats_data = [
        {
            "Hypothesis": "Gender Disparity in Consultations",
            "Tests Conducted": "Welch's t-test (t = 5.72), Mann-Whitney U (p = 6.34e-15), Chi-Square (chi2 = 61.53)",
            "p-value": "< 0.0001",
            "Conclusion": "Reject H0. Females visit doctors significantly more frequently than males."
        },
        {
            "Hypothesis": "Healthcare Support Program Effect",
            "Tests Conducted": "One-Way ANOVA (F = 24.02), Kruskal-Wallis (H = 138.06)",
            "p-value": "< 0.0001",
            "Conclusion": "Reject H0. Veteran/Repat and Low-Income recipients show distinct utilization patterns."
        },
        {
            "Hypothesis": "Activity-Limiting Chronic Illness Impact",
            "Tests Conducted": "Chi-Square (chi2 = 163.12), Mann-Whitney U (p = 1.20e-23)",
            "p-value": "< 0.0001",
            "Conclusion": "Reject H0. Activity-limiting chronic conditions increase visits by > 2.3x."
        }
    ]
    st.table(pd.DataFrame(stats_data))
    
    st.markdown("---")
    st.markdown("#### 2. Machine Learning Benchmark Comparison")
    
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown("##### 🎯 Classification Models (Predicting Consultation Likelihood)")
        clf_table = pd.DataFrame([
            {"Model": "Logistic Regression (Balanced)", "Accuracy": "71.6%", "Recall": "59.5%", "ROC-AUC": "0.736"},
            {"Model": "Random Forest Classifier (Balanced)", "Accuracy": "71.8%", "Recall": "60.0%", "ROC-AUC": "0.737"},
            {"Model": "Gradient Boosting", "Accuracy": "80.5%", "Recall": "23.8%", "ROC-AUC": "0.732"}
        ])
        st.table(clf_table)
        st.caption("*Random Forest is selected as the production classifier due to its superior ROC-AUC and high sensitivity on the minority positive consultation class.*")
        
    with ml_col2:
        st.markdown("##### 📈 Count Regression Models (Predicting Number of Visits)")
        reg_table = pd.DataFrame([
            {"Model": "Ridge Regression", "MAE": "0.393", "RMSE": "0.672", "R2": "0.175"},
            {"Model": "Poisson Regressor", "MAE": "0.412", "RMSE": "0.685", "R2": "0.143"},
            {"Model": "Random Forest Regressor", "MAE": "0.385", "RMSE": "0.695", "R2": "0.117"}
        ])
        st.table(reg_table)
        st.caption("*Random Forest Regressor minimizes Mean Absolute Error (MAE = 0.385) across the zero-inflated target.*")
        
    st.markdown("---")
    st.markdown("#### 3. Random Forest Feature Importance Ranking")
    if 'clf_feature_importances' in bundle:
        imp_series = pd.Series(bundle['clf_feature_importances']).sort_values(ascending=True)
        fig_imp, ax_imp = plt.subplots(figsize=(9, 5))
        imp_series.plot(kind='barh', ax=ax_imp, color='#2563eb', edgecolor='black')
        ax_imp.set_title("Gini Feature Importance for Doctor Visit Prediction", fontweight='bold')
        ax_imp.set_xlabel("Importance Weight")
        st.pyplot(fig_imp)
