import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

# Import our custom prediction logic
# Note: Files starting with numbers cannot be imported using the standard 'import' statement.
try:
    import importlib.util
    import os
    # Import the module dynamically
    module_name = "06_prediction_function"
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), module_name + ".py"))
    if spec is not None:
        module_06 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_06)
        predict_single = module_06.predict_single_patient
    else:
        raise ImportError("Could not find 06_prediction_function.py")
except Exception as e:
     st.error(f"Failed to load prediction logic. Ensure '06_prediction_function.py' is in the same folder. Details: {e}")
     st.stop()






# Page Configuration
st.set_page_config(
    page_title="AI Hospital Readmission",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7d32, #1b5e20);
        color: white;
    }
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 800 !important;
    }
    .developer-card {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for a Premium, Theme-Aware Look
st.markdown("""
<style>
    /* Premium Animations & Styling */
    .stApp {
        background: transparent;
    }
    
    /* Responsive Text Colors */
    h1, h2, h3, .stMarkdown {
        color: var(--text-color);
    }
    
    .stButton>button {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white;
        border: none;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        color: white;
    }
    
    /* Card Styles that work in both modes */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .developer-card {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 25px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Section Dividers */
    .section-header {
        border-left: 5px solid #3b82f6;
        padding-left: 15px;
        margin: 25px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1>🏥 Hospital Readmission Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem;'>Scientific Prediction System for Patient Care Management</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color: rgba(128,128,128,0.2)'>", unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.markdown("""
    <div class='developer-card'>
        <p style='font-size: 0.9rem; opacity: 0.8; margin: 0;'>SYSTEM DEVELOPERS</p>
        <p style='margin:5px 0; font-weight:800; font-size:1.2rem; color: #60a5fa;'>Mohammed Ihsan I</p>
        <p style='margin:0; font-weight:800; font-size:1.2rem; color: #60a5fa;'>& Makavishnu S</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'><b>PROJECT INSIGHTS</b></div>", unsafe_allow_html=True)
    st.write("This neuro-clinical model evaluates patient stability post-discharge.")
    
    st.divider()
    st.markdown("<div class='section-header'><b>QUICK TESTING</b></div>", unsafe_allow_html=True)

    
    # Load High Risk Sample
    if st.button("🚨 Load High-Risk Sample", use_container_width=True):
        st.session_state.sample_type = 'high'
        st.session_state.sample_loaded = True
        st.rerun()
        
    # Load Low Risk Sample
    if st.button("✅ Load Low-Risk Sample", use_container_width=True):
        st.session_state.sample_type = 'low'
        st.session_state.sample_loaded = True
        st.rerun()

    if st.button("🔄 Reset Form", use_container_width=True):
        st.session_state.sample_loaded = False
        st.session_state.sample_type = None
        st.rerun()

# Logic to pick sample data
def get_val(key, default):
    if not st.session_state.get('sample_loaded', False):
        return default
    
    # High Risk Data values
    high_risk = {
        'age': '[70-80)', 'time': 9, 'specialty': 'InternalMedicine',
        'inpatient': 4, 'emergency': 2, 'lab': 82, 'meds': 35, 'diag1': 'Circulatory'
    }
    
    # Low Risk Data values
    low_risk = {
        'age': '[30-40)', 'time': 2, 'specialty': 'Family/GeneralPractice',
        'inpatient': 0, 'emergency': 0, 'lab': 25, 'meds': 12, 'diag1': 'Digestive'
    }
    
    sample = high_risk if st.session_state.get('sample_type') == 'high' else low_risk
    
    mapping = {
        'age': sample['age'], 'time': sample['time'], 'specialty': sample['specialty'],
        'inpatient': sample['inpatient'], 'emergency': sample['emergency'],
        'lab': sample['lab'], 'meds': sample['meds'], 'diag1': sample['diag1']
    }
    return mapping.get(key, default)

# Layout: Two columns for input fields
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("📋 Patient Information")
        age = st.selectbox("Age Bracket", 
                           ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                            '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                           index=['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'].index(get_val('age', '[50-60)')))
                           
        time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 
                                     value=get_val('time', 3))
                                     
        medical_specialty = st.selectbox("Admitting Specialty", 
                                         ['InternalMedicine', 'Cardiology', 'Surgery-General', 
                                          'Orthopedics', 'Family/GeneralPractice', 'Missing', 'Other'],
                                          index=['InternalMedicine', 'Cardiology', 'Surgery-General', 
                                                 'Orthopedics', 'Family/GeneralPractice', 'Missing', 'Other'].index(get_val('specialty', 'InternalMedicine')))
                                          
        st.subheader("🚪 Visitation History")
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            n_inpatient = st.number_input("Inpatient", 0, 20, value=get_val('inpatient', 0))
        with sub_col2:
            n_outpatient = st.number_input("Outpatient", 0, 20, value=0)
        with sub_col3:
            n_emergency = st.number_input("Emergency", 0, 20, value=get_val('emergency', 0))

with col2:
    with st.container(border=True):
        st.subheader("🔬 Clinical Details")
        n_procedures = st.number_input("Number of Procedures", 0, 10, value=1)
        n_lab_procedures = st.number_input("Number of Lab Procedures", 1, 150, value=get_val('lab', 45))
        n_medications = st.number_input("Number of Medications", 1, 100, value=get_val('meds', 15))
        
        st.subheader("🍬 Diabetes Management")
        diag_1 = st.text_input("Primary Diagnosis (e.g., Circulatory)", value=get_val('diag1', "Infectious"))
        
        c2a, c2b = st.columns(2)
        with c2a:
            glucose_test = st.selectbox("Glucose Test", ['not performed', 'normal', 'high'], index=0)
            diabetes_med = st.selectbox("Diabetes Meds?", ['no', 'yes'], index=1)
        with c2b:
            a1c_test = st.selectbox("A1C Test Result", ['not performed', 'normal', 'high'], index=0)
            change_med = st.selectbox("Change in Meds?", ['no', 'yes'], index=0)

# Center-aligned Predict Button
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 1.5, 1])
with btn_col:
    predict_clicked = st.button("🚀 RUN AI DIAGNOSIS", use_container_width=True, type="primary")

# Execute Prediction
if predict_clicked:
    input_data = {
        'age': age, 'time_in_hospital': time_in_hospital, 'n_procedures': n_procedures,
        'n_lab_procedures': n_lab_procedures, 'n_medications': n_medications,
        'n_outpatient': n_outpatient, 'n_inpatient': n_inpatient, 'n_emergency': n_emergency,
        'medical_specialty': medical_specialty, 'diag_1': diag_1 if diag_1 else 'Missing',
        'diag_2': 'Missing', 'diag_3': 'Missing', # Simplified for UI
        'glucose_test': glucose_test, 'A1Ctest': a1c_test, 'change': change_med,
        'diabetes_med': diabetes_med
    }

    with st.spinner("✨ Computing Risk Score..."):
        try:
            results = predict_single(input_data)
            
            if results:
                st.markdown("<div style='background-color: #f1f5f9; padding: 20px; border-radius: 15px; border-left: 5px solid #1e40af;'>", unsafe_allow_html=True)
                st.header("📊 Intelligence Report")
                
                res_col1, res_col2 = st.columns([1, 1.5])
                
                with res_col1:
                    risk_color = "#ef4444" if results['prediction'] == 'Yes' else "#22c55e"
                    risk_text = "🚨 CRITICAL RISK" if results['prediction'] == 'Yes' else "✅ STABLE / LOW RISK"
                    
                    st.markdown(f"<div style='background-color: white; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #64748b; margin-bottom: 0'>PATIENT STATUS</p><h2 style='color:{risk_color}; margin-top:0px;'>{risk_text}</h2>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric(label="Probability of Readmission", value=f"{results['probability']*100:.1f}%")

                with res_col2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = results['probability'] * 100,
                        gauge = {
                            'axis': {'range': [None, 100], 'tickcolor': "#1e3a8a"},
                            'bar': {'color': "#1e3a8a"},
                            'steps': [
                                {'range': [0, results['threshold_used']*100], 'color': "#dcfce7"},
                                {'range': [results['threshold_used']*100, 100], 'color': "#fee2e2"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'value': results['threshold_used']*100}
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#1e293b", 'family': "Inter"})
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### 📝 Clinical Action Plan")
                cols = st.columns(2)
                for i, rec in enumerate(results['recommendations']):
                    with cols[i % 2]:
                        if results['prediction'] == 'Yes':
                            st.error(f"⚠️ {rec}")
                        else:
                            st.success(f"📌 {rec}")
                            
        except Exception as e:
            st.error(f"Analysis failed: {e}")

