import pandas as pd
import numpy as np
import joblib
import os
import re

# Same age cleaning function from preprocessing
def clean_age_column(age_str):
    if pd.isna(age_str):
        return np.nan
    try:
        numbers = re.findall(r'\d+', str(age_str))
        if len(numbers) == 2:
            return (int(numbers[0]) + int(numbers[1])) / 2.0
        elif len(numbers) == 1:
            return float(numbers[0])
    except:
        pass
    return np.nan

def predict_single_patient(patient_data_dict):
    """
    Takes a dictionary representing a single patient's raw features,
    applies all saved preprocessing (encoders, scalers), 
    and predicts the readmission risk using the best model.
    """
    print("\nProcessing New Patient Data...")
    
    # 1. Load artifacts
    try:
        model = joblib.load('models/best_model.joblib')
        encoders = joblib.load('models/label_encoders.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_cols = joblib.load('models/feature_columns.joblib')
        
        # Load optimal threshold if it exists, otherwise default to 0.5
        if os.path.exists('models/optimal_threshold.joblib'):
            threshold = joblib.load('models/optimal_threshold.joblib')
        else:
            threshold = 0.5
            
    except Exception as e:
        print(f"ERROR: Could not load model artifacts. Details: {e}")
        return None

    # 2. Convert raw dictionary into a single-row DataFrame
    df = pd.DataFrame([patient_data_dict])
    
    # 3. Apply the exact same preprocessing logic from Phase 2
    # Convert age
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].apply(clean_age_column)
        df = df.drop('age', axis=1)

    # Encode categoricals using the saved LabelEncoders
    # We must handle unseen categories defensively (if a user enters a new diagnosis)
    for col, le in encoders.items():
        if col in df.columns:
            # Check if the category exists in the encoder's known classes
            known_classes = list(le.classes_)
            # Default to the most frequent class (often index 0 or 'Missing') if unknown
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
            # Transform
            df[col] = le.transform(df[col].astype(str))
            
    # Ensure correct column order as required by the model
    # Missing columns get filled with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_cols] # Reorder to match training data
    
    # Scale numericals
    # The scaler expects exactly the numeric columns it was trained on
    numeric_cols = [c for c in feature_cols if c not in encoders.keys()]
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # 4. Make prediction
    # Predict the probability of class 1 (Readmitted)
    prob_readmission = model.predict_proba(df)[0, 1]
    
    # Apply our custom optimized threshold instead of default 0.5
    prediction = 1 if prob_readmission >= threshold else 0

    # 5. Generate clinical recommendations
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    
    recommendations = []
    if prob_readmission > 0.8:
        recommendations.append("Require scheduling follow-up appointment within 48 hours of discharge.")
        recommendations.append("Assign care manager for weekly check-ins.")
    elif prob_readmission > threshold:
        recommendations.append("Schedule follow-up appointment within 7 days.")
        recommendations.append("Provide detailed discharge instructions and emergency contact info.")
    else:
        recommendations.append("Standard discharge protocol.")
        recommendations.append("Ensure patient understands medication schedule.")

    if patient_data_dict.get('n_medications', 0) > 15:
         recommendations.append("High medication count detected: Pharmacist review recommended.")

    return {
        'probability': prob_readmission,
        'prediction': 'Yes' if prediction == 1 else 'No',
        'risk_level': risk_level,
        'recommendations': recommendations,
        'threshold_used': threshold
    }

if __name__ == "__main__":
    print("="*50)
    print("PHASE 5: SINGLE PATIENT PREDICTION TEST")
    print("="*50)
    
    # Example Test Patient (simulate what someone might type in the Web App)
    test_patient = {
        'age': '[70-80)',
        'time_in_hospital': 8,
        'n_procedures': 2,
        'n_lab_procedures': 65,
        'n_medications': 22,
        'n_outpatient': 1,
        'n_inpatient': 2, # High inpatient past visits increases risk
        'n_emergency': 0,
        'medical_specialty': 'Cardiology',
        'diag_1': 'Heart Failure',
        'diag_2': 'Diabetes',
        'diag_3': 'Hypertension',
        'glucose_test': 'normal',
        'A1Ctest': 'high',
        'change': 'yes',
        'diabetes_med': 'yes'
    }

    result = predict_single_patient(test_patient)
    
    if result:
        print(f"\nResults for Patient:")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability of Readmission: {result['probability']*100:.1f}%")
        print(f"Model Prediction (Readmitted?): {result['prediction']}")
        
        print("\nClinical Recommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")

    print("\n" + "="*50)
    print("Prediction Function Works!")
    print("Next step: Run 'streamlit run app.py' to launch the web dashboard.")
    print("="*50)
