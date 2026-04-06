import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib # Used for saving our scalers and encoders for later use
import os
import re

def clean_age_column(age_str):
    """
    Helper function to convert age brackets (e.g., '[40-50)') into numerical midpoints.
    This provides our model with a continuous numeric value rather than a category,
    which often works better for algorithms measuring 'distance' or magnitude.
    """
    if pd.isna(age_str):
        return np.nan
        
    try:
        # Extract all numbers from the string (e.g., from '[40-50)' we get ['40', '50'])
        numbers = re.findall(r'\d+', str(age_str))
        if len(numbers) == 2:
            # Calculate midpoint
            return (int(numbers[0]) + int(numbers[1])) / 2.0
        elif len(numbers) == 1:
            return float(numbers[0])
    except:
        pass
    
    return np.nan # Fallback if parsing fails

def preprocess_data(filepath="hospital_readmissions.csv"):
    """
    Reads raw data, handles missing values, encodes categories, scales numbers,
    and splits into training/testing sets.
    """
    print("="*50)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*50)

    if not os.path.exists(filepath):
        print(f"ERROR: Cannot find '{filepath}'. Please ensure it is in the current directory.")
        return

    print("1. Loading raw data...")
    df = pd.read_csv(filepath)
    
    # Create models directory to save preprocessing artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)

    print("2. Converting 'age' brackets to numeric midpoints...")
    # Example: '[70-80)' becomes 75.0
    df['age_numeric'] = df['age'].apply(clean_age_column)
    
    # Drop the original string age column as we have the numeric one now
    df = df.drop('age', axis=1)

    print("3. Handling Missing Values...")
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Fill numerical missing values with the median (robust to outliers)
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   - Filled missing {col} with median: {median_val}")

    # Fill categorical missing values with the most frequent value (mode)
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   - Filled missing {col} with mode: {mode_val}")

    print("4. Encoding Categorical Variables with LabelEncoder...")
    # Machine learning models require numbers, not strings.
    # We use LabelEncoder to turn 'Category A', 'Category B' into 0, 1.
    label_encoders = {}
    
    # Isolate target variable so we map it cleanly manually for clarity
    if 'readmitted' in categorical_cols:
        categorical_cols.remove('readmitted')
        
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit and transform the data
        df[col] = le.fit_transform(df[col].astype(str))
        # Save the encoder so we can use it during prediction for new patients
        label_encoders[col] = le
    
    # Map the target variable uniquely (Yes=1, No=0)
    df['readmitted'] = df['readmitted'].map({'no': 0, 'yes': 1, 'No': 0, 'Yes': 1})
    # If the dataset has different values for readmitted, fallback to label encoder
    if df['readmitted'].isnull().any():
         print("   - Using generic LabelEncoder for target variable as map failed.")
         le_target = LabelEncoder()
         df['readmitted'] = le_target.fit_transform(df['readmitted_temp'].astype(str))
         label_encoders['readmitted'] = le_target

    # Save the dictionary of encoders to a file
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    print("   - Saved label encoders to 'models/label_encoders.joblib'")

    print("5. Scaling Numerical Features...")
    # Features like 'n_medications' might be much larger than 'time_in_hospital'. 
    # StandardScaler scales them so they have a mean of 0 and std deviation of 1,
    # preventing features with large numbers from dominating the model.
    features_to_scale = [col for col in numerical_cols if col != 'readmitted']
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("   - Saved scaler to 'models/scaler.joblib'")

    print("6. Splitting Data into Training (80%) and Testing (20%) Sets...")
    # X contains all features (columns) EXCEPT the target 'readmitted'
    X = df.drop('readmitted', axis=1)
    # y contains ONLY the target 'readmitted'
    y = df['readmitted']

    # random_state ensures reproducibility (we get the same split every time we run it)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Note: 'stratify=y' ensures the train and test sets have the same proportion of readmitted vs non-readmitted patients.

    print(f"   - Training Features Shape: {X_train.shape}")
    print(f"   - Testing Features Shape: {X_test.shape}")

    print("7. Saving preprocessed data datasets to disk...")
    # This saves time later, we can load these directly in the model training script
    X_train.to_csv('processed_data/X_train.csv', index=False)
    X_test.to_csv('processed_data/X_test.csv', index=False)
    y_train.to_csv('processed_data/y_train.csv', index=False)
    y_test.to_csv('processed_data/y_test.csv', index=False)

    print("="*50)
    print("Preprocessing Complete!")
    print("Models directory created with 'scaler.joblib' and 'label_encoders.joblib'.")
    print("Processed datasets saved to 'processed_data/' folder.")
    print("Next step: Run 04_model_training.py")
    print("="*50)

if __name__ == "__main__":
    preprocess_data()
