import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def train_and_evaluate_models():
    # ... (loading data code) ...

    """
    Loads preprocessed data, trains 4 different machine learning models,
    evaluates them using cross-validation, and saves the best model.
    """
    print("="*50)
    print("PHASE 3: MODEL TRAINING")
    print("="*50)

    # 1. Load Preprocessed Data
    print("1. Loading processed training and testing data...")
    try:
        X_train = pd.read_csv('processed_data/X_train.csv')
        X_test = pd.read_csv('processed_data/X_test.csv')
        
        # y arrays are loaded as 1D pandas Series using squeeze()
        y_train = pd.read_csv('processed_data/y_train.csv').squeeze()
        y_test = pd.read_csv('processed_data/y_test.csv').squeeze()
        print(f"   Successfully loaded data: {X_train.shape[0]} training rows, {X_test.shape[0]} test rows.")
    except Exception as e:
        print(f"ERROR: Could not load data. Details: {e}")
        print("TROUBLESHOOTING: Did you run 'python 03_data_preprocessing.py' first?")
        return

    # To handle class imbalance (if fewer readmissions vs non-readmissions), 
    # we compute the class weight ratio for XGBoost (num_negative_samples / num_positive_samples)
    pos_cases = sum(y_train == 1)
    neg_cases = sum(y_train == 0)
    scale_pos_weight = neg_cases / pos_cases if pos_cases > 0 else 1.0

    # 2. Initialize Models
    # We define all four models in a dictionary for easy iteration.
    print("2. Initializing Machine Learning Models...")
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', # Handles imbalanced classes automatically
            max_iter=1000,           # Allows more iterations to converge
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,        # Number of trees
            class_weight='balanced',
            random_state=42,
            n_jobs=-1                # Use all CPU cores for faster training
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("   [NOTE] XGBoost not found. Skipping it and using other models.")


    # Dictionary to store performance metrics of all models
    results = {}
    best_model_name = ""
    best_roc_auc = 0.0
    best_model_instance = None

    # 3. Training and Evaluating Models...
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # [SPEED UP]: Skipping 5-fold cross-validation by default to save time.
        # It trains the model 5 extra times per algorithm, which is slow for beginners.
        # print(f"   Performing 5-fold cross validation...")
        # cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        # print(f"   Mean CV ROC-AUC Score: {np.mean(cv_scores):.4f}")

        # Train the model on the entire training set once (much faster)
        model.fit(X_train, y_train)


        # Predict probabilities and classes on the separated Test Set
        # predict() gives 0 or 1. predict_proba() gives probability between [0.0, 1.0]
        y_pred = model.predict(X_test)
        
        # Getting probability of class 1 (readmitted) for AUC calculation
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC is often the best metric for imbalanced classification tasks
        roc_auc = roc_auc_score(y_test, y_prob)

        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }

        print(f"   Test Set ROC-AUC: {roc_auc:.4f}")

        # Track the best model based on ROC-AUC score
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model_instance = model

    # 4. Compare Models
    print("\n" + "-"*50)
    print("4. Model Comparison Summary")
    print("-" + "-"*50)
    
    # Convert results dictionary to a Pandas DataFrame for a nice table view
    results_df = pd.DataFrame(results).T
    print(results_df.sort_values(by='ROC-AUC', ascending=False))

    # 5. Save the best model
    print(f"\n5. Best Model Selected: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model object to a file so we can load it in the web app
    model_path = 'models/best_model.joblib'
    joblib.dump(best_model_instance, model_path)
    
    # Save the feature columns so the web app knows the exact column order to pass in
    joblib.dump(list(X_train.columns), 'models/feature_columns.joblib')
    
    print(f"   Successfully saved best model to '{model_path}'")
    print("\n" + "="*50)
    print("Training Complete! We now have a saved machine learning model ready.")
    print("Next step: Run 05_model_evaluation.py to visualize the best model's performance.")
    print("="*50)

if __name__ == "__main__":
    train_and_evaluate_models()
