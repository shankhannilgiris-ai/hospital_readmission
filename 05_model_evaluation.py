import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score, classification_report)
import joblib
import os

def evaluate_best_model():
    """
    Loads our best saved model and creates visualizations of its performance,
    computes feature importance, and finds the optimal probability threshold.
    """
    print("="*50)
    print("PHASE 4: IN-DEPTH MODEL EVALUATION")
    print("="*50)

    # 1. Load the data and the best model
    try:
        X_test = pd.read_csv('processed_data/X_test.csv')
        y_test = pd.read_csv('processed_data/y_test.csv').squeeze()
        best_model = joblib.load('models/best_model.joblib')
        features = joblib.load('models/feature_columns.joblib')
    except Exception as e:
        print(f"ERROR: Missing files. Details: {e}")
        print("TROUBLESHOOTING: Ensure you ran 04_model_training.py and files exist in 'models/' and 'processed_data/'.")
        return

    # Ensure output directory exists based on user requirement
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Successfully loaded '{type(best_model).__name__}' model.")
    
    # Model predictions on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] # Probability of 'readmitted' (Class 1)

    print("\n1. Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Readmitted', 'Readmitted'], 
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.ylabel('Actual Truth')
    plt.xlabel('Predicted by Model')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, '09_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("2. Generating ROC Curve...")
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, '10_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("3. Generating Precision-Recall Curve...")
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (True Positive Rate)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, '11_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("4. Calculating Feature Importance...")
    # Does the model support feature importances tree-based?
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        # Create a dataframe to sort them
        feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15) # Top 15
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', hue='Feature', legend=False)
        plt.title('Top 15 Most Important Features for Prediction')
        plt.savefig(os.path.join(output_dir, '12_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   - Feature importance bar chart saved.")
    else:
        # If Logistic Regression, use coefficients instead
        if hasattr(best_model, 'coef_'):
            importances = best_model.coef_[0]
            # Get absolute values of coefficients to measure magnitude of impact
            feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': np.abs(importances), 'Direction': np.sign(importances)})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Direction', palette='coolwarm')
            plt.title('Top 15 Most Important Features (Coefficient Magnitudes)')
            plt.savefig(os.path.join(output_dir, '12_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("   - Feature coefficient bar chart saved.")
        else:
            print("   - Chosen model does not natively support feature importances logging.")

    print("\n5. Threshold Optimization...")
    # By default ML models predict Class 1 if Probability > 0.5.
    # However for medical tasks like Readmission, we might want to catch more cases 
    # even if False Positives increase. We find the optimal threshold to balance Precision and Recall.
    
    # A simple way to find optimal threshold using F-score (harmonic mean of precision/recall)
    # Exclude the last threshold which is 1
    fscore = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10) # 1e-10 avoids div by zero
    
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    optimal_threshold = pr_thresholds[ix]
    
    # [USER REQUEST] Increase threshold to be more conservative
    # We add 0.1 to the mathematically "optimal" threshold to focus on high-certainty cases
    conservative_threshold = min(0.9, optimal_threshold + 0.1)
    
    print(f"   Default Threshold (0.5) Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted']))
    
    # Custom predictions using conservative threshold
    custom_y_pred = (y_prob >= conservative_threshold).astype(int)
    print(f"\n   Conservative Threshold ({conservative_threshold:.3f}) Classification Report:")
    print(classification_report(y_test, custom_y_pred, target_names=['Not Readmitted', 'Readmitted']))
    
    # Save the conservative threshold so our App can use it
    joblib.dump(conservative_threshold, 'models/optimal_threshold.joblib')
    print(f"   Saved conservative decision threshold ({conservative_threshold:.3f}) to 'models/optimal_threshold.joblib'")


    print("\n" + "="*50)
    print("Model Evaluation Complete!")
    print("Plots saved in the 'visualizations' folder.")
    print("Next step: Run 06_prediction_function.py to see how to predict a new single patient.")
    print("="*50)

if __name__ == "__main__":
    evaluate_best_model()
