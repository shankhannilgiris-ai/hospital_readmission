import pandas as pd
import numpy as np
import os
import argparse

# Import predict logic from our 06 file
try:
    predict_single = __import__('06_prediction_function').predict_single_patient
except ImportError:
    print("ERROR: Could not import predict_single_patient from 06_prediction_function.py")
    print("Ensure you are running this from the same directory.")
    exit(1)

def batch_predict(input_csv, output_csv):
    """
    Reads a CSV of multiple patients, passes each row to the prediction engine,
    and saves the results to a new CSV.
    """
    print("="*50)
    print("PHASE 7: BATCH PREDICTION")
    print("="*50)

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file '{input_csv}' not found.")
        return

    print(f"Loading patients from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return

    # Check if we accidentally loaded the target variable (we shouldn't predict if we have the answer)
    if 'readmitted' in df.columns:
        df = df.drop('readmitted', axis=1)

    print(f"Processing {len(df)} patients...")
    
    # Store results dynamically
    probabilities = []
    predictions = []
    risk_levels = []

    # Iterate through row by row. 
    # For large datasets (millions of rows), doing this row-by-row with python dicts is slow,
    # and we would normally vectorize it in pandas. But for a beginner project, 
    # reusing the single-patient function is much easier to understand!
    total = len(df)
    
    for idx, row in df.iterrows():
        # Convert pandas Series row to basic Python dictionary
        patient_dict = row.to_dict()
        
        # Suppress prints from predict function by redirecting stdout or just pass 
        # Actually our predict_single prints things. To avoid spamming console:
        # We will just let it print for a small batch, but normally you'd disable prints.
        
        # Small hack to hide the print statements inside predict_single_patient
        import sys, os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            res = predict_single(patient_dict)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        if res is None:
            print("Failed to process row", idx)
            probabilities.append(np.nan)
            predictions.append("Unknown")
            risk_levels.append("Error")
            continue

        probabilities.append(res['probability'])
        predictions.append(res['prediction'])
        risk_levels.append(res['risk_level'])

        # Print progress every 10%
        if (idx+1) % max(1, (total//10)) == 0:
             print(f"   Progress: {idx+1}/{total} ({(idx+1)/total*100:.0f}%) complete...")

    # Attach results to the original DataFrame
    df['Predicted_Probability'] = probabilities
    df['Predicted_Readmitted'] = predictions
    df['Risk_Level'] = risk_levels

    # Save to disk
    df.to_csv(output_csv, index=False)
    
    # Summary Statistics
    high_risk_count = sum([1 for r in risk_levels if r == "High Risk"])
    print("\n" + "-"*50)
    print("BATCH PREDICTION SUMMARY:")
    print(f"Total Patients Analyzed: {total}")
    print(f"Identified as HIGH RISK: {high_risk_count} ({high_risk_count/total*100:.1f}%)")
    print(f"Identified as LOW RISK:  {total - high_risk_count} ({(total - high_risk_count)/total*100:.1f}%)")
    print("-" * 50)
    print(f"Successfully saved predictions to: {output_csv}")
    print("="*50)


if __name__ == "__main__":
    # Setup argument parser so user can run from command line like:
    # python batch_predict.py my_patients.csv results.csv
    parser = argparse.ArgumentParser(description="Run predictions on a batch of patients")
    parser.add_argument('input_file', nargs='?', default='hospital_readmissions.csv', help="Path to input CSV (default: hospital_readmissions.csv)")
    parser.add_argument('output_file', nargs='?', default='batch_predictions_output.csv', help="Path to output CSV (default: batch_predictions_output.csv)")
    
    args = parser.parse_args()
    
    # Warning if using the original dataset just testing
    if args.input_file == 'hospital_readmissions.csv':
         print("\nNOTE: Using the original full dataset for batch testing. This may take a moment...")
         # We will only process the first 50 rows by default so it doesn't take forever for the beginner
         print("Creating a 50-row sample for quick demonstration...")
         temp_df = pd.read_csv('hospital_readmissions.csv').head(50)
         temp_df.to_csv('temp_batch_input.csv', index=False)
         args.input_file = 'temp_batch_input.csv'
         
    batch_predict(args.input_file, args.output_file)
    
    # Cleanup temp file if we made it
    if os.path.exists('temp_batch_input.csv'):
        os.remove('temp_batch_input.csv')
