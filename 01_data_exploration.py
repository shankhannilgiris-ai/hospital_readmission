import pandas as pd
import numpy as np
import os

def explore_data(filepath="hospital_readmissions.csv"):
    """
    Loads and performs initial exploration on the hospital readmission dataset.
    This script is designed for beginners to understand how to evaluate data quality.
    """
    print("="*50)
    print("PHASE 1: DATA EXPLORATION")
    print("="*50)
    
    # Check if file exists, provide troubleshooting message if not
    if not os.path.exists(filepath):
        print(f"ERROR: The file '{filepath}' was not found in the current directory.")
        print("TROUBLESHOOTING:")
        print("1. Please download the dataset from Kaggle: https://www.kaggle.com/datasets/dubradave/hospital-readmissions")
        print("2. Ensure it is extracted and named 'hospital_readmissions.csv'")
        print("3. Place it in the same directory as this script.")
        return None
        
    print(f"Loading dataset from {filepath}...\n")
    try:
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        print("ERROR: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"ERROR: Could not read the CSV file. Details: {e}")
        return None

    # Print basic information about the dataset
    print(f"Dataset Shape (Rows, Columns): {df.shape}")
    print(f"Total number of patient records: {df.shape[0]}")
    print(f"Total number of features/columns: {df.shape[1]}\n")

    # Show the first few rows to get a glimpse of the data
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\n" + "-"*50 + "\n")

    # Display data types of each column
    # This helps us identify categorical vs numerical features
    print("Data Types of Columns:")
    print(df.dtypes)
    print("\n" + "-"*50 + "\n")

    # Check for missing values in every column
    print("Missing Values Check:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("Great! There are no missing values in this dataset.")
    else:
        print("Found missing values:")
        # Only print columns that actually have missing values
        print(missing_values[missing_values > 0])
    print("\n" + "-"*50 + "\n")

    # Generate descriptive statistics for numerical columns
    print("Basic Statistical Summary (Numerical Features):")
    # describe() calculates count, mean, std, min, max, and quartiles
    print(df.describe().T) # .T transposes for better readability
    print("\n" + "-"*50 + "\n")
    
    # Generate descriptive statistics for categorical columns
    print("Basic Statistical Summary (Categorical Features):")
    print(df.describe(include=['object']).T)
    print("\n" + "="*50)
    print("Exploration Complete! Next, run 02_data_visualization.py")
    print("="*50)
    
    return df

if __name__ == "__main__":
    # If this script is run directly, execute the function
    df = explore_data()
