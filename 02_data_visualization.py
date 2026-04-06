import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_data(filepath="hospital_readmissions.csv"):
    """
    Creates informative visualizations for the dataset to understand
    feature relationships and patterns before building a model.
    """
    print("="*50)
    print("PHASE 1: DATA VISUALIZATION")
    print("="*50)

    if not os.path.exists(filepath):
        print(f"ERROR: Cannot find '{filepath}'. Please run 01_data_exploration.py instructions.")
        return

    print("Loading data...")
    df = pd.read_csv(filepath)

    # Create a directory to save our plots
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created '{output_dir}' directory to save plots.")

    # Set consistent visual style using Seaborn
    sns.set_theme(style="whitegrid", palette="muted")

    print("\nGenerating Plot 1: Distribution of Readmissions (Target Variable)...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='readmitted', palette='pastel', hue='readmitted', legend=False)
    plt.title('Target Variable: Hospital Readmission Status', fontsize=14)
    plt.ylabel('Number of Patients')
    plt.xlabel('Readmitted within 30 days?')
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11)
    plt.savefig(os.path.join(output_dir, '01_readmission_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 2: Age Distribution...")
    plt.figure(figsize=(10, 6))
    # Order age brackets chronologically if they are string brackets
    age_order = sorted(df['age'].unique())
    sns.countplot(data=df, x='age', order=age_order, palette='viridis', hue='age', legend=False)
    plt.title('Patient Age Distribution', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, '02_age_distribution.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 3: Length of Hospital Stay Patterns...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='time_in_hospital', bins=14, kde=True, color='teal')
    plt.title('Distribution of Time Spent in Hospital (Days)', fontsize=14)
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, '03_time_in_hospital.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 4: Readmission vs. Time in Hospital...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='readmitted', y='time_in_hospital', palette='Set2', hue='readmitted', legend=False)
    plt.title('Does longer stay lead to higher readmission?', fontsize=14)
    plt.savefig(os.path.join(output_dir, '04_stay_vs_readmission.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 5: Number of Medications vs. Age...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='age', y='n_medications', order=age_order, palette='magma', hue='age', legend=False)
    plt.title('Number of Medications Administered Across Age Groups', fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, '05_medications_vs_age.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 6: Readmission by Diabetes Medication Change...")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='change', hue='readmitted', palette='colorblind')
    plt.title('Readmission Counts based on Change in Diabetes Meds', fontsize=14)
    plt.xlabel('Change in Medication')
    plt.savefig(os.path.join(output_dir, '06_meds_change_readmission.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    print("Generating Plot 7: Medical Specialty Counts (Top 10)...")
    plt.figure(figsize=(12, 6))
    top_specialties = df['medical_specialty'].value_counts().nlargest(10).index
    sns.countplot(data=df, y='medical_specialty', order=top_specialties, palette='rocket', hue='medical_specialty', legend=False)
    plt.title('Top 10 Medical Specialties', fontsize=14)
    plt.xlabel('Number of Patients')
    plt.ylabel('Specialty')
    plt.savefig(os.path.join(output_dir, '07_top_specialties.png'), dpi=100, bbox_inches='tight')
    plt.close()

    print("Generating Plot 8: Correlation Heatmap of Numerical Features...")
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 8))
    # Compute correlation
    corr = numeric_df.corr()
    # Mask the upper triangle for cleaner visualization
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.savefig(os.path.join(output_dir, '08_correlation_heatmap.png'), dpi=100, bbox_inches='tight')

    plt.close()

    print("="*50)
    print(f"Visualization Complete! All 8 plots have been saved to the '{output_dir}' folder.")
    print("Review them to understand the data before moving to preprocessing.")
    print("Next step: Run 03_data_preprocessing.py")
    print("="*50)

if __name__ == "__main__":
    visualize_data()

