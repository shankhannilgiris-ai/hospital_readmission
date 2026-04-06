<<<<<<< HEAD
# hospital_readmission
vanakam gaaa
=======
# Hospital Readmission Prediction System 🏥

Welcome to your first machine learning project! This repository contains a complete, end-to-end Machine Learning pipeline that predicts whether a patient will be readmitted to a hospital within 30 days of discharge. 

## Project Architecture

This project is divided into distinct phases, each with its own Python script so you can learn step-by-step:

1. **`01_data_exploration.py`**: Loads the raw dataset, checks for missing values, and prints basic statistics.
2. **`02_data_visualization.py`**: Generates 8 beautiful charts (saved to the `visualizations/` folder) to help you understand trends in the data.
3. **`03_data_preprocessing.py`**: Cleans the data, fills missing values, converts text to numbers (LabelEncoding), scales numerical values, and splits data into training/testing sets.
4. **`04_model_training.py`**: Trains 4 different algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost), compares them, and saves the best one.
5. **`05_model_evaluation.py`**: Tests the best model, plots ROC and Precision-Recall curves, calculates feature importance, and finds the optimal decision threshold.
6. **`06_prediction_function.py`**: Contains a reusable function that takes raw patient details and converts them into an AI prediction.
7. **`app.py`**: A stunning, interactive web application built with Streamlit that lets you use the AI model in your browser!
8. **`batch_predict.py`**: A utility script to predict readmission for hundreds of patients at once from a CSV file.

---

## 🚀 Setup Instructions

### Prerequisites
- You need Python 3.9 installed on your computer.

### Step 1: Download the Data
1. Download the dataset from Kaggle: [Hospital Readmissions Dataset](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)
2. Extract the file and name it exactly `hospital_readmissions.csv`
3. Place `hospital_readmissions.csv` inside this project folder.

### Step 2: Install Required Libraries
Open your terminal (Command Prompt or PowerShell) in this folder and run:
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run the Project

### The Easy Way (Windows ONLY)
If you are on Windows, simply double-click the `RUN_EVERYTHING.bat` file. 
It will automatically run every script in the correct order, install dependencies, and launch the web app!

### The Step-By-Step Way (Recommended for Learning)
Run these commands one by one in your terminal to see how the system is built:

1. **Explore the data:**
   ```bash
   python 01_data_exploration.py
   ```
2. **Generate graphs (Look inside the `visualizations/` folder after):**
   ```bash
   python 02_data_visualization.py
   ```
3. **Clean the data:**
   ```bash
   python 03_data_preprocessing.py
   ```
4. **Train the AI models (This might take a minute):**
   ```bash
   python 04_model_training.py
   ```
5. **Evaluate the AI:**
   ```bash
   python 05_model_evaluation.py
   ```
6. **Launch the interactive Web App! :**
   ```bash
   streamlit run app.py
   ```

*(Optional) Test batch prediction:*
```bash
python batch_predict.py
```

---

## 🛠 Troubleshooting

**Error: `FileNotFoundError: hospital_readmissions.csv`**
* **Solution**: You forgot to download the data or didn't put it in the same folder as the scripts. See Step 1 above.

**Error: `ModuleNotFoundError: No module named 'pandas'`**
* **Solution**: You haven't installed the requirements. Run `pip install -r requirements.txt`.

**Web App isn't opening / Error about missing models**
* **Solution**: You must run the data preprocessing and model training scripts *before* you can run the web app. The web app needs the saved AI model to function!

---

## 🌟 Suggestions for Future Improvements
Once you understand this code, try these challenges to improve your skills:
1. **Feature Engineering**: Try combining columns (like `n_inpatient` + `n_outpatient`) to create a new `total_prior_visits` column before training the model.
2. **Hyperparameter Tuning**: In `04_model_training.py`, try changing the `n_estimators` or `max_depth` in the RandomForest to see if you can get a higher ROC-AUC score!
3. **Add user Authentication**: Update the Streamlit app to require a doctor's login.
>>>>>>> 8482101 (initialcommit)
