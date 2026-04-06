@echo off
color 0A
title Hospital Readmission ML Pipeline

echo =======================================================
echo    HOSPITAL READMISSION ML PIPELINE
echo =======================================================
echo.

echo [1/6] Checking/Installing dependencies...
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib plotly streamlit
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Some dependencies might have failed. Attempting to continue...
)

echo.
echo [2/6] Running Data Exploration...
python 01_data_exploration.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Data exploration failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/6] Running Data Visualization...
python 02_data_visualization.py

echo.
echo [4/6] Running Data Preprocessing...
python 03_data_preprocessing.py

echo.
echo [5/6] Running Model Training... (Optimized)
python 04_model_training.py

echo.
echo [6/6] Running Model Evaluation...
python 05_model_evaluation.py

echo.
echo =======================================================
echo    PIPELINE COMPLETE! Launching Web App...
echo =======================================================
echo If the app fails to open, check the errors below.
echo Press CTRL+C in this window to stop the server when done.
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Streamlit app failed to start.
    pause
)
