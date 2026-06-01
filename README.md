# Predictive Maintenance Dashboard

An ML-powered dashboard that predicts machine failures before they happen, built with Python, Scikit-learn, and Streamlit.

## Problem Statement
Unexpected machine failures cost factories thousands of dollars per hour in unplanned downtime. This project builds a predictive model that identifies at-risk machines before they fail.

## Dataset
AI4I 2020 Predictive Maintenance Dataset
- 10,000 machine sensor readings
- 6 features: Temperature, Speed, Torque, Tool Wear, Machine Type
- Failure rate: 3.39%

## Key Findings
- Torque (34.5%) and Rotational Speed (26.9%) are the strongest predictors of machine failure
- Optimized threshold at 40% gives best balance (68% recall, 85% precision)
- Tested 3 improvement methods: Threshold Adjustment, SMOTE, Class Weighting
- Best model catches 68% of failures before they happen

## Dashboard Features
- Real-time Machine Risk Predictor with adjustable sensor sliders
- Feature importance visualization
- Model performance comparison across all 3 methods
- Interactive data explorer with filters

## Tech Stack
- Python
- Scikit-learn — Random Forest Classifier
- Pandas — data manipulation
- Plotly — interactive charts
- Streamlit — web dashboard

## How to Run
pip install pandas numpy scikit-learn streamlit plotly imbalanced-learn
streamlit run predictive_maintenance_app.py

## Author
Anas Al-Yousefi — Industrial Engineer
GitHub: github.com/AnasAlyousefi
