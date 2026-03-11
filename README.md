# Bankruptcy Prediction Using Machine Learning

## Project Overview
This project predicts whether a company is likely to go bankrupt using machine learning models. The application analyzes financial and operational risk indicators to estimate bankruptcy risk.

An interactive web dashboard was developed using Streamlit to allow users to input company risk factors and receive real-time predictions.

---
## Live Demo

Streamlit App:
https://bankruptcy-prediction-ml-app.streamlit.app

## Features
- Machine Learning classification models
- Bankruptcy probability prediction
- Risk gauge visualization
- Feature importance analysis
- Interactive Streamlit dashboard

---

## Models Used

| Model | Accuracy |
|------|------|
| Logistic Regression | 100% |
| Random Forest | 100% |
| Support Vector Machine | 100% |
| Decision Tree | 98% |
| KNN | 98% |

Logistic Regression was selected for deployment due to its simplicity and strong performance.

---

## Input Features
The model uses the following features:

- Industrial Risk
- Management Risk
- Financial Flexibility
- Credibility
- Competitiveness
- Operating Risk

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib

## Project Structure
bankruptcy-prediction-ml-app
│
├── app.py                     # Streamlit web application
├── model.pkl                  # Trained ML model
├── columns.pkl                # Feature column order
├── bankruptcy_analysis.ipynb  # Jupyter notebook (model training)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── images/                    # Screenshots for README
│   ├── dashboard.png
│   ├── prediction.png
│   └── gauge_chart.png
│
└── data/
    └── bankruptcy_data.csv    # Dataset used for training (optional)
---

