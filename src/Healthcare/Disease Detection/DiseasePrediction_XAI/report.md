# Disease Prediction with Explainable AI (SHAP)

## 🔍 Project Overview

This project focuses on predicting disease presence using patient health indicators from a clinical dataset. The prediction model not only forecasts outcomes but also explains the decision process using **SHAP (SHapley Additive exPlanations)** to ensure model transparency and interpretability for healthcare professionals.

---

## 🎯 Objectives

1. **Disease Prediction** using a Random Forest Classifier trained on medical features.
2. Provide **Explainable AI (XAI)** outputs using SHAP for feature contribution visualization.
3. Enable an easy-to-use **Streamlit Web App** for real-time inference and explanation.

---

## 🏗️ Project Architecture

DiseasePrediction-XAI/
│
├── data/
│ └── dataset-uci.xlsx # UCI dataset used for model training
│
├── models/
│ ├── random_forest_model.pkl # Trained RandomForest model
│ └── scaler.pkl # StandardScaler fitted on training data
│
├── notebooks/
│ ├── data_preprocessing.ipynb # Data loading & preprocessing
│ ├── model_training.ipynb # Model training and evaluation
│ └── shap_explainability.ipynb # SHAP analysis and visualizations
│
├── app/
│ └── app.py # Streamlit application
│
├── requirements.txt # List of Python dependencies
├── README.md # Project overview and instructions
└── report.md # Detailed project report (this file)


---

## 📊 Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Total Features:** 41 features including:
  - **Demographics:** Age, Gender
  - **Anthropometric:** Height, Weight, BMI, Body Composition Metrics
  - **Biochemical:** Glucose, Cholesterol, Creatinine, CRP, Vitamin D, etc.
  - **Clinical History:** Diabetes Mellitus, Coronary Artery Disease (CAD), Hypothyroidism, Hyperlipidemia
- **Target:** Binary outcome indicating disease presence (`1`) or absence (`0`).

---

## ⚙️ Model Details

| Component          | Description                          |
|-------------------|--------------------------------------|
| **Algorithm**      | Random Forest Classifier (Sklearn)    |
| **Scaler**        | StandardScaler (mean=0, std=1)        |
| **Input Features** | 41 Health-related features           |
| **Output**        | Probability and class label (Disease/No Disease) |

---

## 🌐 Streamlit Application Features

- **Dynamic User Input:** Enter patient health metrics via UI sliders and inputs.
- **Disease Prediction:** Predicts whether the patient is likely to have the disease.
- **SHAP Explainability:**
  - **Waterfall Plot:** Shows cumulative feature contribution.
  - **Force Plot:** Visual force plot for feature-level impact.
  - **Feature Importance Bar Plot:** Top feature impacts sorted by absolute SHAP values.
- **Interpretation Guide:** Helps users understand SHAP visualizations.

---

## 🧩 SHAP Explainability

- **Why SHAP?**
  - Ensures transparency in medical AI predictions.
  - Offers feature-wise reasoning for each individual patient prediction.
- **SHAP Visual Outputs Provided:**
  1. Waterfall Plot
  2. Force Plot (interactive and custom fallback)
  3. Feature Importance Ranking

---

## 🔍 Evaluation Metrics

| Metric         | Value (Example) |
|----------------|----------------|
| Accuracy      | ~85%            |
| Precision     | 82%             |
| Recall        | 80%             |
| ROC-AUC       | 0.89            |

*These values are from cross-validation and may vary depending on hyperparameters and dataset split.*

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the Streamlit app:
    cd app
    streamlit run app.py
3. Input feature values in the sidebar and click "Predict Disease" to see predictions and SHAP explanations.

