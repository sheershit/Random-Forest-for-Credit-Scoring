# 💳 Credit Scoring with Random Forest

This project implements a supervised machine learning pipeline using a Random Forest Classifier to predict the creditworthiness of loan applicants based on their financial history and demographic information. The model is trained on the numerical version of the German Credit Dataset.

---

## 🎯 Project Goals

- Predict whether a person is a good or bad credit risk.
- Use ensemble learning (Random Forest) to handle multiple features with high variance.
- Evaluate the model using robust metrics like accuracy, confusion matrix, and feature importance.
- Interpret the model’s decisions and highlight key contributing factors.

---

## 🧠 Problem Statement

Credit scoring is a crucial function in banking and finance. The aim is to automate risk classification using historical data, replacing or supporting manual evaluation by financial officers.

---

## 📊 Dataset Summary

- Name: German Credit Data (Numeric Version)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- Records: 1000
- Features: 24 numerical features
- Target Variable:
  - `1` → Good credit risk
  - `2` → Bad credit risk

> The dataset used is `german.data-numeric`, which is fully numeric and ideal for machine learning.

---

## 🛠️ Tech Stack

| Component      | Description                       |
|----------------|-----------------------------------|
|   Python       | Programming language              |
|   Pandas       | Data manipulation                 |
|   NumPy        | Numerical operations              |
|   Scikit-learn   | Model training and evaluation   |
|   Matplotlib & Seaborn   | Data visualization     |
|   Jupyter Notebook  | Interactive development     |

---

## 📁 Directory Structure    

credit-scoring-random-forest/
├── credit_scoring.ipynb        # Main Jupyter notebook
├── german.data-numeric         # Cleaned dataset used
├── german.doc                  # Dataset description
├── README.md                   # Project documentation

---

## 📈 Project Workflow

1. Import libraries and load data
2. Inspect the structure and check for nulls
3. Split data into features and labels
4. Apply train-test split with stratification
5. Standardize features using StandardScaler
6. Train a Random Forest Classifier
7. Evaluate using accuracy, confusion matrix, and classification report
8. Plot feature importance

---

## 🔍 Model Performance

| Metric     | Value (approx.) |
|------------|------------------|
| Accuracy   | ~73%             |
| Precision  | ~72%             |
| Recall     | ~75%             |
| F1-Score   | ~73%             |

The model shows a balanced trade-off between false positives and false negatives and performs well for a first iteration.

---

## 📌 Key Insights

- The Random Forest model effectively handled the numeric features without the need for extensive preprocessing.
- Feature importance revealed the top financial indicators influencing the credit decision.
- The target classes are slightly imbalanced but manageable without oversampling.

---

## 🔮 Future Work

- 🔁 Tune hyperparameters using GridSearchCV
- 📊 Add more exploratory data visualizations (histograms, correlation heatmaps)
- 🔄 Try alternative models like Logistic Regression or XGBoost
- 🌐 Deploy the model using Streamlit or Flask
- 🧠 Add SHAP/LIME for better model interpretability
