# Online Payment Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting potentially fraudulent online payment transactions using Machine Learning.

The project includes data preprocessing, exploratory data analysis, handling of class imbalance, machine learning-based fraud prediction, and an interactive Streamlit application for fraud-risk analysis.

## 🎯 Objectives

- Analyze online payment transaction data
- Identify patterns associated with fraudulent transactions
- Perform data preprocessing and EDA
- Handle class imbalance
- Build a Machine Learning classification workflow
- Evaluate model performance
- Develop an interactive fraud detection application

## 📊 Dataset

The dataset contains approximately **6.3 million transactions**.

It includes transaction-related features such as:

- Transaction type
- Transaction amount
- Sender balance before transaction
- Sender balance after transaction
- Receiver balance before transaction
- Receiver balance after transaction
- Fraud indicator

### Target Variable

`isFraud`

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

> The original dataset is not included in this repository because of its large size.

## 🔍 Project Workflow

```text
Data Collection
      ↓
Data Loading
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Class Imbalance Handling
      ↓
Feature Preparation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Fraud Prediction
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Power BI
- Streamlit
- Joblib

## 📈 Exploratory Data Analysis

The project analyzes:

- Transaction type distribution
- Transaction amounts
- Fraudulent vs legitimate transactions
- Fraud patterns across transaction types
- Transaction-level characteristics associated with fraud

## ⚖️ Class Imbalance

Fraudulent transactions are significantly fewer than legitimate transactions.

Therefore, class imbalance was considered during the Machine Learning workflow to improve fraud detection performance.

## 🤖 Machine Learning

A supervised classification approach is used to predict whether a transaction is fraudulent.

The workflow includes:

1. Feature preparation
2. Train-test splitting
3. Model training
4. Prediction
5. Model evaluation
6. Fraud probability and risk classification

## 🌐 Streamlit Application

An interactive Streamlit application was developed for fraud-risk analysis.

The application provides:

- Transaction-level fraud prediction
- Fraud probability
- Risk classification
- Low / Medium / High risk categorization
- Prediction history
- Model performance visualization

The application can be launched using:

```bash
py -m streamlit run app.py
```

> The required model files are kept locally and are not included in this repository.

## 📊 Power BI Dashboard

A Power BI dashboard was developed to analyze transaction and fraud patterns.

The dashboard includes:

- KPI cards
- Fraud transaction analysis
- Fraud amount analysis
- Fraud rate analysis
- Transaction-type analysis
- Interactive slicers
- Data visualizations

> The `.pbix` file is not included in the GitHub repository because of its large file size. The dashboard can be demonstrated locally using Power BI Desktop.

## 📁 Repository Structure

```text
online_payment_fraud_detection_ml/
│
├── Online_Payment_Fraud_Detection.ipynb
├── app.py
└── README.md
```

## 🚀 How to Run the Streamlit Application

Clone the repository:

```bash
git clone https://github.com/vkboss4353-ux/online_payment_fraud_detection_ml.git
```

Navigate to the project directory:

```bash
cd online_payment_fraud_detection_ml
```

Run the Streamlit application:

```bash
py -m streamlit run app.py
```

> The required model files and dataset are not included in the repository and need to be available locally for the complete application to run.

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Data preprocessing
- Exploratory Data Analysis
- Handling imbalanced datasets
- Machine Learning classification
- Fraud detection
- Model evaluation
- Data visualization
- Power BI dashboard development
- Streamlit application development
- Working with large datasets

## 🔮 Future Improvements

- Feature engineering
- Hyperparameter optimization
- Model explainability using SHAP
- Real-time fraud monitoring
- API-based model deployment
- Real-time transaction risk scoring

## 👨‍💻 Author

**Vishal Kumar**

Engineering Student | Machine Learning | Data Analytics | Cybersecurity
