import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Load Model Files
# =========================

model = joblib.load("fraud_detection_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
threshold = joblib.load("fraud_threshold.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
history_file = "prediction_history.csv"

if not os.path.exists(history_file):
    history_df = pd.DataFrame(columns=[
        "Time",
        "Transaction Type",
        "Amount",
        "Fraud Probability",
        "Risk Level",
        "Prediction"
    ])
    history_df.to_csv(history_file, index=False)
feature_names = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================
# Styling
# =========================

st.markdown(
"""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1 {
    text-align:center;
    font-size:45px;
    font-weight:800;
}

p {
    font-size:18px;
}

.stButton button {

    width:100%;
    height:3em;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;

}


div[data-testid="metric-container"] {

    background-color:#f5f7fb;
    padding:15px;
    border-radius:15px;

}

</style>
""",
unsafe_allow_html=True
)



# =========================
# Header
# =========================

st.markdown(
"""
<h1>
💳 Online Payment Fraud Detection System
</h1>

<p style="text-align:center;">
AI-powered fraud detection using Random Forest Machine Learning
</p>

""",
unsafe_allow_html=True
)


st.write(
"""
This dashboard predicts whether a transaction is **Fraudulent or Normal**.

**Technologies Used**

- Python
- Pandas
- Scikit-learn
- Streamlit
"""
)


st.divider()



# =========================
# Model Information
# =========================

with st.expander("🤖 Model Information"):

    st.write(
    """
    **Algorithm:** Random Forest Classifier

    **Number of Trees:** 50

    **Maximum Depth:** 10

    **Class Handling:** Balanced class weights

    **Threshold:** Optimized using threshold tuning

    **Objective:** Detect fraudulent online transactions
    """
    )



# =========================
# Tabs
# =========================

tab1, tab2, tab3 = st.tabs(
    [
        "🏠 Fraud Prediction",
        "📊 Model Performance",
        "📈 Feature Importance"
    ]
)



# =========================
# Prediction Tab
# =========================

with tab1:


    st.subheader("📝 Transaction Details")


    col1, col2 = st.columns(2)


    with col1:

        step = st.number_input(
            "Step",
            min_value=1,
            value=1
        )


        amount = st.number_input(
            "Amount",
            min_value=0.0
        )


        oldbalanceOrg = st.number_input(
            "Old Balance Origin",
            min_value=0.0
        )


        newbalanceOrig = st.number_input(
            "New Balance Origin",
            min_value=0.0
        )



    with col2:

        transaction_type = st.selectbox(
            "Transaction Type",
            label_encoder.classes_
        )


        oldbalanceDest = st.number_input(
            "Old Balance Destination",
            min_value=0.0
        )


        newbalanceDest = st.number_input(
            "New Balance Destination",
            min_value=0.0
        )



    st.divider()



    if st.button("🔍 Predict Fraud"):


        type_encoded = label_encoder.transform(
            [transaction_type]
        )[0]


        input_data = pd.DataFrame({

            "step":[step],

            "type":[type_encoded],

            "amount":[amount],

            "oldbalanceOrg":[oldbalanceOrg],

            "newbalanceOrig":[newbalanceOrig],

            "oldbalanceDest":[oldbalanceDest],

            "newbalanceDest":[newbalanceDest]

        })

        probabilities = model.predict_proba(input_data)

        st.subheader("🐞 Debug Information")

        st.write("Input Data")
        st.dataframe(input_data)

        st.write("Column Order")
        st.write(input_data.columns.tolist())

        st.write("Encoded Transaction Type")
        st.write(type_encoded)

        st.write("Raw Model Output")
        st.write(probabilities)

        fraud_probability = probabilities[0][1]



        with st.expander("📄 Transaction Summary"):

            st.dataframe(input_data)



        st.subheader("📊 Prediction Result")
        # =========================
# Dashboard Statistics
# =========================

    

        # =========================
# Save Prediction History
# =========================

        prediction_text = "Fraud" if fraud_probability >= threshold else "Normal"

        if fraud_probability < 0.3:
            risk = "Low"
        elif fraud_probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        st.write("Prediction:", prediction_text)
        st.write("Fraud Probability:", f"{fraud_probability:.2%}")

        if risk == "Low":
            st.success("🟢 Low Risk")
        elif risk == "Medium":
             st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        new_record = pd.DataFrame({
        "Time": [datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        "Transaction Type": [transaction_type],
        "Amount": [amount],
        "Fraud Probability": [f"{fraud_probability:.2%}"],
        "Risk Level": [risk],
        "Prediction": [prediction_text]
        })
        csv = new_record.to_csv(index=False).encode('utf-8')
        history = pd.read_csv(history_file)

        history = pd.concat(
        [history, new_record],
        ignore_index=True
        )

        history.to_csv(
        history_file,
        index=False
        )

        if fraud_probability >= threshold:

            st.error(
                "🚨 Fraud Transaction Detected"
            )

        else:

            st.success(
                "✅ Normal Transaction"
            )
        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="fraud_prediction_report.csv",
            mime="text/csv"
        )



# =========================
# Prediction History Download
# =========================

st.divider()

st.subheader("📂 Prediction History")

if os.path.exists(history_file):

    history = pd.read_csv(history_file)

    st.dataframe(history)

    history_csv = history.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Complete Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )




with tab2:

    st.subheader("📊 Model Performance")


with tab2:

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "98.45%")

    with col2:
        st.metric("Recall", "99.08%")

    with col3:
        st.metric("F1 Score", "14.19%")

    st.info(
        """
        Fraud detection datasets are highly imbalanced.
        High Recall is preferred because catching fraud is more important
        than avoiding every false alert.
        """
    )

    st.divider()

    st.subheader("📌 Confusion Matrix")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(cm)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))

    ax.set_xticklabels(["Normal","Fraud"])
    ax.set_yticklabels(["Normal","Fraud"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,
                color="white"
            )

    plt.colorbar(im)

    st.pyplot(fig)
    st.divider()

    st.subheader("📈 ROC Curve")

# Predict probabilities on test set
    y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC values
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

# Plot ROC Curve
    fig2, ax2 = plt.subplots(figsize=(6,5))

    ax2.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
    )

# Random guess line
    ax2.plot(
    [0,1],
    [0,1],
    linestyle="--"
    )

    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    st.pyplot(fig2)

    st.metric("AUC Score", f"{roc_auc:.4f}")


    with col1:

        st.metric(
            "Accuracy",
            "98.45%"
        )


    with col2:

        st.metric(
            "Recall",
            "99.08%"
        )


    with col3:

        st.metric(
            "F1 Score",
            "14.19%"
        )


    st.info(
    """
    Fraud detection datasets are highly imbalanced.
    Recall is prioritized because detecting fraud cases
    is more important than avoiding every false alert.
    """
    )





# =========================
# Feature Importance Tab
# =========================

with tab3:


    st.subheader("📈 Feature Importance")


    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )


    fig, ax = plt.subplots()


    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )


    ax.set_xlabel(
        "Importance Score"
    )


    ax.set_title(
        "Random Forest Feature Importance"
    )


    st.pyplot(fig)