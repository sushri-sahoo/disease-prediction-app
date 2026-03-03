import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model, mlb, le = joblib.load("model.pkl")

st.set_page_config(
    page_title="AI Disease Prediction System",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 AI Disease Prediction")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Disease", "Model Info"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":
    st.title("🩺 AI-Based Disease Prediction System")

    st.markdown("""
    This system uses **Machine Learning (Logistic Regression Classification)**  
    to predict diseases based on selected symptoms.

    ### 🔍 Features:
    - Multi-symptom selection
    - AI-based classification
    - Top-3 disease predictions
    - Confidence score visualization
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966487.png", width=200)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif menu == "Predict Disease":

    st.title("🧾 Enter Symptoms")

    selected_symptoms = st.multiselect(
        "Select Symptoms:",
        mlb.classes_
    )

    if st.button("🔎 Predict Now"):

        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")
        else:
            # Convert input to binary
            input_data = mlb.transform([selected_symptoms])

            # Predict probabilities
            probabilities = model.predict_proba(input_data)[0]

            # Get top 3 predictions
            top3_idx = np.argsort(probabilities)[-3:][::-1]

            st.subheader("🧠 Prediction Results")

            for i in top3_idx:
                disease_name = le.inverse_transform([i])[0]
                confidence = probabilities[i] * 100

                st.write(f"**{disease_name}** — {confidence:.2f}%")

            # Bar chart visualization
            chart_data = pd.DataFrame({
                "Disease": le.inverse_transform(top3_idx),
                "Confidence (%)": probabilities[top3_idx] * 100
            })

            st.bar_chart(chart_data.set_index("Disease"))

# -----------------------------
# MODEL INFO PAGE
# -----------------------------
elif menu == "Model Info":

    st.title("📊 Model Information")

    st.markdown("""
    ### Algorithm Used:
    Logistic Regression (Multi-Class Classification)

    ### Dataset:
    Symptom-based Disease Dataset  
    Total Unique Symptoms: 131  

    ### Preprocessing:
    - MultiLabelBinarizer for symptom encoding
    - LabelEncoder for disease encoding
    - Train-Test Split (80-20)

    ### Model Type:
    Supervised Machine Learning (Classification)
    """)

    st.success("Model trained successfully and integrated into web app.")