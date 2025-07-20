import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="Salary Scope", layout="wide")

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

animation = load_lottie("animations/money.json")
st_lottie(animation, height=180, key="money")

# Load model and encoders
model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.markdown("<h2 class='header'>Salary Scope - AI Salary Predictor</h2>", unsafe_allow_html=True)

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        workclass = st.selectbox("Workclass", label_encoders["workclass"].classes_)
        education = st.selectbox("Education", label_encoders["education"].classes_)
        marital_status = st.selectbox("Marital Status", label_encoders["marital-status"].classes_)
        occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)

    with col2:
        relationship = st.selectbox("Relationship", label_encoders["relationship"].classes_)
        race = st.selectbox("Race", label_encoders["race"].classes_)
        gender = st.selectbox("Gender", label_encoders["gender"].classes_)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", label_encoders["native-country"].classes_)

    submitted = st.form_submit_button("🎯 Predict Income Class")

    if submitted:
        input_data = pd.DataFrame([{
            'age': age,
            'workclass': workclass,
            'education': education,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'gender': gender,
            'hours-per-week': hours_per_week,
            'native-country': native_country
        }])

        # Encode categorical inputs
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Ensure column order matches training
        input_data = input_data[feature_names]

        # Predict class
        prediction = model.predict(input_data)[0]

        # Decode income class back to string
        decoded_prediction = label_encoders["income"].inverse_transform([prediction])[0]

        st.success(f"✅ Predicted Income Class: **{decoded_prediction}**")
