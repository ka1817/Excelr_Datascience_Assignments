import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)
@st.cache_resource  
def load_artifacts():
    try:
        model = joblib.load("Logistic_Regression_model.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'logistic_regression_model.pkl' not found. Please ensure it is in the same directory.")
        return None

model = load_artifacts()

st.title("🩺 Diabetes Risk Prediction App")
st.write("""
This application uses a **Logistic Regression** model to predict the likelihood of diabetes 
based on diagnostic clinical metrics. Adjust the values in the sidebar to generate a prediction.
""")

st.markdown("---")

st.sidebar.header(" Patient Clinical Metrics")

pregnancies = st.sidebar.number_input("Pregnancies (Count)", min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, step=1)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function Score", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f")
age = st.sidebar.slider("Age (Years)", min_value=1, max_value=120, value=33, step=1)

input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age
}

st.subheader("Current Input Summary")
st.dataframe(pd.DataFrame([input_data]))

# --- 5. PREDICTION LOGIC ---
if st.button("🔮 Generate Prediction"):
    if model is not None:
        features = np.array([[
            pregnancies, glucose, skin_thickness, 
            insulin, bmi, diabetes_pedigree, age
        ]])
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        if prediction == 1:
            st.error(f"**High Risk / Positive Outcome** (Confidence: {probabilities[1]*100:.2f}%)")
            st.write("The model classifies this clinical profile as positive for diabetes risk based on the provided metrics.")
        else:
            st.success(f"**Low Risk / Negative Outcome** (Confidence: {probabilities[0]*100:.2f}%)")
            st.write("The model classifies this clinical profile as negative/normal based on the provided metrics.")
            
        st.progress(float(probabilities[1]))
        st.caption("Probability Bar (0% Low Risk [Left] to 100% High Risk [Right])")
        
    else:
        st.warning("Prediction aborted: Model artifacts are missing.")