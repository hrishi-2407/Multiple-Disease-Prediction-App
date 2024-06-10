import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Define the path to the saved --ML-- models directory
ml_models_dir = 'F:\Multiple Disease Prediction\saved ml models'

# Construct the full file paths for each --ML-- model
diabetes_model_path = os.path.join(ml_models_dir, 'diabetes.pkl')
heart_disease_model_path = os.path.join(ml_models_dir, 'heart_disease.pkl')
parkinsons_model_path = os.path.join(ml_models_dir, 'parkinsons.pkl')

# Load the --ML-- models
diabetes_model = joblib.load(diabetes_model_path)
heart_disease_model = joblib.load(heart_disease_model_path)
parkinsons_model = joblib.load(parkinsons_model_path)


# Define the path to the saved --SCALER-- models directory
scaler_models_dir = 'F:\Multiple Disease Prediction\saved scaler models'

# Construct the full file paths for each --SCALER-- model
diabetes_scaler_model = os.path.join(scaler_models_dir, 'scaler_diabetes.pkl')
heart_disease_scaler_model = os.path.join(scaler_models_dir, 'scaler_heart.pkl')
parkinsons_scaler_model = os.path.join(scaler_models_dir, 'scaler_parkinsons.pkl')

# Load the --SCALER-- models
diabetes_scaler = joblib.load(diabetes_scaler_model)
heart_disease_scaler = joblib.load(heart_disease_scaler_model)
parkinsons_scaler = joblib.load(parkinsons_scaler_model)

# Function to make predictions
def predict_diabetes(input_data):
    # input_data = np.asarray(input_data).reshape(1, -1)
    # Feature names based on the original dataset
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Convert input data to a DataFrame with feature names
    input_data_as_df = pd.DataFrame([input_data], columns=feature_names)
    input_data_scaled = diabetes_scaler.transform(input_data_as_df)
    prediction = diabetes_model.predict(input_data_scaled)
    if prediction[0] == 1:
        return "You might be Diabetic. Please consult a doctor and take necessary steps."
    else:
        return "You are not Diabetic. However, it's always good to maintain a healthy lifestyle."
    
def predict_heart_disease(input_data):
    # input_data = np.asarray(input_data).reshape(1, -1)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data_as_df = pd.DataFrame([input_data], columns=feature_names)
    input_data_scaled = heart_disease_scaler.transform(input_data_as_df)
    prediction = heart_disease_model.predict(input_data_scaled)
    if prediction[0] == 1:
        return "You might have Heart Disease. Please consult a doctor and take necessary steps."
    else:
        return "You don't seem to have Heart Disease. However, it's advisable to maintain a healthy lifestyle."

def predict_parkinsons(input_data):
    # input_data = np.asarray(input_data).reshape(1, -1)
    feature_names = ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    input_data_as_df = pd.DataFrame([input_data], columns=feature_names)
    input_data_scaled = parkinsons_scaler.transform(input_data_as_df)
    prediction = parkinsons_model.predict(input_data_scaled)
    if prediction[0] == 1:
        return "You might have Parkinson's Disease. Please consult a doctor and take necessary steps."
    else:
        return "You don't seem to have Parkinson's Disease. However, if you have concerns, it's best to consult a specialist."

# Streamlit app
# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes', 'Heart Disease', 'Parkinsons'])

if selected == 'Diabetes':
    st.title("Diabetes Prediction")
    st.write('Diabetes is a chronic condition that affects how your body turns food into energy. Use this tool to predict the likelihood of diabetes based on various health parameters.')
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose")
    BloodPressure = st.text_input("BloodPressure")
    SkinThickness = st.text_input("SkinThickness (subcutaneous fat)")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI (Body mass index)")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction (score of likelihood of diabetes based on family history)")
    Age = st.text_input("Age")
    
    if st.button("Predict the Result"):
        result = predict_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.write(result)

elif selected == 'Heart Disease':
    st.title("Heart Disease Prediction")
    st.write('Heart disease refers to various types of heart conditions. This tool helps predict the risk of heart disease using multiple cardiovascular health indicators.')
    age = st.text_input("Age")
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("CP (chest pain type)", [0, 1, 2, 3])
    trestbps = st.text_input("Trestbps (resting blood pressure)")
    chol = st.text_input("Chol (serum cholesterol)")
    fbs = st.selectbox("FBS (fasting blood sugar)", [0, 1])
    restecg = st.selectbox("Restecg (resting electrocardiographic results)", [0, 1, 2])
    thalach = st.text_input("Thalach (maximum heart rate achieved during exercise)")
    exang = st.selectbox("Exang (exercise induced angina)", [0, 1])
    oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest)")
    slope = st.selectbox("Slope (slope of the peak exercise ST segment)", [0, 1, 2])
    ca = st.text_input("CA (number of major vessels colored by fluoroscopy)")
    thal = st.selectbox("Thal (Thalassemia blood disorder)", [0, 1, 2, 3])
    
    if st.button("Predict the Result"):
        result = predict_heart_disease([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        st.write(result)

elif selected == 'Parkinsons':
    st.title("Parkinson's Disease Prediction")
    st.write('Parkinsons disease is a neurodegenerative disorder that affects movement. This tool predicts the likelihood of Parkinsons disease based on voice measurements and other factors.')
    fo = st.text_input("fo (Average fundamental frequency)")
    fhi = st.text_input("fhi (Maximum fundamental frequency)")
    flo = st.text_input("flo (Minimum fundamental frequency)")
    Jitter_percent = st.text_input("Jitter_percent (Frequency variation- measured in percentage)")
    Jitter_Abs = st.text_input("Jitter_Abs (Absolute jitter- measured in seconds)")
    RAP = st.text_input("RAP (Relative amplitude perturbation)")
    PPQ = st.text_input("PPQ (Five-point period perturbation quotient)")
    DDP = st.text_input("DDP (Average absolute difference of differences between periods)")
    Shimmer = st.text_input("Shimmer (Amplitude variation in the voice)")
    Shimmer_dB = st.text_input("Shimmer_dB Amplitude variation in the voice in decibels")
    APQ3 = st.text_input("APQ3 (Three-point amplitude perturbation quotient)")
    APQ5 = st.text_input("APQ5  (Five-point amplitude perturbation quotient)")
    APQ = st.text_input("APQ (Amplitude perturbation quotient)")
    DDA = st.text_input("DDA (Average absolute difference of differences between amplitudes)")
    NHR = st.text_input("NHR (Noise-to-harmonics ratio)")
    HNR = st.text_input("HNR (Harmonics-to-noise ratio)")
    RPDE = st.text_input("RPDE (Recurrence period density entropy)")
    DFA = st.text_input("DFA (Detrended fluctuation analysis)")
    spread1 = st.text_input("spread1 (Nonlinear measure of fundamental frequency variation)")
    spread2 = st.text_input("spread2 (Nonlinear measure of fundamental frequency variation)")
    D2 = st.text_input("D2 (Correlation dimension)")
    PPE = st.text_input("PPE (Pitch period entropy)")
    
    if st.button("Predict the Result"):
        input_data = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
        result = predict_parkinsons(input_data)
        st.write(result)