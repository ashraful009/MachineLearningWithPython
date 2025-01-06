import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load(r"MachineLearningWithPython\Sessions\Day12\insurance_prediction_model.pkl")
scaler = joblib.load(r"MachineLearningWithPython\Sessions\Day12\scaler.joblib")

def predict_charges(age, bmi, children, smoker):
    # Create input array with only the required features
    features = np.array([[age, bmi, children, smoker]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    return prediction[0]

# Create the web app
st.title('Insurance Charges Prediction')

# Get user inputs for only the required features
age = st.slider('Age', 18, 100, 25)
bmi = st.number_input('BMI', 15.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 10, 0)
smoker = st.selectbox('Smoker', ['yes', 'no'])

# Convert smoker to binary
smoker = 1 if smoker == 'yes' else 0

if st.button('Predict Insurance Charges'):
    prediction = predict_charges(age, bmi, children, smoker)
    st.success(f'Predicted Insurance Charges: ${prediction:,.2f}')
