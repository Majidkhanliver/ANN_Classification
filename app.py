import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder 
import pandas as pd
import pickle

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)  
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

## Streamlit app
st.title("Customer Churn Prediction")   

# Input fields
geography = st.selectbox("Geography", onehot_encoder.categories_[0])  # Exclude first category for one-hot encoding
gender = st.selectbox("Gender", label_encoder.classes_  )
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Tenure", 0,10)
num_of_products = st.slider("Number of Products", 1,4 )
has_cr_card = st.radio("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],     
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],   
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True ), geo_encoded_df], axis=1)  # Exclude first column to avoid dummy variable trap

# Scale the input data
input_data_scaled = scaler.transform(input_data)    

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_prob = prediction[0][0]
    if churn_prob > 0.5:
        st.error(f"The customer is likely to churn with a probability of {churn_prob:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1 - churn_prob:.2f}")