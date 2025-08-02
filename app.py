import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import numpy as np

# Load model
model = tf.keras.models.load_model('model.keras')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# UI inputs
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode geography using one-hot
geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Encode gender using label encoder
gender_encoded = label_encoder_gender.transform([gender])[0]

# Combine all input features
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Merge input data with encoded geography
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output
st.write(f'**Churn Probability:** `{prediction_proba:.2f}`')

if prediction_proba > 0.5:
    st.warning('🔻 The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')
