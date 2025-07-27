#Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


# Load the model and preprocessing objects
model=tf.keras.models.load_model('model.h5')

#Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder.categories_[0]) # onehot_encoder.categories_[0] means the first category of the one-hot encoded geography
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],  # Transform [0] means we are getting the first element of the transformed array
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
}) 
# }) means we are creating a dictionary with the input data, where each key is the name of the column and each value is a list with one element, which is the value of the input data
#() this is converting the input data into a dictionary, where each key is the name of the column and each value is a list with one element, which is the value of the input data

# One-hot encode the geography
geo_encode=onehot_encoder.transform([[geography]]).toarray()# why [[geography]]? because the onehot_encoder expects a 2D array, so we need to wrap geography in an additional list to make it a 2D array
# what that 2d array looks like? it looks like [[1, 0, 0]] if geography is France, [[0, 1, 0]] if geography is Spain and [[0, 0, 1]] if geography is Germany
#it is 1d array, so we need to wrap it in an additional list to make it a 2D array
#after additional list, it looks like [[1, 0, 0]] if geography is France, [[0, 1, 0]] if geography is Spain and [[0, 0, 1]] if geography is Germany

# Create a DataFrame for the input data
geo_enocded_df = pd.DataFrame(geo_encode, columns=onehot_encoder.get_feature_names_out(['Geography']))


#combine the input data with the one-hot encoded geography
input_data=pd.concat([input_data.reset_index(drop=True), geo_enocded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions Churn
prediction= model.predict(input_data_scaled)
prediction_probability = prediction[0][0] # Get the probability of churn
# [0][0] means we are getting the first element of the first array, because the model returns a 2D array with one row and one column

if prediction_probability > 0.5:
    st.write("Customer will exit")
else:
    st.write("Customer will not exit")

# Display the prediction probability
st.write(f"Prediction Probability: {prediction_probability:.2f}") 