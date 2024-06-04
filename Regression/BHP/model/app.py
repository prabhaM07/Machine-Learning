import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning)

# Load X data
X = pd.read_csv('X_data.csv')

# Define functions
def predict_price(location,sqft,bath,bhk,chosen_model):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return chosen_model.predict([x])[0]

def load_model(model_name):
    model_path = f'best_{model_name}_model.pkl'
    if os.path.exists(model_path):
        loaded_model = joblib.load(model_path)
        st.write(f"Model '{model_name}' loaded successfully.")
        return loaded_model
    else:
        st.write(f"Model file '{model_path}' does not exist.")
        return None

# Streamlit UI
st.title("Real Estate Price Prediction")

# Load model
chosen_model_name = st.selectbox("Choose Model", ['linear_regression', 'lasso', 'decision_tree'])

if chosen_model_name:
    chosen_model = load_model(chosen_model_name)

    if chosen_model:
        # Input fields
        location = st.text_input("Enter location")
        sqft = st.number_input("Enter total square feet area")
        bath = st.number_input("Enter number of bathrooms")
        bhk = st.number_input("Enter number of BHK")

        # Prediction
        if st.button("Predict Price"):
            predicted_price = predict_price(location, sqft, bath, bhk, chosen_model)
            st.write(f"Predicted price for the property in {location} with {sqft} sqft, {bath} bathrooms, and {bhk} BHK: {predicted_price}")
