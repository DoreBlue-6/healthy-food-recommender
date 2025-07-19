import streamlit as st
import joblib
import pandas as pd
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'food_health_model.pkl')
model = joblib.load(model_path)

# Label mapping
label_map = {0: "Unhealthy", 1: "Healthy"}

# Streamlit UI
st.title("🥗 Healthy Food Recommendation System")
st.write("Enter food features to predict if it's Healthy or Unhealthy.")

# Input fields (example: total_calories)
calories = st.number_input("Total Calories", min_value=0.0, step=1.0)

# You can add more input fields here if you trained with more

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([[calories]], columns=["total_calories"])  # Adjust column name
    prediction = model.predict(input_df)
    st.success(f"🩺 Prediction: **{label_map[prediction[0]]}**")
