import joblib
import pandas as pd
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'food_health_model.pkl')
model = joblib.load(model_path)

# Define the label mapping (update this if needed)
label_map = {0: "Unhealthy", 1: "Healthy"}

# Sample input
sample_input = pd.DataFrame([[95.73]], columns=["total_calories"])

# Make prediction
prediction = model.predict(sample_input)

# Print readable output
print("Prediction:", label_map[prediction[0]])
