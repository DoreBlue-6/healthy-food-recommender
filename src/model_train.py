import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('data/food_dataset.csv')

# Features and labels
X = df[['total_calories']]
y = df['label']
y = y.map({'Unhealthy': 0, 'Healthy': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save model
joblib.dump(model, 'food_health_model.pkl')
print("✅ Model saved as 'food_health_model.pkl'")
