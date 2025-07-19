import pandas as pd

# Load your dataset
df = pd.read_csv('data/food_dataset.csv')

# Example logic: If calories > 250 → Unhealthy, else → Healthy
df['label'] = df['total_calories'].apply(lambda x: 'Unhealthy' if x > 250 else 'Healthy')

# Save it back
df.to_csv('data/food_dataset.csv', index=False)

print("✅ Labels added successfully!")
print(df[['total_calories', 'label']].head())
