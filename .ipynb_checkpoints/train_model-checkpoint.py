"""
Simple script to train and save the Gradient Boosting model
Run this once before starting the Flask app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load cleaned data
print("Loading data...")
df = pd.read_csv('cardio.csv')

# Prepare features and target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")
print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")

