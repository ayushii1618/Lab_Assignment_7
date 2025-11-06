# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------------------
# a) Dataset Selection
# ------------------------------
# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns)
print("\nFirst 5 Rows:\n", df.head())

# ------------------------------
# b) Pre-processing
# ------------------------------
# Check for missing values
print("\nMissing values in dataset:\n", df.isnull().sum())

# Standardize the features (optional but improves model stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('MedHouseVal', axis=1))
y = df['MedHouseVal']

# ------------------------------
# c) Model Training
# ------------------------------
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients and intercept
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# ------------------------------
# d) Evaluation
# ------------------------------
# Predict target values
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# ------------------------------
# Plot Predicted vs Actual Values
# ------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual House Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.grid(True)
plt.show()
