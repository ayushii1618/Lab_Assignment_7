# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mlxtend.plotting import plot_decision_regions

# ------------------------------
# a) Dataset Selection
# ------------------------------
# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features for visualization (sepal length & sepal width)
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names[:2])
df['target'] = y

print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------
# b) Pre-processing
# ------------------------------
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------------
# c) Model Training
# ------------------------------
# Create and train Logistic Regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)

# Print model coefficients and intercepts
print("\nModel Coefficients (for each class):\n", model.coef_)
print("Model Intercepts:\n", model.intercept_)

# ------------------------------
# d) Evaluation
# ------------------------------
# Predict categories on test set
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------
# e) Visualization (Decision Boundary)
# ------------------------------
plt.figure(figsize=(8, 6))
plot_decision_regions(X_scaled, y, clf=model, legend=2)
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.title("Decision Boundaries using Logistic Regression (Iris Dataset)")
plt.show()
