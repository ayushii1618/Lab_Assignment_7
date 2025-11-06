# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# a) Dataset Selection
# ------------------------------
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------
# b) Pre-processing
# ------------------------------
# Scale the features (optional for Decision Trees, but to match Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------------
# c) Model Training with Different Depths
# ------------------------------
depths = [3, 5, None]  # None means unlimited depth

for depth in depths:
    print(f"\n==========================")
    print(f"Decision Tree (max_depth = {depth})")
    print("==========================")
    
    # Create and train model
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # ------------------------------
    # d) Evaluation
    # ------------------------------
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy:  {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:\n", cm)

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=iris.target_names))

    # ------------------------------
    # e) Visualization of the Tree
    # ------------------------------
    plt.figure(figsize=(12, 8))
    plot_tree(clf,
              filled=True,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              rounded=True)
    plt.title(f"Decision Tree Visualization (max_depth = {depth})")
    plt.show()
