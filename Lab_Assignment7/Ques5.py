# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# a) Dataset Selection
# ------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------
# b) Pre-processing
# ------------------------------
# Scale features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# c) SVM Classifier with Cross-Validation
# ------------------------------
print("\n============================")
print("SVM Classifier (with 5-Fold Cross Validation)")
print("============================")

svm_model = SVC(kernel='rbf', random_state=42)

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_cv_scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='accuracy')

print("\nCross-Validation Accuracies:", np.round(svm_cv_scores, 4))
print("Average Accuracy: {:.4f}".format(svm_cv_scores.mean()))
print("Standard Deviation: {:.4f}".format(svm_cv_scores.std()))

# Train SVM with single train-test split
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_test_acc = accuracy_score(y_test, y_pred_svm)
print("\nSingle Train-Test Split Accuracy (SVM): {:.4f}".format(svm_test_acc))

# ------------------------------
# d) Random Forest Classifier + Cross Validation
# ------------------------------
print("\n============================")
print("Random Forest Classifier")
print("============================")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Accuracy on test set
rf_test_acc = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Test Accuracy: {:.4f}".format(rf_test_acc))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=data.target_names))

# Perform 5-fold CV for Random Forest
rf_cv_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
print("Cross-Validation Accuracies:", np.round(rf_cv_scores, 4))
print("Average Accuracy: {:.4f}".format(rf_cv_scores.mean()))
print("Standard Deviation: {:.4f}".format(rf_cv_scores.std()))

# ------------------------------
# e) Feature Importance (Random Forest)
# ------------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(data.feature_names)

# Plot top 10 features
plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices][:10], align='center', color='skyblue', edgecolor='k')
plt.xticks(range(10), features[indices][:10], rotation=45, ha='right')
plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Feature Names")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
for i in range(5):
    print(f"{i+1}. {features[indices[i]]} ({importances[indices[i]]:.4f})")

# ------------------------------
# f) Model Comparison
# ------------------------------
print("\n============================")
print("Model Comparison Summary")
print("============================")
print(f"SVM Cross-Val Mean Accuracy: {svm_cv_scores.mean():.4f}")
print(f"SVM Test Accuracy: {svm_test_acc:.4f}")
print(f"Random Forest Cross-Val Mean Accuracy: {rf_cv_scores.mean():.4f}")
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")

if rf_test_acc > svm_test_acc:
    print("\n✅ Random Forest performed better due to ensemble averaging and ability to handle feature interactions.")
else:
    print("\n✅ SVM performed better — likely due to better separation in high-dimensional space.")
