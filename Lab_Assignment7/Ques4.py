# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA

# ------------------------------
# a) Dataset Selection
# ------------------------------
iris = load_iris()
X = iris.data
y_true = iris.target  # true labels (only for evaluation, not used in clustering)

df = pd.DataFrame(X, columns=iris.feature_names)
print("Dataset Loaded Successfully!")
print(df.head())

# ------------------------------
# b) Preprocessing
# ------------------------------
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# c) Apply K-Means Clustering
# ------------------------------
k = 3  # since Iris dataset has 3 natural classes
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Print cluster centers and inertia
print("\nCluster Centers:\n", kmeans.cluster_centers_)
print("\nInertia (Sum of Squared Distances):", kmeans.inertia_)

# ------------------------------
# d) Evaluate Clustering
# ------------------------------
# Elbow Method to find optimal k
inertias = []
K = range(1, 11)
for i in K:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# ------------------------------
# e) Compare with True Labels (if available)
# ------------------------------
# Adjusted Rand Index (measures similarity between true and predicted clusters)
ari = adjusted_rand_score(y_true, y_kmeans)

# Silhouette Score (measures how well data points fit within clusters)
sil_score = silhouette_score(X_scaled, y_kmeans)

print("\nAdjusted Rand Index (ARI):", ari)
print("Silhouette Score:", sil_score)

# Optional: Rough Accuracy (best label mapping)
# Note: Cluster labels are arbitrary; we align them to true labels manually
from scipy.stats import mode

def cluster_accuracy(y_true, y_kmeans):
    labels = np.zeros_like(y_kmeans)
    for i in range(k):
        mask = (y_kmeans == i)
        labels[mask] = mode(y_true[mask], keepdims=True)[0]
    return accuracy_score(y_true, labels)

acc = cluster_accuracy(y_true, y_kmeans)
print("Cluster Accuracy (approximate):", acc)

# ------------------------------
# Visualization using PCA (2D Projection)
# ------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='rainbow', s=50, alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering (Iris Dataset - PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
