import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- Step 1: Synthetic High-Dimensional Dataset Setup & Preprocessing ---
# Creating a dummy 6-dimensional dataset for illustration
X_raw, _ = make_blobs(
    n_samples=500, n_features=6, centers=4, cluster_std=1.5, random_state=42
)
feature_names = [f"Feature_{i+1}" for i in range(6)]
df = pd.DataFrame(X_raw, columns=feature_names)

# Standardize high-dimensional numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- Step 2: Principal Component Analysis (PCA) & Scree Plot ---
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot Explained Variance Ratio (Scree Plot)
plt.figure(figsize=(8, 5))
plt.bar(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    alpha=0.6,
    align="center",
    label="Individual Explained Variance",
)
plt.step(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    where="mid",
    label="Cumulative Explained Variance",
    color="red",
)
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Component Index")
plt.title("PCA Scree Plot")
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Retain optimal components (e.g., top 3 components)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# --- Step 3: Optimal K Selection (Elbow Method & Silhouette Score) ---
inertia = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca_3d)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca_3d, kmeans.labels_))

fig, ax1 = plt.subplots(figsize=(10, 4))

color = "tab:blue"
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Elbow Method)", color=color)
ax1.plot(k_range, inertia, marker="o", color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Silhouette Score", color=color)
ax2.plot(k_range, silhouette_scores, marker="s", linestyle="--", color=color)
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Elbow Method & Silhouette Score Analysis for K-Means")
plt.grid(True)
plt.show()

# --- Step 4: Clustering Execution ---
optimal_k = 4

# K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca_3d)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca_3d)

# Hierarchical / Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_pca_3d)

# --- Step 5: 2D & 3D Cluster Visualizations ---
df_pca = pd.DataFrame(
    X_pca_3d, columns=["PC1", "PC2", "PC3"]
)
df_pca["KMeans_Cluster"] = kmeans_labels.astype(str)

# 2D Projection using Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca, x="PC1", y="PC2", hue="KMeans_Cluster", palette="viridis", s=70
)
plt.title("2D PCA Projection with K-Means Clusters")
plt.grid(True)
plt.show()

# 3D Interactive Projection using Plotly
fig_3d = px.scatter_3d(
    df_pca,
    x="PC1",
    y="PC2",
    z="PC3",
    color="KMeans_Cluster",
    title="3D PCA Cluster Projection (Plotly)",
    opacity=0.8,
)
fig_3d.show()