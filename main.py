import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Generate synthetic data
n_samples = 300
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Function to perform k-means clustering
def custom_k_means(data, cluster_number, iter_val):
    # Initialize centroids using K-Means++
    kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=1, random_state=42)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    for _ in range(iter_val):
        # Assign each data point to the nearest centroid
        closest_centroids = pairwise_distances_argmin_min(data, centroids)[0]

        # Update centroids as the mean of data points in each cluster
        new_centroids = []
        for i in range(cluster_number):
            cluster_points = data[closest_centroids == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])  # If a cluster is empty, keep the old centroid
        centroids = np.array(new_centroids)

    # Assign data points to the final clusters
    final_clusters = pairwise_distances_argmin_min(data, centroids)[0]

    return centroids, final_clusters

# Find the optimal number of clusters using Silhouette and Calinski-Harabasz scores
cluster_range = range(2, 11)
silhouette_scores = []
calinski_harabasz_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_std, labels)
    ch_score = calinski_harabasz_score(X_std, labels)
    silhouette_scores.append(silhouette_avg)
    calinski_harabasz_scores.append(ch_score)

optimal_cluster_number_silhouette = cluster_range[np.argmax(silhouette_scores)]
optimal_cluster_number_calinski = cluster_range[np.argmax(calinski_harabasz_scores)]

print(f"Optimal Cluster Number (Silhouette Score): {optimal_cluster_number_silhouette}")
print(f"Optimal Cluster Number (Calinski-Harabasz Score): {optimal_cluster_number_calinski}")

# Perform k-means clustering with the optimal number of clusters (e.g., 3)
final_centroids, final_clusters = custom_k_means(X_std, optimal_cluster_number_calinski, iter_val=10)

# Visualize the results with PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='viridis')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('K-Means Clustering Results (PCA)')
plt.legend()

# Visualize the Silhouette scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid()

plt.tight_layout()
plt.show()



# Assign cluster labels to the original data
X_labeled = X.copy()
X_labeled = np.column_stack((X_labeled, final_clusters))

# Plot the original data with cluster labels
plt.figure(figsize=(8, 6))
for cluster in range(optimal_cluster_number_calinski):
    cluster_data = X_labeled[X_labeled[:, -1] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data with Cluster Labels')
plt.legend()
plt.grid()
plt.show()

# Compute and display silhouette scores for individual data points
silhouette_samples = silhouette_samples(X_std, final_clusters)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(X_std)), silhouette_samples, c=final_clusters, cmap='viridis')
plt.axhline(y=np.mean(silhouette_samples), color="red", linestyle="--")
plt.xlabel('Data Point Index')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Individual Data Points')
plt.grid()
plt.show()
