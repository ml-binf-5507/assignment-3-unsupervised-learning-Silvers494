from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from dimensionality_reduction import apply_pca, apply_tsne, apply_umap
from clustering import run_kmeans, run_hierarchical
from visualization import plot_representation_comparison, plot_cluster_comparison

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2d = apply_pca(X_scaled)
tsne_2d = apply_tsne(X_scaled)
umap_2d = apply_umap(X_scaled)

kmeans_labels, kmeans_centers = run_kmeans(X_scaled, n_clusters=2)
# hier_labels = run_hierarchical(X_scaled, n_clusters=2)  # if you'd rather use this one

plot_representation_comparison(pca_2d, tsne_2d, umap_2d, true_labels=y, filename="figure1.png")
plot_cluster_comparison(pca_2d, tsne_2d, umap_2d, cluster_labels=kmeans_labels, filename="figure2.png")