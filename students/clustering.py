"""
clustering.py

Wrappers for k-means and hierarchical clustering.  All functions should return
labels and any additional information required for plotting or evaluation.
"""

from typing import Any, Tuple

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def run_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Any]:
    """Run k-means clustering and return labels and cluster centers.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Integer cluster labels for each sample.
    centers : Any
        Object representing cluster centers (e.g. np.ndarray).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers


def run_hierarchical(X: np.ndarray, n_clusters: int = 3, linkage: str = "ward") -> np.ndarray:
    """Perform agglomerative clustering and return labels.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int
        The number of clusters to find.
    linkage : str
        Linkage criterion for clustering ("ward", "complete", etc.).

    Returns
    -------
    labels : np.ndarray
        Integer labels for each sample.
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return labels