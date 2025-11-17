"""
Embedding utility functions for novelty and regime analysis.

Provides helper functions to compute:
- Embedding novelty scores (distance to existing embeddings)
- Regime clustering (grouping episodes by visual features)

These are additive helpers - no changes to Phase B math or RL training loops.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any


def compute_embedding_novelty(
    embedding: np.ndarray,
    reference_embeddings: List[np.ndarray],
    k: int = 5,
    method: str = "mean_k_nearest"
) -> float:
    """
    Compute novelty score for an embedding relative to reference set.

    Args:
        embedding: Query embedding (1D array)
        reference_embeddings: List of reference embeddings
        k: Number of nearest neighbors to consider
        method: Novelty computation method
            - "mean_k_nearest": Mean distance to k nearest neighbors
            - "min_distance": Distance to nearest neighbor
            - "percentile_10": 10th percentile distance

    Returns:
        Novelty score (higher = more novel)

    Note:
        This is a logging/analysis helper - does not affect rewards or training.
    """
    if len(reference_embeddings) == 0:
        # First embedding is maximally novel
        return 1.0

    # Compute distances to all reference embeddings
    distances = []
    for ref_emb in reference_embeddings:
        dist = np.linalg.norm(embedding - ref_emb)
        distances.append(dist)

    distances = np.array(distances)

    if method == "mean_k_nearest":
        k_actual = min(k, len(distances))
        k_nearest = np.sort(distances)[:k_actual]
        novelty = float(np.mean(k_nearest))

    elif method == "min_distance":
        novelty = float(np.min(distances))

    elif method == "percentile_10":
        novelty = float(np.percentile(distances, 10))

    else:
        raise ValueError(f"Unknown novelty method: {method}")

    return novelty


def compute_embedding_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    method: str = "cosine"
) -> float:
    """
    Compute similarity between two embeddings.

    Args:
        embedding1: First embedding (1D array)
        embedding2: Second embedding (1D array)
        method: Similarity metric
            - "cosine": Cosine similarity
            - "euclidean": Negative Euclidean distance
            - "dot": Dot product

    Returns:
        Similarity score (higher = more similar)
    """
    if method == "cosine":
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    elif method == "euclidean":
        # Return negative distance so higher = more similar
        return -float(np.linalg.norm(embedding1 - embedding2))

    elif method == "dot":
        return float(np.dot(embedding1, embedding2))

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_regime_cluster(
    embedding: np.ndarray,
    regime_centroids: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Assign embedding to nearest regime cluster.

    Args:
        embedding: Query embedding (1D array)
        regime_centroids: Dict mapping regime name to centroid embedding

    Returns:
        Tuple of (regime_name, confidence_score)
        - regime_name: Name of closest regime
        - confidence_score: Normalized confidence (0-1, higher = more confident)

    Note:
        This is for logging/analysis - does not affect training behavior.
    """
    if not regime_centroids:
        return "unknown", 0.0

    # Compute distances to each regime centroid
    distances = {}
    for regime_name, centroid in regime_centroids.items():
        dist = np.linalg.norm(embedding - centroid)
        distances[regime_name] = dist

    # Find closest regime
    closest_regime = min(distances, key=distances.get)
    min_dist = distances[closest_regime]

    # Compute confidence as inverse of distance (normalized)
    # Higher confidence when distance is small relative to other distances
    if len(distances) == 1:
        confidence = 1.0
    else:
        all_dists = np.array(list(distances.values()))
        # Softmax-like confidence
        exp_neg_dists = np.exp(-all_dists)
        confidence = float(np.exp(-min_dist) / np.sum(exp_neg_dists))

    return closest_regime, confidence


def build_regime_centroids_from_embeddings(
    embeddings: List[np.ndarray],
    labels: List[str]
) -> Dict[str, np.ndarray]:
    """
    Build regime centroids from labeled embeddings.

    Args:
        embeddings: List of embeddings
        labels: List of regime labels (same length as embeddings)

    Returns:
        Dict mapping regime name to mean centroid embedding
    """
    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must have same length")

    # Group embeddings by label
    regime_groups = {}
    for emb, label in zip(embeddings, labels):
        if label not in regime_groups:
            regime_groups[label] = []
        regime_groups[label].append(emb)

    # Compute mean centroid for each regime
    centroids = {}
    for regime_name, emb_list in regime_groups.items():
        centroids[regime_name] = np.mean(emb_list, axis=0)

    return centroids


def cluster_embeddings_kmeans(
    embeddings: List[np.ndarray],
    n_clusters: int = 3,
    max_iters: int = 100,
    seed: int = 42
) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """
    Simple K-means clustering for regime discovery.

    Args:
        embeddings: List of embeddings to cluster
        n_clusters: Number of clusters (regimes) to find
        max_iters: Maximum iterations for K-means
        seed: Random seed for reproducibility

    Returns:
        Tuple of (cluster_assignments, centroids)
        - cluster_assignments: List of cluster IDs for each embedding
        - centroids: Dict mapping cluster ID to centroid embedding

    Note:
        This is a simple implementation for exploration.
        For production, use sklearn.cluster.KMeans.
    """
    if len(embeddings) < n_clusters:
        # Not enough data; assign each to its own cluster
        assignments = list(range(len(embeddings)))
        centroids = {i: embeddings[i] for i in range(len(embeddings))}
        return assignments, centroids

    np.random.seed(seed)
    embeddings_array = np.array(embeddings)

    # Initialize centroids randomly
    indices = np.random.choice(len(embeddings), n_clusters, replace=False)
    centroids_array = embeddings_array[indices].copy()

    assignments = np.zeros(len(embeddings), dtype=int)

    for iteration in range(max_iters):
        # Assignment step: assign each embedding to nearest centroid
        old_assignments = assignments.copy()
        for i, emb in enumerate(embeddings_array):
            distances = np.linalg.norm(centroids_array - emb, axis=1)
            assignments[i] = np.argmin(distances)

        # Check for convergence
        if np.all(assignments == old_assignments):
            break

        # Update step: recompute centroids
        for k in range(n_clusters):
            cluster_members = embeddings_array[assignments == k]
            if len(cluster_members) > 0:
                centroids_array[k] = np.mean(cluster_members, axis=0)

    # Convert to dict format
    centroids_dict = {k: centroids_array[k] for k in range(n_clusters)}

    return assignments.tolist(), centroids_dict


def compute_embedding_statistics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute summary statistics for a collection of embeddings.

    Args:
        embeddings: List of embeddings

    Returns:
        Dict with statistics:
        - mean_norm: Mean L2 norm of embeddings
        - std_norm: Std of L2 norms
        - mean_pairwise_distance: Mean distance between embeddings
        - variance_explained_pc1: Variance explained by first PC (rough)
    """
    if len(embeddings) == 0:
        return {
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "mean_pairwise_distance": 0.0,
            "variance_explained_pc1": 0.0,
        }

    embeddings_array = np.array(embeddings)

    # Norm statistics
    norms = np.linalg.norm(embeddings_array, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    # Pairwise distance (sample if too many)
    if len(embeddings) <= 100:
        pairwise_dists = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
                pairwise_dists.append(dist)
        mean_pairwise = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
    else:
        # Sample pairwise distances
        np.random.seed(42)
        n_samples = 500
        pairwise_dists = []
        for _ in range(n_samples):
            i, j = np.random.choice(len(embeddings), 2, replace=False)
            dist = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
            pairwise_dists.append(dist)
        mean_pairwise = float(np.mean(pairwise_dists))

    # Simple variance explained by first PC (using power iteration)
    if len(embeddings) >= 2:
        centered = embeddings_array - np.mean(embeddings_array, axis=0)
        # One iteration of power method to approximate first PC
        v = np.random.randn(embeddings_array.shape[1])
        v = v / np.linalg.norm(v)
        for _ in range(10):
            Av = centered.T @ (centered @ v)
            v = Av / np.linalg.norm(Av)
        # Variance along first PC
        projections = centered @ v
        var_pc1 = float(np.var(projections))
        total_var = float(np.sum(np.var(centered, axis=0)))
        variance_explained_pc1 = var_pc1 / total_var if total_var > 0 else 0.0
    else:
        variance_explained_pc1 = 0.0

    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "mean_pairwise_distance": mean_pairwise,
        "variance_explained_pc1": variance_explained_pc1,
    }
