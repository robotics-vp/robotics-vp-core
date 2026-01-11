"""Small-world graph construction and metrics for geometry embeddings.

This module implements graph construction from BEV grid embeddings or token sequences,
computing small-world metrics (clustering, path length, sigma) and bounded navigability.

Key concepts:
- Local edges: 4/8-neighbor lattice on 2D grid (or 2-neighbor on 1D tokens)
- Shortcut edges: kNN in embedding space, filtered by minimum lattice distance
- Small-worldness σ = (C/C_rand)/(L/L_rand) where C=clustering, L=path length
- Navigability: Greedy routing using embedding distance to target
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.contracts.schemas import GraphSpecV1, GraphSummaryV1
from src.utils.config_digest import sha256_json


@dataclass
class GraphEdgeCounts:
    """Edge count statistics."""
    local_edges: int
    shortcut_edges: int
    total_edges: int


@dataclass
class GraphMetrics:
    """Computed graph metrics."""
    clustering_coefficient: float
    avg_path_length: float
    c_rand: float
    l_rand: float
    sigma: float
    shortcut_fraction: float
    shortcut_lattice_hop_mean: float
    shortcut_lattice_hop_p50: float
    shortcut_lattice_hop_p90: float
    shortcut_score_mean: float
    shortcut_score_min: float
    shortcut_score_max: float
    shortcut_score_p50: float
    shortcut_score_p90: float
    nav_success_rate: float
    nav_mean_hops: float
    nav_stretch: float
    nav_visited_nodes_mean: float
    nav_success_lattice: float
    nav_gain: float


def build_small_world_graph(
    embeddings: np.ndarray,
    grid_shape: Optional[Tuple[int, int]],
    config: GraphSpecV1,
    seed: int,
) -> Tuple[Dict[int, List[int]], GraphEdgeCounts, List[int], Dict[int, List[int]], List[float]]:
    """Build lattice + kNN shortcut graph with score-based selection.
    
    Args:
        embeddings: (H, W, D) for grid mode or (N, D) for token mode
        grid_shape: (H, W) if grid mode, None for token/pooled mode
        config: Graph construction configuration
        seed: Random seed for reproducibility
        
    Returns:
        adjacency: Dict mapping node_id -> sorted list of neighbor_ids (full graph)
        counts: GraphEdgeCounts with edge statistics
        shortcut_lattice_hops: List of lattice distances for shortcut edges
        lattice_adjacency: Lattice-only adjacency (for nav baseline)
        shortcut_scores: List of quality scores for each shortcut
    """
    rng = np.random.default_rng(seed)
    
    if grid_shape is not None:
        # Grid mode: reshape to (H*W, D)
        H, W = grid_shape
        N = H * W
        if embeddings.shape[:2] == (H, W):
            flat_embeddings = embeddings.reshape(N, -1)
        else:
            flat_embeddings = embeddings.reshape(N, -1)
    else:
        # Token/pooled mode: (N, D)
        flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1]) if embeddings.ndim > 2 else embeddings
        N = flat_embeddings.shape[0]
        H, W = N, 1  # 1D lattice
    
    # Initialize adjacency
    adjacency: Dict[int, set] = {i: set() for i in range(N)}
    local_edges = 0
    shortcut_edges = 0
    shortcut_lattice_hops: List[int] = []
    
    # Build local lattice edges
    for i in range(N):
        if grid_shape is not None:
            # 2D grid
            row, col = divmod(i, W)
            neighbors = []
            # 4-connectivity
            if row > 0:
                neighbors.append((row - 1) * W + col)
            if row < H - 1:
                neighbors.append((row + 1) * W + col)
            if col > 0:
                neighbors.append(row * W + (col - 1))
            if col < W - 1:
                neighbors.append(row * W + (col + 1))
            # 8-connectivity (diagonals)
            if config.local_connectivity == 8:
                if row > 0 and col > 0:
                    neighbors.append((row - 1) * W + (col - 1))
                if row > 0 and col < W - 1:
                    neighbors.append((row - 1) * W + (col + 1))
                if row < H - 1 and col > 0:
                    neighbors.append((row + 1) * W + (col - 1))
                if row < H - 1 and col < W - 1:
                    neighbors.append((row + 1) * W + (col + 1))
        else:
            # 1D lattice (adjacent tokens)
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < N - 1:
                neighbors.append(i + 1)
        
        for j in neighbors:
            if j not in adjacency[i]:
                adjacency[i].add(j)
                adjacency[j].add(i)
                local_edges += 1
    
    # Save lattice-only adjacency before adding shortcuts (for nav baseline)
    lattice_adjacency = {i: sorted(list(neighbors)) for i, neighbors in adjacency.items()}
    
    # Build kNN shortcut edges with sparsity controls
    # Compute pairwise cosine similarities
    norms = np.linalg.norm(flat_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = flat_embeddings / norms
    similarities = normalized @ normalized.T  # (N, N) cosine similarities
    
    # For each node, find k nearest neighbors in embedding space
    k = min(config.knn_k, N - 1)
    
    # First pass: collect all candidate shortcuts with scores
    # Score = cos_sim * log(1 + lattice_dist) - rewards long-range similar pairs
    knn_candidates: Dict[int, set] = {i: set() for i in range(N)}
    for i in range(N):
        sim_row = similarities[i].copy()
        sim_row[i] = -np.inf  # Exclude self
        top_k_indices = np.argsort(sim_row)[-k:][::-1]
        
        for idx in top_k_indices:
            j = int(idx)
            cos_sim = float(similarities[i, j])
            
            # Check lattice distance
            if grid_shape is not None:
                row_i, col_i = divmod(i, W)
                row_j, col_j = divmod(j, W)
                lattice_dist = abs(row_i - row_j) + abs(col_i - col_j)
            else:
                lattice_dist = abs(i - j)
            
            # Only consider if far enough on lattice
            if lattice_dist >= config.min_lattice_hops_for_shortcut:
                # Compute shortcut quality score
                score = cos_sim * np.log1p(lattice_dist)
                knn_candidates[i].add((j, lattice_dist, score))
    
    # Second pass: filter by mutual kNN if enabled, collect scored candidates
    candidate_edges: List[Tuple[int, int, int, float]] = []  # (i, j, lattice_dist, score)

    for i in range(N):
        for j, lattice_dist, score in knn_candidates[i]:
            if i < j:  # Avoid duplicates
                is_mutual = any(cand[0] == i for cand in knn_candidates[j])
                if config.mutual_knn_only and not is_mutual:
                    continue
                candidate_edges.append((i, j, lattice_dist, score))

    # Sort candidates by SCORE (highest first), with deterministic tie-break by (i, j)
    candidate_edges.sort(key=lambda x: (-x[3], x[0], x[1]))

    # Third pass: select shortcuts based on mode
    # Use shortcut_top_m_per_node for top_m_per_node mode, fall back to shortcut_budget_per_node
    top_m = config.shortcut_top_m_per_node
    if top_m is None and config.shortcut_budget_per_node is not None:
        top_m = config.shortcut_budget_per_node

    shortcuts_per_node: Dict[int, int] = {i: 0 for i in range(N)}
    selected_edges: List[Tuple[int, int, int, float]] = []

    if config.shortcut_select_mode == "threshold":
        # Threshold mode: keep edges with score >= threshold
        threshold = config.shortcut_score_threshold
        if threshold is None:
            threshold = 0.0  # Default: accept all if no threshold specified
        for i, j, lattice_dist, score in candidate_edges:
            if score >= threshold:
                selected_edges.append((i, j, lattice_dist, score))

    elif config.shortcut_select_mode == "target_nav_gain":
        # Target nav_gain mode: add shortcuts until nav_gain >= target (Goldilocks sparse)
        # This iteratively adds shortcuts and checks navigability
        target = config.target_nav_gain if config.target_nav_gain is not None else 0.1
        step = config.target_nav_gain_step

        # Compute lattice-only nav success first
        flat_emb = embeddings.reshape(N, -1) if embeddings.ndim > 2 else embeddings
        nav_rng = np.random.default_rng(seed)
        lattice_nav_success, _, _, _ = _compute_navigability(
            lattice_adjacency, flat_emb, config.n_queries, config.max_hops, nav_rng
        )

        # Iteratively add shortcuts in score order until nav_gain >= target
        test_adjacency = {i: set(neighbors) for i, neighbors in lattice_adjacency.items()}
        current_nav_gain = 0.0
        idx = 0

        while idx < len(candidate_edges) and current_nav_gain < target:
            # Add a batch of shortcuts
            batch_added = 0
            while batch_added < step and idx < len(candidate_edges):
                i, j, lattice_dist, score = candidate_edges[idx]
                idx += 1
                if j not in test_adjacency[i]:
                    test_adjacency[i].add(j)
                    test_adjacency[j].add(i)
                    selected_edges.append((i, j, lattice_dist, score))
                    batch_added += 1

            # Check nav_gain with current shortcuts
            test_adj_sorted = {k: sorted(list(v)) for k, v in test_adjacency.items()}
            nav_rng = np.random.default_rng(seed)  # Reset RNG for consistent comparison
            current_nav_success, _, _, _ = _compute_navigability(
                test_adj_sorted, flat_emb, config.n_queries, config.max_hops, nav_rng
            )
            current_nav_gain = current_nav_success - lattice_nav_success

    else:
        # top_m_per_node mode: for each node, keep top M scoring candidates
        # Process in score order (highest first), respecting per-node budget
        for i, j, lattice_dist, score in candidate_edges:
            if top_m is not None:
                if shortcuts_per_node[i] >= top_m or shortcuts_per_node[j] >= top_m:
                    continue
            selected_edges.append((i, j, lattice_dist, score))
            shortcuts_per_node[i] += 1
            shortcuts_per_node[j] += 1

    # Fourth pass: apply global safety ceiling
    # max_shortcuts_allowed = floor(max_shortcut_fraction * (local + shortcuts))
    # Solving: shortcuts <= fraction * (local + shortcuts)
    # => shortcuts * (1 - fraction) <= fraction * local
    # => shortcuts <= fraction * local / (1 - fraction)
    max_total_shortcuts = int(local_edges * config.max_shortcut_fraction / (1 - config.max_shortcut_fraction)) if config.max_shortcut_fraction < 1.0 else float('inf')

    # If we exceed the safety ceiling, keep highest-scoring shortcuts (already sorted by score)
    if len(selected_edges) > max_total_shortcuts:
        # Re-sort to ensure deterministic ordering by (score, i, j)
        selected_edges.sort(key=lambda x: (-x[3], x[0], x[1]))
        selected_edges = selected_edges[:int(max_total_shortcuts)]

    shortcut_scores: List[float] = []

    for i, j, lattice_dist, score in selected_edges:
        # Add edge
        if j not in adjacency[i]:
            adjacency[i].add(j)
            adjacency[j].add(i)
            shortcut_edges += 1
            shortcut_lattice_hops.append(lattice_dist)
            shortcut_scores.append(score)
    
    # Convert to sorted lists for determinism
    sorted_adjacency = {i: sorted(list(neighbors)) for i, neighbors in adjacency.items()}
    
    counts = GraphEdgeCounts(
        local_edges=local_edges,
        shortcut_edges=shortcut_edges,
        total_edges=local_edges + shortcut_edges,
    )
    
    return sorted_adjacency, counts, shortcut_lattice_hops, lattice_adjacency, shortcut_scores


def _compute_clustering_coefficient(adjacency: Dict[int, List[int]]) -> float:
    """Compute average local clustering coefficient.
    
    Clustering coefficient of node i = (# edges among neighbors) / (k*(k-1)/2)
    where k = degree of node i.
    """
    N = len(adjacency)
    if N == 0:
        return 0.0
    
    coefficients = []
    for i in range(N):
        neighbors = adjacency[i]
        k = len(neighbors)
        if k < 2:
            coefficients.append(0.0)
            continue
        
        # Count edges among neighbors
        neighbor_set = set(neighbors)
        edges_among_neighbors = 0
        for n in neighbors:
            for m in adjacency[n]:
                if m in neighbor_set and m > n:
                    edges_among_neighbors += 1
        
        max_edges = k * (k - 1) / 2
        coefficients.append(edges_among_neighbors / max_edges if max_edges > 0 else 0.0)
    
    return float(np.mean(coefficients))


def _estimate_avg_path_length(
    adjacency: Dict[int, List[int]], 
    n_sources: int, 
    rng: np.random.Generator
) -> float:
    """Estimate average shortest path length using sampled BFS.
    
    Args:
        adjacency: Graph adjacency list
        n_sources: Number of source nodes to sample
        rng: Random number generator
        
    Returns:
        Estimated average path length
    """
    N = len(adjacency)
    if N <= 1:
        return 0.0
    
    # Sample source nodes deterministically
    sources = rng.choice(N, size=min(n_sources, N), replace=False)
    
    all_distances = []
    for source in sources:
        # BFS from source
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        # Collect non-zero distances
        for dist in distances.values():
            if dist > 0:
                all_distances.append(dist)
    
    return float(np.mean(all_distances)) if all_distances else 0.0


def _compute_random_baseline(
    N: int, 
    mean_degree: float, 
    n_sources: int, 
    rng: np.random.Generator
) -> Tuple[float, float]:
    """Compute clustering and path length for Erdos-Renyi random graph.
    
    Args:
        N: Number of nodes
        mean_degree: Target mean degree
        n_sources: Number of sources for path length estimation
        rng: Random number generator
        
    Returns:
        (C_rand, L_rand) clustering coefficient and path length
    """
    if N <= 1:
        return 0.0, 0.0
    
    # Edge probability for target mean degree
    p = min(1.0, mean_degree / (N - 1))
    
    # Generate random graph
    adjacency: Dict[int, List[int]] = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Sort for determinism
    for i in adjacency:
        adjacency[i].sort()
    
    c_rand = _compute_clustering_coefficient(adjacency)
    l_rand = _estimate_avg_path_length(adjacency, n_sources, rng)

    return c_rand, l_rand


def _compute_configuration_model_baseline(
    degree_sequence: List[int],
    n_sources: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Compute clustering and path length for configuration model (degree-matched).

    Uses stub-matching to generate a random graph with the same degree sequence.
    More accurate sigma than ER when degree distribution is non-uniform.

    Args:
        degree_sequence: Degree of each node in the original graph
        n_sources: Number of sources for path length estimation
        rng: Random number generator

    Returns:
        (C_rand, L_rand) clustering coefficient and path length
    """
    N = len(degree_sequence)
    if N <= 1:
        return 0.0, 0.0

    # Create stub list: each node i appears degree_sequence[i] times
    stubs = []
    for i, deg in enumerate(degree_sequence):
        stubs.extend([i] * deg)

    if len(stubs) == 0:
        return 0.0, 0.0

    # Shuffle stubs and pair them
    rng.shuffle(stubs)

    # Build adjacency from paired stubs
    adjacency: Dict[int, List[int]] = {i: [] for i in range(N)}

    # Pair consecutive stubs (may create self-loops and multi-edges, we'll filter)
    for idx in range(0, len(stubs) - 1, 2):
        i, j = stubs[idx], stubs[idx + 1]
        if i != j:  # Skip self-loops
            adjacency[i].append(j)
            adjacency[j].append(i)

    # Remove multi-edges by converting to sets then back to sorted lists
    for i in adjacency:
        adjacency[i] = sorted(set(adjacency[i]))

    c_rand = _compute_clustering_coefficient(adjacency)
    l_rand = _estimate_avg_path_length(adjacency, n_sources, rng)

    return c_rand, l_rand


def _compute_navigability(
    adjacency: Dict[int, List[int]],
    embeddings: np.ndarray,
    n_queries: int,
    max_hops: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Compute bounded navigability via greedy routing.
    
    At each step, move to neighbor that minimizes embedding distance to target.
    
    Args:
        adjacency: Graph adjacency list
        embeddings: (N, D) node embeddings
        n_queries: Number of source-target pairs to test
        max_hops: Maximum hops allowed
        rng: Random number generator
        
    Returns:
        (success_rate, mean_hops, stretch, visited_nodes_mean)
    """
    N = len(adjacency)
    if N <= 1:
        return 1.0, 0.0, 1.0, 0.0
    
    # Sample source-target pairs
    sources = rng.choice(N, size=min(n_queries, N), replace=True)
    targets = rng.choice(N, size=min(n_queries, N), replace=True)
    
    successes = 0
    total_hops = []
    stretches = []
    visited_counts = []
    
    for source, target in zip(sources, targets):
        if source == target:
            successes += 1
            total_hops.append(0)
            stretches.append(1.0)
            visited_counts.append(1)
            continue
        
        # Greedy routing
        current = source
        visited = {current}
        hops = 0
        
        while current != target and hops < max_hops:
            neighbors = adjacency[current]
            if not neighbors:
                break
            
            # Choose neighbor closest to target in embedding space
            best_neighbor = None
            best_dist = float('inf')
            target_emb = embeddings[target]
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                dist = float(np.linalg.norm(embeddings[neighbor] - target_emb))
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = neighbor
            
            if best_neighbor is None:
                # All neighbors visited, backtrack not implemented
                break
            
            current = best_neighbor
            visited.add(current)
            hops += 1
        
        visited_counts.append(len(visited))
        
        if current == target:
            successes += 1
            total_hops.append(hops)
            # Estimate optimal path length (BFS)
            optimal = _bfs_distance(adjacency, source, target)
            if optimal > 0:
                stretches.append(hops / optimal)
            else:
                stretches.append(1.0)
    
    success_rate = successes / len(sources) if sources.size > 0 else 0.0
    mean_hops = float(np.mean(total_hops)) if total_hops else 0.0
    mean_stretch = float(np.mean(stretches)) if stretches else 1.0
    visited_mean = float(np.mean(visited_counts)) if visited_counts else 0.0
    
    return success_rate, mean_hops, mean_stretch, visited_mean


def _bfs_distance(adjacency: Dict[int, List[int]], source: int, target: int) -> int:
    """Compute shortest path distance via BFS."""
    if source == target:
        return 0
    
    distances = {source: 0}
    queue = deque([source])
    
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor == target:
                return distances[node] + 1
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return -1  # Unreachable


def compute_graph_metrics(
    adjacency: Dict[int, List[int]],
    embeddings: np.ndarray,
    shortcut_lattice_hops: List[int],
    edge_counts: GraphEdgeCounts,
    config: GraphSpecV1,
    seed: int,
    lattice_adjacency_input: Optional[Dict[int, List[int]]] = None,
    shortcut_scores: Optional[List[float]] = None,
) -> GraphMetrics:
    """Compute small-world and navigability metrics.
    
    Args:
        adjacency: Graph adjacency list (full graph with shortcuts)
        embeddings: (N, D) node embeddings
        shortcut_lattice_hops: Lattice distances for shortcut edges
        edge_counts: Edge count statistics
        config: Graph configuration
        seed: Random seed
        lattice_adjacency_input: Optional lattice-only adjacency for nav baseline
        shortcut_scores: Optional quality scores for each shortcut
        
    Returns:
        GraphMetrics with all computed values
    """
    rng = np.random.default_rng(seed)
    N = len(adjacency)
    
    # Mean degree
    total_degree = sum(len(neighbors) for neighbors in adjacency.values())
    mean_degree = total_degree / N if N > 0 else 0.0
    
    # Clustering coefficient
    clustering = _compute_clustering_coefficient(adjacency)

    # Average path length
    avg_path = _estimate_avg_path_length(adjacency, config.n_sources, rng)

    # Random baseline - choose method based on config
    if config.baseline_type == "configuration_model":
        # Degree-matched baseline (more accurate sigma)
        degree_sequence = [len(neighbors) for neighbors in adjacency.values()]
        c_rand, l_rand = _compute_configuration_model_baseline(degree_sequence, config.n_sources, rng)
    else:
        # ER baseline (fast, approximate)
        c_rand, l_rand = _compute_random_baseline(N, mean_degree, config.n_sources, rng)

    # Small-world sigma
    if c_rand > 0 and l_rand > 0:
        sigma = (clustering / c_rand) / (avg_path / l_rand) if avg_path > 0 else 0.0
    else:
        sigma = 0.0
    
    # Shortcut hop stats
    shortcut_fraction = edge_counts.shortcut_edges / edge_counts.total_edges if edge_counts.total_edges > 0 else 0.0
    if shortcut_lattice_hops:
        hop_array = np.array(shortcut_lattice_hops)
        shortcut_hop_mean = float(np.mean(hop_array))
        shortcut_hop_p50 = float(np.percentile(hop_array, 50))
        shortcut_hop_p90 = float(np.percentile(hop_array, 90))
    else:
        shortcut_hop_mean = 0.0
        shortcut_hop_p50 = 0.0
        shortcut_hop_p90 = 0.0
    
    # Shortcut score stats
    if shortcut_scores:
        score_array = np.array(shortcut_scores)
        shortcut_score_mean = float(np.mean(score_array))
        shortcut_score_min = float(np.min(score_array))
        shortcut_score_max = float(np.max(score_array))
        shortcut_score_p50 = float(np.percentile(score_array, 50))
        shortcut_score_p90 = float(np.percentile(score_array, 90))
    else:
        shortcut_score_mean = 0.0
        shortcut_score_min = 0.0
        shortcut_score_max = 0.0
        shortcut_score_p50 = 0.0
        shortcut_score_p90 = 0.0
    
    # Navigability on full graph (with shortcuts)
    flat_embeddings = embeddings.reshape(N, -1) if embeddings.ndim > 2 else embeddings
    nav_success, nav_hops, nav_stretch, nav_visited = _compute_navigability(
        adjacency, flat_embeddings, config.n_queries, config.max_hops, rng
    )
    
    # Navigability on lattice-only graph (without shortcuts)
    # Build lattice-only adjacency by filtering
    lattice_adjacency: Dict[int, List[int]] = {}
    if lattice_adjacency_input is not None:
        lattice_adjacency = lattice_adjacency_input
    else:
        # Fallback: reconstruct by keeping only local edges (degree <= local_connectivity)
        # This is approximate; ideally pass in explicitly
        lattice_adjacency = adjacency  # Fallback to full graph
    
    nav_success_lattice, _, _, _ = _compute_navigability(
        lattice_adjacency, flat_embeddings, config.n_queries, config.max_hops, rng
    )
    
    nav_gain = nav_success - nav_success_lattice
    
    return GraphMetrics(
        clustering_coefficient=clustering,
        avg_path_length=avg_path,
        c_rand=c_rand,
        l_rand=l_rand,
        sigma=sigma,
        shortcut_fraction=shortcut_fraction,
        shortcut_lattice_hop_mean=shortcut_hop_mean,
        shortcut_lattice_hop_p50=shortcut_hop_p50,
        shortcut_lattice_hop_p90=shortcut_hop_p90,
        shortcut_score_mean=shortcut_score_mean,
        shortcut_score_min=shortcut_score_min,
        shortcut_score_max=shortcut_score_max,
        shortcut_score_p50=shortcut_score_p50,
        shortcut_score_p90=shortcut_score_p90,
        nav_success_rate=nav_success,
        nav_mean_hops=nav_hops,
        nav_stretch=nav_stretch,
        nav_visited_nodes_mean=nav_visited,
        nav_success_lattice=nav_success_lattice,
        nav_gain=nav_gain,
    )


def graph_summary_from_embeddings(
    embeddings: np.ndarray,
    grid_shape: Optional[Tuple[int, int]],
    config: GraphSpecV1,
    seed: int,
) -> GraphSummaryV1:
    """Build graph summary from embeddings.
    
    Args:
        embeddings: (H, W, D) for grid mode or (N, D) for token mode
        grid_shape: (H, W) if grid mode, None for token mode
        config: Graph construction configuration
        seed: Random seed
        
    Returns:
        GraphSummaryV1 with all metrics
    """
    start_time = time.perf_counter()
    
    # Build graph (returns full adj, counts, shortcut hops, lattice-only adj, scores)
    adjacency, edge_counts, shortcut_hops, lattice_adjacency, shortcut_scores = build_small_world_graph(
        embeddings, grid_shape, config, seed
    )
    
    # Compute metrics (pass lattice adjacency for nav baseline, scores for quality stats)
    metrics = compute_graph_metrics(
        adjacency, embeddings, shortcut_hops, edge_counts, config, seed,
        lattice_adjacency_input=lattice_adjacency,
        shortcut_scores=shortcut_scores,
    )
    
    compute_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Determine node mode
    if grid_shape is not None:
        node_mode = "grid"
    elif embeddings.ndim == 2:
        node_mode = "tokens"
    else:
        node_mode = "pooled"
    
    N = len(adjacency)
    total_degree = sum(len(neighbors) for neighbors in adjacency.values())
    mean_degree = total_degree / N if N > 0 else 0.0
    
    # Build summary dict for hashing (include selection mode fields)
    summary_data = {
        "graph_spec_id": config.spec_id,
        "node_mode": node_mode,
        "node_count": N,
        "mean_degree": round(mean_degree, 6),
        "local_edge_count": edge_counts.local_edges,
        "shortcut_edge_count": edge_counts.shortcut_edges,
        "clustering_coefficient": round(metrics.clustering_coefficient, 6),
        "avg_path_length": round(metrics.avg_path_length, 6),
        "c_rand": round(metrics.c_rand, 6),
        "l_rand": round(metrics.l_rand, 6),
        "sigma": round(metrics.sigma, 6),
        "shortcut_score_mode": config.shortcut_score_mode,
        "shortcut_select_mode": config.shortcut_select_mode,
        "shortcut_score_threshold_used": config.shortcut_score_threshold if config.shortcut_select_mode == "threshold" else None,
        "shortcut_fraction": round(metrics.shortcut_fraction, 6),
        "shortcut_lattice_hop_mean": round(metrics.shortcut_lattice_hop_mean, 6),
        "shortcut_lattice_hop_p50": round(metrics.shortcut_lattice_hop_p50, 6),
        "shortcut_lattice_hop_p90": round(metrics.shortcut_lattice_hop_p90, 6),
        "shortcut_score_mean": round(metrics.shortcut_score_mean, 6),
        "shortcut_score_min": round(metrics.shortcut_score_min, 6),
        "shortcut_score_max": round(metrics.shortcut_score_max, 6),
        "shortcut_score_p50": round(metrics.shortcut_score_p50, 6),
        "shortcut_score_p90": round(metrics.shortcut_score_p90, 6),
        "nav_success_rate": round(metrics.nav_success_rate, 6),
        "nav_mean_hops": round(metrics.nav_mean_hops, 6),
        "nav_stretch": round(metrics.nav_stretch, 6),
        "nav_visited_nodes_mean": round(metrics.nav_visited_nodes_mean, 6),
        "nav_success_lattice": round(metrics.nav_success_lattice, 6),
        "nav_gain": round(metrics.nav_gain, 6),
        "baseline_type": config.baseline_type,
    }
    summary_sha = sha256_json(summary_data)

    return GraphSummaryV1(
        graph_spec_id=config.spec_id,
        graph_spec_sha=config.sha256(),
        summary_sha=summary_sha,
        node_mode=node_mode,
        node_count=N,
        mean_degree=mean_degree,
        local_edge_count=edge_counts.local_edges,
        shortcut_edge_count=edge_counts.shortcut_edges,
        clustering_coefficient=metrics.clustering_coefficient,
        avg_path_length=metrics.avg_path_length,
        c_rand=metrics.c_rand,
        l_rand=metrics.l_rand,
        sigma=metrics.sigma,
        baseline_type=config.baseline_type,
        shortcut_score_mode=config.shortcut_score_mode,
        shortcut_select_mode=config.shortcut_select_mode,
        shortcut_score_threshold_used=config.shortcut_score_threshold if config.shortcut_select_mode == "threshold" else None,
        shortcut_fraction=metrics.shortcut_fraction,
        shortcut_lattice_hop_mean=metrics.shortcut_lattice_hop_mean,
        shortcut_lattice_hop_p50=metrics.shortcut_lattice_hop_p50,
        shortcut_lattice_hop_p90=metrics.shortcut_lattice_hop_p90,
        shortcut_score_mean=metrics.shortcut_score_mean,
        shortcut_score_min=metrics.shortcut_score_min,
        shortcut_score_max=metrics.shortcut_score_max,
        shortcut_score_p50=metrics.shortcut_score_p50,
        shortcut_score_p90=metrics.shortcut_score_p90,
        nav_success_rate=metrics.nav_success_rate,
        nav_mean_hops=metrics.nav_mean_hops,
        nav_stretch=metrics.nav_stretch,
        nav_visited_nodes_mean=metrics.nav_visited_nodes_mean,
        nav_success_lattice=metrics.nav_success_lattice,
        nav_gain=metrics.nav_gain,
        compute_time_ms=compute_time_ms,
    )


def graph_summary_from_repr_tokens(
    repr_tokens: Dict[str, Any],
    config: GraphSpecV1,
    seed: int,
) -> Optional[GraphSummaryV1]:
    """Build graph summary from repr_tokens payload.
    
    Prioritizes geometry_bev with bev_grid, falls back to token sequences.
    
    Args:
        repr_tokens: Dict with repr_name -> {tokens, metadata, ...}
        config: Graph configuration
        seed: Random seed
        
    Returns:
        GraphSummaryV1 or None if no suitable embeddings found
    """
    # Priority 1: geometry_bev with bev_grid
    if "geometry_bev" in repr_tokens:
        bev_data = repr_tokens["geometry_bev"]
        metadata = bev_data.get("metadata", {})
        
        if "bev_grid" in metadata:
            # Has grid data - use it
            bev_grid = np.array(metadata["bev_grid"])
            if bev_grid.ndim == 4:
                # (T, H, W, C) - use last timestep or average
                embeddings = bev_grid[-1]  # (H, W, C)
                grid_shape = (embeddings.shape[0], embeddings.shape[1])
                return graph_summary_from_embeddings(embeddings, grid_shape, config, seed)
        
        # Fallback: use tokens
        if "tokens" in bev_data or "features" in bev_data:
            tokens = np.array(bev_data.get("tokens") or bev_data.get("features"))
            return graph_summary_from_embeddings(tokens, None, config, seed)
    
    # Priority 2: Any available token sequence
    for repr_name, repr_data in repr_tokens.items():
        if "tokens" in repr_data or "features" in repr_data:
            tokens = np.array(repr_data.get("tokens") or repr_data.get("features"))
            if tokens.size > 0:
                return graph_summary_from_embeddings(tokens, None, config, seed)
    
    return None


__all__ = [
    "GraphEdgeCounts",
    "GraphMetrics",
    "build_small_world_graph",
    "compute_graph_metrics",
    "graph_summary_from_embeddings",
    "graph_summary_from_repr_tokens",
]
