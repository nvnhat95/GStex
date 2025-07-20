import torch
import numpy as np
import logging
from tqdm import tqdm
from pytorch3d.ops import ball_query
from nerfstudio.utils.rotations import quaternion_to_matrix
from dataclasses import dataclass

# Set up logger
logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """Setup logger with appropriate formatting and level."""
    if not logger.handlers:  # Avoid adding multiple handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


@dataclass
class GaussianPrimitives:
    xyz: torch.Tensor        # (N, 3)
    scaling: torch.Tensor    # (N, 3)
    rotation: torch.Tensor   # (N, 4)
    opacity: torch.Tensor    # (N, 1)
    features_dc: torch.Tensor# (N, 3, 1)
    features_rest: torch.Tensor# (N, 3, SH_coeffs - 1)
    
    @classmethod
    def from_tensor(cls, gaussian_tensor: torch.Tensor):
        """
        Decompose a tensor of shape (N, D) into GaussianPrimitives.
        """
        xyz = gaussian_tensor[:, :3]
        scaling = gaussian_tensor[:, 3:6]
        rotation = gaussian_tensor[:, 6:10]
        opacity = gaussian_tensor[:, 10:11]
        features_dc = gaussian_tensor[:, 11:14]
        features_rest = gaussian_tensor[:, 14:]
        N = xyz.shape[0]
        return cls(
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            features_dc=features_dc.reshape(N, 3, 1),
            features_rest=features_rest.reshape(N, 3, -1)
        )
    
    def to_tensor(self) -> torch.Tensor:
        """Convert back to a single tensor."""
        N = self.xyz.shape[0]
        return torch.cat([
            self.xyz,
            self.scaling,
            self.rotation,
            self.opacity,
            self.features_dc.squeeze(-1),  # reshape from (N, 3, 1) to (N, 3)
            self.features_rest.reshape(N, -1), # (N, 3*(SH_coeffs - 1))
        ], dim=1)


def make_covariance_3d(scale, quat):
    """
    Build a 3×3 covariance matrix from scale (std-devs) and unit quaternion.
    scale: (..., 3), quat: (..., 4) → cov: (..., 3, 3)
    """
    if scale.shape[1] == 2: # 3DGS compatibility
        scale = torch.cat([scale, torch.zeros_like(scale[..., :1])], dim=-1) # (..., 3)

    R = quaternion_to_matrix(quat)          # (..., 3, 3)
    D = torch.diag_embed(scale**2)          # (..., 3, 3)
    return R @ D @ R.transpose(-1, -2)      # (..., 3, 3)


def sqrtm_psd_3x3(mat, eps=1e-12):
    """
    Matrix square‐root of a batch of symmetric PSD 3×3 matrices.
    mat: (..., 3, 3) → sqrt_mat: (..., 3, 3)
    """
    # Eigen‐decompose: v @ diag(e) @ v.T
    e, v = torch.linalg.eigh(mat) # e: (..., 3), v: (..., 3, 3)
    # clamp for numerical stability
    e_clamped = torch.clamp(e, min=eps)
    sqrt_e = torch.sqrt(e_clamped) # (..., 3)
    return (v * sqrt_e.unsqueeze(-2)) @ v.transpose(-1, -2) # (..., 3, 3)


def wasserstein_3d_gaussians(mu1, scale1, quat1, mu2, scale2, quat2):
    """
    Squared 2‐Wasserstein distance between two 3D Gaussian splats.
    
    Args:
      mu1, mu2     : (..., 3)
      scale1, scale2 : (..., 3) log-scales for first 2 dims, last dim is tied to mean of first 2
      quat1, quat2 : (..., 4) unit quaternions
    Returns:
      W2^2: tensor of shape (...)
    """
    # Convert log-scales to positive std-devs following GS convention
    scale1_exp = torch.zeros_like(scale1)
    scale2_exp = torch.zeros_like(scale2)
    
    # First two dims are exp(log_scale) with min clamp
    scale1_exp[..., :-1] = torch.clamp(torch.exp(scale1[..., :-1]), min=1e-9)
    scale2_exp[..., :-1] = torch.clamp(torch.exp(scale2[..., :-1]), min=1e-9)
    
    # Last dim is tied to mean of first two dims
    scale1_exp[..., -1] = 1e-5 * torch.mean(scale1_exp[..., :-1], dim=-1)
    scale2_exp[..., -1] = 1e-5 * torch.mean(scale2_exp[..., :-1], dim=-1)
    
    # Build covariance matrices
    cov1 = make_covariance_3d(scale1_exp, quat1)   # (..., 3, 3)
    cov2 = make_covariance_3d(scale2_exp, quat2)   # (..., 3, 3)

    # Squared distance between means
    mean_term = torch.sum((mu1 - mu2)**2, dim=-1)  # (...)

    # Compute the cross term sqrtm(cov2^(1/2) cov1 cov2^(1/2))
    sqrt_cov2 = sqrtm_psd_3x3(cov2)                # (..., 3, 3)
    inner = sqrt_cov2 @ cov1 @ sqrt_cov2.transpose(-1, -2)
    sqrt_inner = sqrtm_psd_3x3(inner)              # (..., 3, 3)

    # Trace term
    trace_term = torch.diagonal(cov1 + cov2 - 2*sqrt_inner, dim1=-2, dim2=-1).sum(-1)  # (...)

    return mean_term + trace_term


class BallQueryDBSCAN:
    def __init__(self, eps, min_pts, search_radius_multiplier=2.0):
        """
        Initialize ball query-based DBSCAN clustering algorithm for Gaussian Splats.
        
        Args:
            eps: Maximum wasserstein distance for points to be considered neighbors
            min_pts: Minimum number of points required to form a cluster
            search_radius_multiplier: Multiplier for initial ball_query radius 
                                    (since Euclidean != Wasserstein distance)
        
        Logging:
            The class uses the module logger for debugging. To enable debug output:
            - Call setup_logger(logging.DEBUG) before using the class
            - Debug logs include neighbor finding details, cluster expansion info, etc.
            - Info logs show clustering progress and final results
            - Warning logs indicate potential issues (e.g., too many candidates in ball query)
        """
        self.eps = eps
        self.min_pts = min_pts
        self.search_radius_multiplier = search_radius_multiplier

    def plot_k_distance_graph(self, gaussians, k=None, max_radius=10.0, n_samples=None):
        """
        Plot k-distance graph to help estimate a good eps value.
        
        Args:
            gaussians: GaussianPrimitives object
            k: Number of neighbors to consider (defaults to min_pts)
            max_radius: Maximum radius for ball query
            n_samples: Number of random points to sample (None means use all points)
            
        Returns:
            Suggested eps value based on elbow detection
        """
        import matplotlib.pyplot as plt
        from pytorch3d.ops import knn_points
        import numpy as np
        from scipy.signal import savgol_filter
        
        if k is None:
            k = self.min_pts
            
        points = gaussians.xyz
        n_total = points.shape[0]
        
        # Optionally sample a subset of points
        if n_samples is not None and n_samples < n_total:
            indices = torch.randperm(n_total)[:n_samples]
            points_subset = points[indices]
            scaling_subset = gaussians.scaling[indices]
            rotation_subset = gaussians.rotation[indices]
            points_all = points
            scaling_all = gaussians.scaling
            rotation_all = gaussians.rotation
        else:
            points_subset = points
            scaling_subset = gaussians.scaling
            rotation_subset = gaussians.rotation
            points_all = points
            scaling_all = gaussians.scaling
            rotation_all = gaussians.rotation
            
        n_points = points_subset.shape[0]
        logger.info(f"Computing k-distance graph using {n_points} points (k={k})")
        
        # First use kNN to get approximate neighbors (Euclidean distance)
        k_larger = min(k * 4, n_total)  # Get more neighbors than needed since Wasserstein != Euclidean
        knn_dists, knn_idx, _ = knn_points(
            points_subset.unsqueeze(0),   # (1, N, 3)
            points_all.unsqueeze(0),      # (1, M, 3)
            K=k_larger,
            return_sorted=True
        )
        
        # Compute Wasserstein distances to these candidates
        k_distances = []
        for i in range(n_points):
            query_mu = points_subset[i:i+1]      # (1, 3)
            query_scale = scaling_subset[i:i+1]  # (1, 3)
            query_quat = rotation_subset[i:i+1]  # (1, 4)
            
            # Get candidates from kNN
            candidates = knn_idx[0, i]  # (K,)
            cand_mu = points_all[candidates]       # (K, 3)
            cand_scale = scaling_all[candidates]   # (K, 3)
            cand_quat = rotation_all[candidates]   # (K, 4)
            
            # Compute Wasserstein distances
            w_dists = wasserstein_3d_gaussians(
                query_mu, query_scale, query_quat,
                cand_mu, cand_scale, cand_quat
            )  # (K,)
            
            # Sort and get the k-th distance
            k_dist = torch.sort(w_dists)[0][k-1].item()
            k_distances.append(k_dist)
            
        # Sort distances for the plot
        k_distances = np.array(sorted(k_distances))
        points = np.arange(len(k_distances))
        
        # Smooth the curve for better elbow detection
        window = min(21, len(k_distances) // 5)
        if window % 2 == 0:
            window += 1
        y_smooth = savgol_filter(k_distances, window, 3)
        
        # Compute approximate derivative
        dy = np.gradient(y_smooth)
        d2y = np.gradient(dy)
        
        # Find elbow point using maximum curvature
        curvature = np.abs(d2y) / (1 + dy**2)**(3/2)
        elbow_idx = np.argmax(curvature[10:-10]) + 10  # Avoid edges
        suggested_eps = k_distances[elbow_idx]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(points, k_distances, 'b-', alpha=0.5, label='Raw distances')
        plt.plot(points, y_smooth, 'r-', label='Smoothed')
        plt.axvline(x=elbow_idx, color='g', linestyle='--', label='Suggested elbow')
        plt.axhline(y=suggested_eps, color='g', linestyle='--')
        
        plt.xlabel('Points (sorted by distance)')
        plt.ylabel(f'{k}-th nearest neighbor Wasserstein distance')
        plt.title(f'K-Distance Graph (k={k})')
        plt.legend()
        plt.grid(True)
        
        # Add text annotation for suggested eps
        plt.text(0.02, 0.98, f'Suggested eps: {suggested_eps:.4f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        
        return suggested_eps

    def find_neighbors(self, gaussians, point_idx):
        """
        Find neighbors of a single point using PyTorch3D's ball_query and wasserstein distance.
        
        Args:
            gaussians: GaussianPrimitives object
            point_idx: Index of the query point
            
        Returns:
            neighbors: Tensor of neighbor indices within wasserstein distance eps
        """
        points = gaussians.xyz
        device = points.device
        
        logger.debug(f"Finding neighbors for point {point_idx} at position {points[point_idx]}")
        
        # Query point
        query_point = points[point_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        all_points = points.unsqueeze(0)  # (1, N, 3)
        
        # Use ball_query to get initial candidates within Euclidean radius
        # We use a larger radius since wasserstein distance can be different from Euclidean
        search_radius = self.eps * self.search_radius_multiplier
        
        # Set K to a reasonable upper bound (or all points if dataset is small)
        max_neighbors = min(points.shape[0], 1000)
        
        try:
            _, neighbor_indices, _ = ball_query(
                query_point, all_points,
                radius=search_radius,
                K=max_neighbors,
                return_nn=False
            ) # (1, 1, K)
            
            # Extract valid neighbors (ball_query pads with -1)
            candidates = neighbor_indices[0, 0]  # (K,)
            valid_mask = candidates >= 0
            candidate_indices = candidates[valid_mask]
            
            # Debug: Check if we're getting too many candidates
            if len(candidate_indices) > points.shape[0] * 0.8:  # More than 80% of points
                logger.warning(f"Ball query found {len(candidate_indices)}/{points.shape[0]} candidates (search_radius={search_radius:.3f}). Reducing radius.")
                # Reduce to smaller radius for this query
                reduced_radius = search_radius * 0.5  # Use smaller radius
                _, neighbor_indices, _ = ball_query(
                    query_point, all_points,
                    radius=reduced_radius,
                    K=max_neighbors,
                    return_nn=False
                )
                candidates = neighbor_indices[0, 0]
                valid_mask = candidates >= 0
                candidate_indices = candidates[valid_mask]
                logger.debug(f"Reduced search found {len(candidate_indices)} candidates")
            
        except Exception as e:
            # Fallback: if ball_query fails, use all points as candidates
            logger.warning(f"Ball query failed: {e}. Using all points as candidates.")
            candidate_indices = torch.arange(points.shape[0], device=device)
        
        # Remove self from candidates
        candidate_indices = candidate_indices[candidate_indices != point_idx]
        
        if len(candidate_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=device)
        
        # Compute wasserstein distances for all candidates
        query_mu = gaussians.xyz[point_idx].unsqueeze(0)  # (1, 3)
        query_scale = gaussians.scaling[point_idx].unsqueeze(0)  # (1, 3)
        query_quat = gaussians.rotation[point_idx].unsqueeze(0)  # (1, 4)
        
        candidate_mu = gaussians.xyz[candidate_indices]  # (M, 3)
        candidate_scale = gaussians.scaling[candidate_indices]  # (M, 3)
        candidate_quat = gaussians.rotation[candidate_indices]  # (M, 4)
        
        # Compute wasserstein distances
        wasserstein_dists = wasserstein_3d_gaussians(
            query_mu, query_scale, query_quat,
            candidate_mu, candidate_scale, candidate_quat
        )  # (M,)
        
        logger.debug(f"Computed {len(wasserstein_dists)} wasserstein distances, "
                    f"min: {wasserstein_dists.min():.4f}, max: {wasserstein_dists.max():.4f}, "
                    f"mean: {wasserstein_dists.mean():.4f}")
        
        # Filter neighbors based on wasserstein distance threshold
        valid_neighbors_mask = wasserstein_dists <= self.eps
        neighbors = candidate_indices[valid_neighbors_mask]
        
        logger.debug(f"Found {len(neighbors)} valid neighbors within eps={self.eps}")
        
        return neighbors

    def fit(self, gaussians):
        """
        Perform DBSCAN clustering on Gaussian Splats using ball query for neighbor search.
        
        Args:
            gaussians: GaussianPrimitives object
        
        Returns:
            labels: Tensor of shape (n_points,) containing cluster assignments
                   -1 indicates noise points
                   >= 0 indicates cluster assignments
        """
        points = gaussians.xyz
        n_points = points.shape[0]
        device = points.device
        
        # Initialize labels: -2 = unclassified, -1 = noise, >= 0 = cluster ID
        labels = torch.full((n_points,), -2, dtype=torch.long, device=device)
        cluster_id = 0

        logger.info(f"Starting ball query-based DBSCAN clustering on {n_points} Gaussian splats...")
        logger.info(f"Parameters: eps={self.eps}, min_pts={self.min_pts}")

        # Iterate over all points
        for point_idx in tqdm(range(n_points), desc="DBSCAN clustering"):
            if labels[point_idx] != -2:  # Skip if already classified
                continue
                
            # Find neighbors using ball query + wasserstein distance
            neighbor_indices = self.find_neighbors(gaussians, point_idx)
            
            # Check if point is a core point
            if len(neighbor_indices) < self.min_pts:
                labels[point_idx] = -1  # Mark as noise
                logger.debug(f"Point {point_idx} marked as noise (only {len(neighbor_indices)} neighbors)")
                continue
                
            # Start new cluster
            cluster_id += 1
            labels[point_idx] = cluster_id
            logger.debug(f"Starting new cluster {cluster_id} at point {point_idx} with {len(neighbor_indices)} initial neighbors")
            
            # Convert to list for dynamic expansion
            neighbor_list = neighbor_indices.tolist()
            
            # Expand cluster using neighbors
            i = 0
            while i < len(neighbor_list):
                neighbor_idx = neighbor_list[i]
                
                # A point should be processed if it is unclassified (-2) or was marked as noise (-1)
                if labels[neighbor_idx] == -2 or labels[neighbor_idx] == -1:
                    # Assign to current cluster
                    labels[neighbor_idx] = cluster_id
                    
                    # Check if this neighbor is a core point to expand the cluster
                    neighbor_neighbors = self.find_neighbors(gaussians, neighbor_idx)
                    
                    # If the neighbor has enough neighbors, it is a core point.
                    if len(neighbor_neighbors) >= self.min_pts:
                        # Add new neighbors to expansion list. This point is a core point.
                        new_neighbors_count = 0
                        for nn in neighbor_neighbors:
                            nn_item = nn.item()
                            # Add neighbors that are unclassified or noise to the queue.
                            # Checking `not in neighbor_list` avoids duplicates in the queue.
                            if (labels[nn_item] == -2 or labels[nn_item] == -1) and (nn_item not in neighbor_list):
                                neighbor_list.append(nn_item)
                                new_neighbors_count += 1
                        
                        if new_neighbors_count > 0:
                            logger.debug(f"Expanded cluster {cluster_id} from core point {neighbor_idx}, "
                                       f"added {new_neighbors_count} new neighbors (total: {len(neighbor_list)})")
                
                i += 1
        
        return labels

    def analyze_clusters(self, labels, ground_truth_labels=None):
        """
        Analyze clustering results and optionally compare with ground truth.
        
        Args:
            labels: Cluster labels from fit()
            ground_truth_labels: Optional ground truth cluster labels for comparison
            
        Returns:
            dict: Analysis results
        """
        logger.debug("Starting cluster analysis...")
        
        unique_labels = torch.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = (labels == -1).sum().item()
        n_total = len(labels)
        
        logger.debug(f"Found {n_clusters} clusters and {n_noise} noise points out of {n_total} total points")
        
        cluster_sizes = []
        for cluster_id in unique_labels[unique_labels >= 0]:
            size = (labels == cluster_id).sum().item()
            cluster_sizes.append(size)
        
        results = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'n_total_points': n_total,
            'noise_ratio': n_noise / n_total,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0
        }
        
        # Compare with ground truth if provided
        if ground_truth_labels is not None:
            gt_results = self._compare_with_ground_truth(labels, ground_truth_labels)
            results.update(gt_results)
        
        return results
    
    def _compare_with_ground_truth(self, predicted_labels, ground_truth_labels):
        """
        Compare clustering results with ground truth labels.
        
        Args:
            predicted_labels: Cluster labels from DBSCAN
            ground_truth_labels: Ground truth cluster labels
            
        Returns:
            dict: Comparison metrics
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
        
        # Convert to numpy for sklearn metrics
        pred_np = predicted_labels.cpu().numpy()
        gt_np = ground_truth_labels.cpu().numpy()
        
        # Calculate clustering metrics
        ari = adjusted_rand_score(gt_np, pred_np)
        nmi = normalized_mutual_info_score(gt_np, pred_np)
        homogeneity = homogeneity_score(gt_np, pred_np)
        completeness = completeness_score(gt_np, pred_np)
        v_measure = v_measure_score(gt_np, pred_np)
        
        # Analyze cluster correspondence
        gt_unique = torch.unique(ground_truth_labels)
        pred_unique = torch.unique(predicted_labels[predicted_labels >= 0])
        
        # Count points correctly clustered vs misclassified
        correct_clustered = 0
        misclassified = 0
        
        for gt_cluster in gt_unique:
            gt_mask = ground_truth_labels == gt_cluster
            gt_cluster_points = predicted_labels[gt_mask]
            
            # Find the most common predicted cluster for this ground truth cluster
            if len(gt_cluster_points) > 0:
                most_common_pred = torch.mode(gt_cluster_points)[0]
                if most_common_pred >= 0:  # Not noise
                    correct_in_cluster = (gt_cluster_points == most_common_pred).sum().item()
                    correct_clustered += correct_in_cluster
                    misclassified += len(gt_cluster_points) - correct_in_cluster
                else:
                    misclassified += len(gt_cluster_points)  # All marked as noise
        
        total_points = len(ground_truth_labels)
        accuracy = correct_clustered / total_points if total_points > 0 else 0
        
        comparison_results = {
            'adjusted_rand_score': ari,
            'normalized_mutual_info_score': nmi,
            'homogeneity_score': homogeneity,
            'completeness_score': completeness,
            'v_measure_score': v_measure,
            'clustering_accuracy': accuracy,
            'correctly_clustered_points': correct_clustered,
            'misclassified_points': misclassified,
            'n_ground_truth_clusters': len(gt_unique),
            'n_predicted_clusters': len(pred_unique)
        }
        
        logger.info(f"Ground truth comparison:")
        logger.info(f"  Adjusted Rand Score: {ari:.4f}")
        logger.info(f"  Normalized Mutual Info: {nmi:.4f}")
        logger.info(f"  Clustering Accuracy: {accuracy:.4f}")
        logger.info(f"  Correctly clustered: {correct_clustered}/{total_points}")
        
        return comparison_results


# Example usage and testing
if __name__ == "__main__":
    # Setup logger - change to logging.DEBUG for detailed debugging info
    setup_logger(level=logging.DEBUG)  # Use logging.DEBUG for more verbose output
    
    # Example with synthetic data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create some synthetic Gaussian primitives with more separated clusters
    n_points = 1000
    n_clusters = 5
    cluster_size = n_points // n_clusters
    
    # Create separated cluster centers
    cluster_centers = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0], 
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 5.0, 0.0]
    ], device=device)
    
    xyz_list = []
    ground_truth_labels = []
    
    for i in range(n_clusters):
        # Generate points around each cluster center
        cluster_points = cluster_centers[i] + torch.randn(cluster_size, 3, device=device) * 0.5
        xyz_list.append(cluster_points)
        ground_truth_labels.extend([i] * cluster_size)
    
    # Add remaining points as noise
    remaining = n_points - n_clusters * cluster_size
    if remaining > 0:
        noise_points = torch.randn(remaining, 3, device=device) * 10.0  # Spread out noise
        xyz_list.append(noise_points)
        ground_truth_labels.extend([-1] * remaining)  # -1 for noise points
    
    xyz = torch.cat(xyz_list, dim=0)
    ground_truth_labels = torch.tensor(ground_truth_labels, device=device)
    
    # Generate other properties
    scaling = torch.abs(torch.randn(n_points, 3, device=device)) * 0.1
    rotation = torch.randn(n_points, 4, device=device)
    rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)  # Normalize quaternions
    opacity = torch.rand(n_points, 1, device=device)
    features_dc = torch.randn(n_points, 3, 1, device=device)
    features_rest = torch.randn(n_points, 3, 15, device=device)  # SH degree 3
    
    gaussians = GaussianPrimitives(
        xyz=xyz,
        scaling=scaling,
        rotation=rotation,
        opacity=opacity,
        features_dc=features_dc,
        features_rest=features_rest
    )
    
    # Print some stats about the data
    print(f"Generated {n_points} points")
    print(f"Point cloud bounds: {xyz.min(dim=0)[0]} to {xyz.max(dim=0)[0]}")
    print(f"Mean pairwise distance: {torch.cdist(xyz[:100], xyz[:100]).mean():.3f}")
    
    # First create a DBSCAN instance with dummy eps (will be updated)
    min_pts = 10
    dbscan = BallQueryDBSCAN(eps=1.0, min_pts=min_pts, search_radius_multiplier=1.5)
    
    # Plot k-distance graph and get suggested eps
    suggested_eps = dbscan.plot_k_distance_graph(gaussians, k=min_pts, n_samples=500)
    print(f"\nSuggested eps from k-distance graph: {suggested_eps:.4f}")
    
    # Update eps and run DBSCAN
    dbscan.eps = suggested_eps
    print(f"\nRunning DBSCAN with eps={dbscan.eps}, min_pts={min_pts}")
    
    labels = dbscan.fit(gaussians)
    
    # Analyze results with ground truth comparison
    results = dbscan.analyze_clusters(labels, ground_truth_labels)
    print("\nClustering Results:")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Number of noise points: {results['n_noise_points']}")
    print(f"Noise ratio: {results['noise_ratio']:.2%}")
    if results['cluster_sizes']:
        print(f"Average cluster size: {results['avg_cluster_size']:.1f}")
        print(f"Largest cluster: {results['largest_cluster_size']} points")
        print(f"Smallest cluster: {results['smallest_cluster_size']} points")
        print(f"Cluster sizes: {sorted(results['cluster_sizes'], reverse=True)}")
    
    # Print ground truth comparison results
    if 'adjusted_rand_score' in results:
        print("\nGround Truth Comparison:")
        print(f"Adjusted Rand Score: {results['adjusted_rand_score']:.4f}")
        print(f"Normalized Mutual Info: {results['normalized_mutual_info_score']:.4f}")
        print(f"Homogeneity Score: {results['homogeneity_score']:.4f}")
        print(f"Completeness Score: {results['completeness_score']:.4f}")
        print(f"V-Measure Score: {results['v_measure_score']:.4f}")
        print(f"Clustering Accuracy: {results['clustering_accuracy']:.4f}")
        print(f"Correctly clustered: {results['correctly_clustered_points']}/{results['n_total_points']}")
        print(f"Ground truth clusters: {results['n_ground_truth_clusters']}")
        print(f"Predicted clusters: {results['n_predicted_clusters']}") 