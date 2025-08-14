import math
import time
import torch
import numpy as np
from nerfstudio.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
from plyfile import PlyData
from dataclasses import dataclass
from tqdm import tqdm
from nerfstudio.cameras.cameras import Cameras
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import ball_query, knn_points


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


def load_ply(path, sh_degree=0, fix_init=False):
    """
    Load a PLY file and return the point cloud data.
    
    Args:
        path (str): Path to the PLY file
        sh_degree (int, optional): Spherical harmonics degree. Defaults to 0.
        fix_init (bool, optional): Blender uses False, COLMAP/DTU uses True.
    
    Returns:
        list: [xyz, features_dc, features_rest, opacity, scaling, rotations]
    """
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.tensor(xyz, dtype=torch.float).requires_grad_(True)
    features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
    opacity = torch.tensor(opacities, dtype=torch.float)
    scaling = torch.tensor(scales, dtype=torch.float)
    rotation = torch.tensor(rots, dtype=torch.float)

    if fix_init:
        new_xyz = torch.zeros_like(xyz)
        new_xyz[:,0] = xyz[:,0]
        new_xyz[:,1] = xyz[:,2]
        new_xyz[:,2] = -xyz[:,1]

        rotmats = quaternion_to_matrix(rotation)
        rotmats_fix = torch.zeros_like(rotmats)
        rotmats_fix[:,0,:] = rotmats[:,0,:]
        rotmats_fix[:,1,:] = rotmats[:,2,:]
        rotmats_fix[:,2,:] = -rotmats[:,1,:,]
        new_rotations = matrix_to_quaternion(rotmats_fix)
    else:
        new_xyz = xyz
        new_rotations = rotation

    gaussians = GaussianPrimitives(
        xyz=new_xyz,
        scaling=scaling,
        rotation=new_rotations,
        opacity=opacity,
        features_dc=features_dc,
        features_rest=features_rest
    )
    return gaussians


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
      scale1, scale2 : (..., 3) positive std-devs
      quat1, quat2 : (..., 4) unit quaternions
    Returns:
      W2^2: tensor of shape (...)
    """
    # Build covariance matrices
    cov1 = make_covariance_3d(scale1, quat1)   # (..., 3, 3)
    cov2 = make_covariance_3d(scale2, quat2)   # (..., 3, 3)

    # Squared distance between means
    mean_term = torch.sum((mu1 - mu2)**2, dim=-1)  # (...)

    # Compute the cross term sqrtm(cov2^(1/2) cov1 cov2^(1/2))
    sqrt_cov2 = sqrtm_psd_3x3(cov2)                # (..., 3, 3)
    inner = sqrt_cov2 @ cov1 @ sqrt_cov2.transpose(-1, -2)
    sqrt_inner = sqrtm_psd_3x3(inner)              # (..., 3, 3)

    # Trace term
    trace_term = torch.diagonal(cov1 + cov2 - 2*sqrt_inner, dim1=-2, dim2=-1).sum(-1)  # (...)

    return mean_term + trace_term


class DBSCAN:
    def __init__(self, eps, min_pts):
        """
        Initialize DBSCAN clustering algorithm.
        
        Args:
            eps: Maximum distance for points to be considered neighbors
            min_pts: Minimum number of points required to form a cluster
        """
        self.eps = eps
        self.min_pts = min_pts

    def set_level(self, points, cameras: Cameras, dist_ratio=0.95, levels=-1):
        all_dist = torch.tensor([])
        camera_centers = cameras.camera_to_worlds[..., 3][:, :3]  # Get camera centers from transform matrix
        for cam_center in camera_centers:
            dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
            dist_max = torch.quantile(dist, dist_ratio)
            dist_min = torch.quantile(dist, 1 - dist_ratio)
            new_dist = torch.tensor([dist_min, dist_max]).float()
            all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
            
    def octree_sample(self, data, init_pos, device="cuda"):
        """
        Inputs:
            data: (N, 3)
            init_pos: (3,)
        """
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().to(device) # positions of voxel centers
        self._level = torch.empty(0).int().to(device) # level of voxel centers
        self.point_to_voxel_idx = torch.full((data.shape[0],), -1, dtype=torch.long, device=device) 
        
        # Keep track of unassigned points
        unassigned_mask = torch.ones(data.shape[0], dtype=torch.bool, device=device)
        
        for cur_level in range(self.levels):
            if not unassigned_mask.any():  # Exit early if all points are assigned
                break
            
            # Get current unassigned points
            unassigned_points = data[unassigned_mask] # (N, 3)
            unassigned_indices = torch.where(unassigned_mask)[0] # (N,)
            
            cur_size = self.voxel_size/(float(self.fork) ** cur_level) # smaller as level increases

            # Map points to voxel coordinates
            voxel_coords = torch.round((unassigned_points - init_pos) / cur_size) # (N, 3)
            unique_voxels, inverse_indices, counts = torch.unique(voxel_coords, dim=0, return_inverse=True, return_counts=True) # (V, 3), (N,), (V,)
            
            # Only assign points to voxels that have enough density
            dense_voxel_mask = counts >= self.min_points_per_voxel # (V,)
            points_in_dense_voxels = torch.isin(inverse_indices, torch.where(dense_voxel_mask)[0]) # (N,)
            
            if points_in_dense_voxels.any():
                # Get indices of points that will be assigned in this level
                points_to_assign = unassigned_indices[points_in_dense_voxels] # (P,), P > V
                
                # The point index becomes the voxel index
                self.point_to_voxel_idx[points_to_assign] = points_to_assign # (P,)
                
                # Mark these points as assigned
                unassigned_mask[points_to_assign] = False
                
                # Add voxel centers and levels
                new_positions = data[points_to_assign]  # Use original point positions as voxel centers
                new_level = torch.ones(len(points_to_assign), dtype=torch.int, device=device) * cur_level
                
                # Append new voxel centers and levels
                self.positions = torch.cat((self.positions, new_positions), dim=0)
                self._level = torch.cat((self._level, new_level), dim=0)
        
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def voxelize(self, points):
        """Map points to voxel indices (as integer coordinates)."""
        return torch.floor(points / self.eps).int()

    def compute_voxel_keys(self, voxel_coords):
        """Create unique hash keys from voxel coordinates."""
        multipliers = torch.tensor([1, 1000, 1000000], device=voxel_coords.device)
        return (voxel_coords * multipliers).sum(dim=1)

    def get_neighbor_offsets(self, device):
        """Return 27 3D offset vectors."""
        offsets = torch.tensor([-1, 0, 1], device=device)
        return torch.cartesian_prod(offsets, offsets, offsets)  # (27, 3)

    def find_neighbors(self, gaussians, point_idx, voxel_coords, voxel_to_point_indices, key_to_index):
        """
        Find neighbors of a single point at index `point_idx` within radius `eps`.
        """
        points = gaussians.xyz
        device = points.device
        pi = points[point_idx] # (3,)
        vi = voxel_coords[point_idx] # (3,)
        neighbor_offsets = self.get_neighbor_offsets(device) # (27, 3)
        neighbor_voxels = vi[None, :] + neighbor_offsets # (27, 3)

        neighbor_keys = self.compute_voxel_keys(neighbor_voxels) # (27,)

        # collect all point indices from 27 neighbor voxels
        candidate_idxs = []
        for key in neighbor_keys:
            k = key.item()
            if k in key_to_index:
                candidate_idxs.extend(voxel_to_point_indices[key_to_index[k]])

        # Distance filter (vectorized)
        if candidate_idxs:
            #dists = torch.norm(candidates - pi, dim=1) # (m,)
            # wasserstein distance
            mu_candidates = gaussians.xyz[candidate_idxs]
            scale_candidates = gaussians.scaling[candidate_idxs]
            quat_candidates = gaussians.rotation[candidate_idxs]
            mu1 = pi[None, :] # (1, 3)
            scale1 = gaussians.scaling[point_idx][None, :] # (1, 3)
            quat1 = gaussians.rotation[point_idx][None, :] # (1, 4)
            dists = wasserstein_3d_gaussians(mu1, scale1, quat1,  # dists.shape: (m,)
                                             mu_candidates, scale_candidates, quat_candidates)
            neighbors = [idx for idx, d in zip(candidate_idxs, dists) if d <= self.eps and idx != point_idx]
        else:
            neighbors = []

        return torch.tensor(neighbors, device=device)

    def fit(self, gaussians):
        """
        Perform DBSCAN clustering on the input points.
        
        Args:
            gaussians: GaussianPrimitives object
        
        Returns:
            labels: Tensor of shape (n_points,) containing cluster assignments
                   -1 indicates noise points
                   >= 0 indicates cluster assignments
        """
        import ipdb; ipdb.set_trace()
        points = gaussians.xyz
        n_points = points.shape[0]
        labels = torch.full((n_points,), -2, dtype=torch.long)
        cluster_id = 0

        # Step 1: Voxelization and hashing
        voxel_coords = self.voxelize(points)
        voxel_keys = self.compute_voxel_keys(voxel_coords)

        # Step 2: Build inverse map using unique keys
        unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)

        # Step 3: Group points by voxel
        # Create a list of point indices per unique voxel key
        voxel_to_point_indices = [[] for _ in range(len(unique_keys))]
        for idx, group in enumerate(inverse_indices):
            voxel_to_point_indices[group.item()].append(idx)

        # Step 4: Build fast lookup table: key -> point indices
        key_to_index = {k.item(): i for i, k in enumerate(unique_keys)}

        # Step 5: Iterate over points and classify
        for point_idx in tqdm(range(n_points), desc="DBSCAN clustering"):
            if labels[point_idx] != -2: # already classified
                continue
                
            neighbor_indices = self.find_neighbors(gaussians, point_idx, voxel_coords, voxel_to_point_indices, key_to_index)
            
            if len(neighbor_indices) < self.min_pts:
                labels[point_idx] = -1 # mark as noise
                continue
                
            cluster_id += 1
            labels[point_idx] = cluster_id

            # expand cluster
            i = 0
            while i < len(neighbor_indices): # NOTE: neighbor_indices might expand
                neighbor_idx = neighbor_indices[i] # p'
                
                if labels[neighbor_idx] == -1: # if p' is noise, it is a border point, add to cluster
                    labels[neighbor_idx] = cluster_id
                elif labels[neighbor_idx] == -2: # if p' is unclassified, implicitly a core point or a border point
                    labels[neighbor_idx] = cluster_id
                    
                    neighbor_neighbors = self.find_neighbors(gaussians, neighbor_idx, voxel_coords, voxel_to_point_indices, key_to_index)
                    
                    if len(neighbor_neighbors) >= self.min_pts:
                        neighbor_indices = torch.cat([ # merge neighbor_neighbors into neighbor_indices
                            neighbor_indices,
                            neighbor_neighbors[~torch.isin(neighbor_neighbors, neighbor_indices)]
                        ])
                
                i += 1
        return labels

    def octree_sample_pytorch3d(self, data, device="cuda", max_depth=10):
        """
        Use PyTorch3D for octree-like sampling.
        """
        torch.cuda.synchronize(); t0 = time.time()
        
        point_to_voxel_idx, voxel_centers, voxel_levels = build_octree_with_pytorch3d(
            data, max_depth=max_depth, min_points_per_voxel=self.min_pts, device=device
        )
        
        self.point_to_voxel_idx = point_to_voxel_idx
        self.positions = voxel_centers
        self._level = voxel_levels
        
        torch.cuda.synchronize(); t1 = time.time()
        print(f"PyTorch3D octree time: {int((t1-t0) // 60)} min {(t1-t0) % 60:.2f} sec")

    def find_neighbors_pytorch3d(self, gaussians, point_idx):
        """
        Find neighbors using PyTorch3D's ball_query.
        """
        points = gaussians.xyz
        query_point = points[point_idx].unsqueeze(0)  # (1, 3)
        
        neighbor_lists = efficient_neighbor_search_pytorch3d(
            points, query_point, self.eps, device=points.device
        )
        
        neighbors = neighbor_lists[0]  # Get neighbors for the single query point
        # Remove self if present
        neighbors = neighbors[neighbors != point_idx]
        
        return neighbors

# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--fix_init", type=bool, default=False)
    args = parser.parse_args()

    gaussians = load_ply(args.path, args.sh_degree, args.fix_init)

    print(gaussians.xyz.shape, gaussians.features_dc.shape, gaussians.features_rest.shape, gaussians.opacity.shape, gaussians.scaling.shape, gaussians.rotation.shape)

    # Parameters
    eps = 1.5
    min_pts = 20
    
    # Initialize and run DBSCAN
    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    labels = dbscan.fit(gaussians)
    
    print(f"Number of clusters: {labels.max().item() + 1}")
    print(f"Number of noise points: {(labels == -1).sum().item()}")


def build_octree_with_pytorch3d(points, max_depth=10, min_points_per_voxel=20, device="cuda"):
    """
    Build octree-like structure using PyTorch3D operations.
    
    Args:
        points: torch.Tensor of shape (N, 3)
        max_depth: Maximum octree depth
        min_points_per_voxel: Minimum points required for a voxel
    
    Returns:
        point_to_voxel_idx: torch.Tensor mapping each point to its voxel representative
        voxel_centers: torch.Tensor of voxel center positions
        voxel_levels: torch.Tensor of depth level for each voxel
    """
    points = points.to(device)
    N = points.shape[0]
    
    # Initialize outputs
    point_to_voxel_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    voxel_centers = []
    voxel_levels = []
    
    # Compute bounding box
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    bbox_size = (max_coords - min_coords).max()
    bbox_center = (min_coords + max_coords) / 2
    
    # Track unassigned points
    unassigned_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    for level in range(max_depth):
        if not unassigned_mask.any():
            break
            
        # Current voxel size at this level
        voxel_size = bbox_size / (2 ** level)
        
        # Get unassigned points
        unassigned_points = points[unassigned_mask]
        unassigned_indices = torch.where(unassigned_mask)[0]
        
        if len(unassigned_points) == 0:
            break
        
        # Compute voxel coordinates
        voxel_coords = torch.floor((unassigned_points - (bbox_center - bbox_size/2)) / voxel_size).long()
        
        # Find unique voxels and their point counts
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_coords, dim=0, return_inverse=True, return_counts=True
        )
        
        # Filter for dense voxels
        dense_mask = counts >= min_points_per_voxel
        if not dense_mask.any():
            continue
            
        dense_voxels = unique_voxels[dense_mask]
        
        # Process each dense voxel
        for i, voxel_coord in enumerate(dense_voxels):
            # Find points in this voxel
            voxel_idx_in_unique = torch.where(dense_mask)[0][i]
            points_in_voxel_mask = (inverse_indices == voxel_idx_in_unique)
            points_in_voxel = unassigned_indices[points_in_voxel_mask]
            
            if len(points_in_voxel) == 0:
                continue
                
            # Compute voxel center in world coordinates
            voxel_center_world = (bbox_center - bbox_size/2) + (voxel_coord.float() + 0.5) * voxel_size
            
            # Find closest point to voxel center using PyTorch3D's knn_points
            voxel_points = points[points_in_voxel].unsqueeze(0)  # (1, M, 3)
            query_point = voxel_center_world.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            
            # Use knn to find closest point
            _, closest_idx, _ = knn_points(query_point, voxel_points, K=1)
            closest_local_idx = closest_idx[0, 0, 0]  # Extract scalar index
            closest_global_idx = points_in_voxel[closest_local_idx]
            
            # Assign all points in voxel to the closest point
            point_to_voxel_idx[points_in_voxel] = closest_global_idx
            
            # Mark points as assigned
            unassigned_mask[points_in_voxel] = False
            
            # Store voxel info
            voxel_centers.append(points[closest_global_idx])
            voxel_levels.append(level)
    
    # Convert to tensors
    if voxel_centers:
        voxel_centers = torch.stack(voxel_centers)
        voxel_levels = torch.tensor(voxel_levels, dtype=torch.int, device=device)
    else:
        voxel_centers = torch.empty(0, 3, device=device)
        voxel_levels = torch.empty(0, dtype=torch.int, device=device)
    
    return point_to_voxel_idx, voxel_centers, voxel_levels


def efficient_neighbor_search_pytorch3d(points, query_points, radius, device="cuda"):
    """
    Efficient neighbor search using PyTorch3D's ball_query.
    
    Args:
        points: torch.Tensor of shape (N, 3) - all points
        query_points: torch.Tensor of shape (M, 3) - query points  
        radius: float - search radius
        
    Returns:
        neighbor_lists: List of tensors, each containing neighbor indices for a query point
    """
    points = points.to(device)
    query_points = query_points.to(device)
    
    # PyTorch3D expects batch dimension
    points_batch = points.unsqueeze(0)  # (1, N, 3)
    query_batch = query_points.unsqueeze(0)  # (1, M, 3)
    
    # Use ball_query to find neighbors within radius
    # Note: ball_query returns fixed K neighbors, so we use a large K and filter
    max_neighbors = min(points.shape[0], 1000)  # Reasonable upper bound
    
    _, neighbor_indices, _ = ball_query(
        query_batch, points_batch, 
        radius=radius, 
        K=max_neighbors,
        return_nn=False
    )
    
    # Process results
    neighbor_lists = []
    for i in range(query_points.shape[0]):
        # Get neighbors for this query point
        neighbors = neighbor_indices[0, i]  # (K,)
        
        # Filter out invalid neighbors (ball_query pads with -1)
        valid_mask = neighbors >= 0
        valid_neighbors = neighbors[valid_mask]
        
        neighbor_lists.append(valid_neighbors)
    
    return neighbor_lists

