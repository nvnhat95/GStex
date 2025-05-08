import torch
import numpy as np
from nerfstudio.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
from plyfile import PlyData


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
    return [new_xyz, features_dc, features_rest, opacity, scaling, new_rotations]


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

    def find_neighbors(self, points, point_idx, voxel_coords, voxel_to_point_indices, key_to_index):
        """
        Find neighbors of a single point at index `point_idx` within radius `eps`.
        """
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
            candidates = points[candidate_idxs] # (m, 3)
            dists = torch.norm(candidates - pi, dim=1) # (m,)
            neighbors = [idx for idx, d in zip(candidate_idxs, dists) if d <= self.eps and idx != point_idx]
        else:
            neighbors = []

        return neighbors

    def fit(self, points):
        """
        Perform DBSCAN clustering on the input points.
        
        Args:
            points: Tensor of shape (n_points, n_features)
        
        Returns:
            labels: Tensor of shape (n_points,) containing cluster assignments
                   -1 indicates noise points
                   >= 0 indicates cluster assignments
        """
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
        for point_idx in range(n_points):
            if labels[point_idx] != -2: # already classified
                continue
                
            neighbor_indices = self.find_neighbors(points, point_idx, voxel_coords, voxel_to_point_indices, key_to_index)
            
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
                    
                    neighbor_neighbors = self.find_neighbors(points, neighbor_idx, voxel_coords, voxel_to_point_indices, key_to_index)
                    
                    if len(neighbor_neighbors) >= self.min_pts:
                        neighbor_indices = torch.cat([ # merge neighbor_neighbors into neighbor_indices
                            neighbor_indices,
                            neighbor_neighbors[~torch.isin(neighbor_neighbors, neighbor_indices)]
                        ])
                
                i += 1
        
        return labels

# Example usage:
if __name__ == "__main__":
    # Create sample data
    points = np.array([
        [1, 2, 0], [2, 2, 0], [2, 3, 0],
        [8, 7, 0], [8, 8, 0], [7, 8, 0],
        [0, 1, 0], [5, 5, 0], [5, 6, 0],
        [7, 6, 0], [10, 1, 0], [9, 2, 0]
    ])
    points = torch.tensor(points, dtype=torch.float)

    # Parameters
    eps = 1.5
    min_pts = 3
    
    # Initialize and run DBSCAN
    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    labels = dbscan.fit(points)
    
    print(f"Number of clusters: {labels.max().item() + 1}")
    print(f"Number of noise points: {(labels == -1).sum().item()}")

