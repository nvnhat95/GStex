#!/usr/bin/env python3
"""
Demo script for BallQueryDBSCAN with PLY data.
This script shows how to load PLY files and run the ball query-based DBSCAN clustering.
"""

import argparse
import torch
import numpy as np
from plyfile import PlyData
from nerfstudio.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
from dbscan_ballquery import BallQueryDBSCAN, GaussianPrimitives


def load_ply(path, sh_degree=0, fix_init=False):
    """
    Load a PLY file and return the point cloud data.
    
    Args:
        path (str): Path to the PLY file
        sh_degree (int, optional): Spherical harmonics degree. Defaults to 0.
        fix_init (bool, optional): Blender uses False, COLMAP/DTU uses True.
    
    Returns:
        GaussianPrimitives: Loaded Gaussian primitives
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


def save_cluster_results(gaussians, labels, output_path):
    """
    Save clustering results to a text file with cluster statistics.
    
    Args:
        gaussians: GaussianPrimitives object
        labels: Cluster labels
        output_path: Path to save results
    """
    import os
    
    # Analyze clusters
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    n_noise = (labels == -1).sum().item()
    n_total = len(labels)
    
    with open(output_path, 'w') as f:
        f.write("Ball Query DBSCAN Clustering Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total points: {n_total}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Noise points: {n_noise}\n")
        f.write(f"Noise ratio: {n_noise / n_total:.2%}\n\n")
        
        f.write("Cluster Details:\n")
        f.write("-" * 30 + "\n")
        
        for cluster_id in unique_labels[unique_labels >= 0]:
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum().item()
            cluster_points = gaussians.xyz[cluster_mask]
            
            # Compute cluster center and bounding box
            center = cluster_points.mean(dim=0)
            min_coords = cluster_points.min(dim=0)[0]
            max_coords = cluster_points.max(dim=0)[0]
            bbox_size = max_coords - min_coords
            
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Size: {cluster_size} points\n")
            f.write(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n")
            f.write(f"  Bounding box: {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}\n\n")
    
    print(f"Results saved to: {output_path}")


def visualize_clusters_simple(gaussians, labels, max_clusters_to_show=10):
    """
    Simple visualization of cluster statistics.
    """
    unique_labels = torch.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    n_noise = (labels == -1).sum().item()
    
    print(f"\n{'='*60}")
    print(f"CLUSTER VISUALIZATION")
    print(f"{'='*60}")
    print(f"Total clusters found: {len(cluster_labels)}")
    print(f"Noise points: {n_noise}")
    print(f"{'='*60}")
    
    # Show top clusters by size
    cluster_sizes = []
    for cluster_id in cluster_labels:
        size = (labels == cluster_id).sum().item()
        cluster_sizes.append((cluster_id.item(), size))
    
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {min(max_clusters_to_show, len(cluster_sizes))} clusters by size:")
    print(f"{'Cluster ID':<12} {'Size':<8} {'Percentage':<12}")
    print(f"{'-'*32}")
    
    total_points = len(labels)
    for i, (cluster_id, size) in enumerate(cluster_sizes[:max_clusters_to_show]):
        percentage = size / total_points * 100
        print(f"{cluster_id:<12} {size:<8} {percentage:<12.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run BallQueryDBSCAN on PLY files")
    parser.add_argument("--path", type=str, required=True, help="Path to PLY file")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--fix_init", action="store_true", help="Fix initialization for COLMAP/DTU")
    parser.add_argument("--eps", type=float, default=1.5, help="DBSCAN eps parameter (wasserstein distance threshold)")
    parser.add_argument("--min_pts", type=int, default=20, help="DBSCAN min_pts parameter")
    parser.add_argument("--search_multiplier", type=float, default=2.0, help="Search radius multiplier for ball_query")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--output", type=str, default=None, help="Output file to save results")
    parser.add_argument("--max_points", type=int, default=None, help="Maximum number of points to process (for testing)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Loading PLY file: {args.path}")
    
    # Load PLY data
    try:
        gaussians = load_ply(args.path, args.sh_degree, args.fix_init)
        print(f"Loaded {gaussians.xyz.shape[0]} Gaussian primitives")
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return
    
    # Move to device
    gaussians.xyz = gaussians.xyz.to(device)
    gaussians.scaling = gaussians.scaling.to(device)
    gaussians.rotation = gaussians.rotation.to(device)
    gaussians.opacity = gaussians.opacity.to(device)
    gaussians.features_dc = gaussians.features_dc.to(device)
    gaussians.features_rest = gaussians.features_rest.to(device)
    
    # Optionally subsample for testing
    if args.max_points and gaussians.xyz.shape[0] > args.max_points:
        print(f"Subsampling to {args.max_points} points for testing...")
        indices = torch.randperm(gaussians.xyz.shape[0])[:args.max_points]
        gaussians.xyz = gaussians.xyz[indices]
        gaussians.scaling = gaussians.scaling[indices]
        gaussians.rotation = gaussians.rotation[indices]
        gaussians.opacity = gaussians.opacity[indices]
        gaussians.features_dc = gaussians.features_dc[indices]
        gaussians.features_rest = gaussians.features_rest[indices]
    
    print(f"Processing {gaussians.xyz.shape[0]} points")
    print(f"Ball Query DBSCAN parameters: eps={args.eps}, min_pts={args.min_pts}")
    
    # Initialize and run DBSCAN
    dbscan = BallQueryDBSCAN(
        eps=args.eps, 
        min_pts=args.min_pts,
        search_radius_multiplier=args.search_multiplier
    )
    
    try:
        labels = dbscan.fit(gaussians)
        
        # Analyze results
        results = dbscan.analyze_clusters(labels)
        
        print(f"\n{'='*60}")
        print(f"CLUSTERING COMPLETED")
        print(f"{'='*60}")
        print(f"Number of clusters: {results['n_clusters']}")
        print(f"Number of noise points: {results['n_noise_points']}")
        print(f"Noise ratio: {results['noise_ratio']:.2%}")
        if results['cluster_sizes']:
            print(f"Average cluster size: {results['avg_cluster_size']:.1f}")
            print(f"Largest cluster: {results['largest_cluster_size']} points")
            print(f"Smallest cluster: {results['smallest_cluster_size']} points")
        
        # Simple visualization
        visualize_clusters_simple(gaussians, labels)
        
        # Save results if requested
        if args.output:
            save_cluster_results(gaussians, labels, args.output)
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 