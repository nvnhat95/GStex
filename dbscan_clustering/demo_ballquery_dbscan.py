#!/usr/bin/env python3
"""
Demo script for BallQueryDBSCAN with PLY data.
This script shows how to load PLY files and run the ball query-based DBSCAN clustering.
"""

import argparse
import torch
import numpy as np
import time
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
    Save clustering results to a text file with detailed cluster statistics.
    
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
        
        # Overall statistics
        cluster_sizes = []
        cluster_centers = []
        cluster_bounds = []
        cluster_opacities = []
        cluster_scales = []
        
        for cluster_id in unique_labels[unique_labels >= 0]:
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum().item()
            cluster_points = gaussians.xyz[cluster_mask]
            cluster_opacity = gaussians.opacity[cluster_mask]
            cluster_scale = gaussians.scaling[cluster_mask]
            
            # Compute cluster statistics
            center = cluster_points.mean(dim=0)
            min_coords = cluster_points.min(dim=0)[0]
            max_coords = cluster_points.max(dim=0)[0]
            bbox_size = max_coords - min_coords
            avg_opacity = cluster_opacity.mean().item()
            avg_scale = cluster_scale.mean(dim=0)
            
            cluster_sizes.append(cluster_size)
            cluster_centers.append(center)
            cluster_bounds.append(bbox_size)
            cluster_opacities.append(avg_opacity)
            cluster_scales.append(avg_scale)
            
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Size: {cluster_size} points ({cluster_size/n_total*100:.1f}%)\n")
            f.write(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n")
            f.write(f"  Bounding box: {bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}\n")
            f.write(f"  Avg opacity: {avg_opacity:.3f}\n")
            f.write(f"  Avg scale: ({avg_scale[0]:.3f}, {avg_scale[1]:.3f}, {avg_scale[2]:.3f})\n\n")
        
        # Summary statistics
        if cluster_sizes:
            f.write("Summary Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f}\n")
            f.write(f"Largest cluster: {max(cluster_sizes)} points\n")
            f.write(f"Smallest cluster: {min(cluster_sizes)} points\n")
            f.write(f"Cluster size std dev: {np.std(cluster_sizes):.1f}\n")
            
            # Spatial distribution
            centers = torch.stack(cluster_centers)
            center_distances = torch.cdist(centers, centers)
            avg_inter_cluster_dist = center_distances[center_distances > 0].mean().item()
            f.write(f"Average inter-cluster distance: {avg_inter_cluster_dist:.3f}\n")
            
            # Opacity and scale statistics
            avg_opacity = sum(cluster_opacities) / len(cluster_opacities)
            f.write(f"Average cluster opacity: {avg_opacity:.3f}\n")
            
            avg_scales = torch.stack(cluster_scales).mean(dim=0)
            f.write(f"Average cluster scale: ({avg_scales[0]:.3f}, {avg_scales[1]:.3f}, {avg_scales[2]:.3f})\n\n")
    
    print(f"Results saved to: {output_path}")


def visualize_clusters_simple(gaussians, labels, max_clusters_to_show=10):
    """
    Enhanced visualization of cluster statistics with detailed analysis.
    """
    unique_labels = torch.unique(labels)
    cluster_labels = unique_labels[unique_labels >= 0]
    n_noise = (labels == -1).sum().item()
    n_total = len(labels)
    
    print(f"\n{'='*80}")
    print(f"CLUSTER ANALYSIS & VISUALIZATION")
    print(f"{'='*80}")
    print(f"Total clusters found: {len(cluster_labels)}")
    print(f"Noise points: {n_noise} ({n_noise/n_total*100:.1f}%)")
    print(f"Clustered points: {n_total - n_noise} ({(n_total - n_noise)/n_total*100:.1f}%)")
    print(f"{'='*80}")
    
    # Collect cluster statistics
    cluster_stats = []
    for cluster_id in cluster_labels:
        cluster_mask = labels == cluster_id
        cluster_size = cluster_mask.sum().item()
        cluster_points = gaussians.xyz[cluster_mask]
        cluster_opacity = gaussians.opacity[cluster_mask]
        cluster_scale = gaussians.scaling[cluster_mask]
        
        # Compute statistics
        center = cluster_points.mean(dim=0)
        bbox_size = cluster_points.max(dim=0)[0] - cluster_points.min(dim=0)[0]
        avg_opacity = cluster_opacity.mean().item()
        avg_scale = cluster_scale.mean(dim=0)
        scale_magnitude = torch.norm(avg_scale).item()
        
        cluster_stats.append({
            'id': cluster_id.item(),
            'size': cluster_size,
            'percentage': cluster_size / n_total * 100,
            'center': center,
            'bbox_size': bbox_size,
            'avg_opacity': avg_opacity,
            'avg_scale': avg_scale,
            'scale_magnitude': scale_magnitude
        })
    
    # Sort by size
    cluster_stats.sort(key=lambda x: x['size'], reverse=True)
    
    # Display top clusters
    print(f"Top {min(max_clusters_to_show, len(cluster_stats))} clusters by size:")
    print(f"{'ID':<6} {'Size':<8} {'%':<6} {'Center (x,y,z)':<25} {'BBox':<20} {'Opacity':<8} {'Scale':<12}")
    print(f"{'-'*90}")
    
    for i, stats in enumerate(cluster_stats[:max_clusters_to_show]):
        center_str = f"({stats['center'][0]:.2f},{stats['center'][1]:.2f},{stats['center'][2]:.2f})"
        bbox_str = f"{stats['bbox_size'][0]:.2f}x{stats['bbox_size'][1]:.2f}x{stats['bbox_size'][2]:.2f}"
        scale_str = f"{stats['scale_magnitude']:.3f}"
        
        print(f"{stats['id']:<6} {stats['size']:<8} {stats['percentage']:<6.1f} {center_str:<25} {bbox_str:<20} {stats['avg_opacity']:<8.3f} {scale_str:<12}")
    
    # Summary statistics
    if cluster_stats:
        sizes = [s['size'] for s in cluster_stats]
        opacities = [s['avg_opacity'] for s in cluster_stats]
        scales = [s['scale_magnitude'] for s in cluster_stats]
        
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Cluster size - Mean: {np.mean(sizes):.1f}, Std: {np.std(sizes):.1f}, Min: {min(sizes)}, Max: {max(sizes)}")
        print(f"Opacity - Mean: {np.mean(opacities):.3f}, Std: {np.std(opacities):.3f}, Min: {min(opacities):.3f}, Max: {max(opacities):.3f}")
        print(f"Scale magnitude - Mean: {np.mean(scales):.3f}, Std: {np.std(scales):.3f}, Min: {min(scales):.3f}, Max: {max(scales):.3f}")
        
        # Spatial distribution analysis
        centers = torch.stack([s['center'] for s in cluster_stats])
        try:
            # Try using batched computation if number of centers is large
            if len(centers) > 1000:
                inter_cluster_dists = compute_center_distances_batched(centers)
            else:
                center_distances = torch.cdist(centers, centers)
                inter_cluster_dists = center_distances[center_distances > 0]
                
            if len(inter_cluster_dists) > 0:
                print(f"Inter-cluster distances - Mean: {inter_cluster_dists.mean():.3f}, Std: {inter_cluster_dists.std():.3f}")
        except RuntimeError as e:
            print(f"Warning: Could not compute inter-cluster distances due to memory constraints")
        
        # Density analysis
        total_volume = sum(s['bbox_size'][0] * s['bbox_size'][1] * s['bbox_size'][2] for s in cluster_stats)
        avg_density = sum(sizes) / total_volume if total_volume > 0 else 0
        print(f"Average cluster density: {avg_density:.3f} points per unit volume")
    
    print(f"{'='*80}")


def create_cluster_visualization(gaussians, labels, max_clusters_to_show=10):
    """
    Create a simple 3D scatter plot visualization of clusters.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Get unique cluster labels
        unique_labels = torch.unique(labels)
        cluster_labels = unique_labels[unique_labels >= 0]
        
        # Limit number of clusters to show
        if len(cluster_labels) > max_clusters_to_show:
            # Get top clusters by size
            cluster_sizes = [(label, (labels == label).sum().item()) for label in cluster_labels]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            cluster_labels = torch.tensor([label for label, _ in cluster_sizes[:max_clusters_to_show]])
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster with different color
        colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_labels)))
        
        for i, cluster_id in enumerate(cluster_labels):
            cluster_mask = labels == cluster_id
            cluster_points = gaussians.xyz[cluster_mask].cpu().numpy()
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                      c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=1)
        
        # Plot noise points in black
        noise_mask = labels == -1
        if noise_mask.sum() > 0:
            noise_points = gaussians.xyz[noise_mask].cpu().numpy()
            ax.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                      c='black', label='Noise', alpha=0.3, s=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'BallQueryDBSCAN Clustering Results ({len(cluster_labels)} clusters shown)')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available, skipping 3D visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


def compute_center_distances_batched(centers, batch_size=1000):
    """
    Compute pairwise distances between centers in a memory-efficient way using batching.
    
    Args:
        centers (torch.Tensor): Tensor of cluster centers
        batch_size (int): Size of batches to process at once
    
    Returns:
        torch.Tensor: Tensor containing non-zero pairwise distances
    """
    n_centers = centers.shape[0]
    device = centers.device
    all_distances = []
    
    for i in range(0, n_centers, batch_size):
        end_idx = min(i + batch_size, n_centers)
        batch_centers = centers[i:end_idx]
        
        # Compute distances between current batch and all centers
        batch_distances = torch.cdist(batch_centers, centers)
        
        # Extract non-zero distances from this batch
        batch_nonzero = batch_distances[batch_distances > 0]
        all_distances.append(batch_nonzero)
    
    # Combine all non-zero distances
    return torch.cat(all_distances) if all_distances else torch.tensor([], device=device)


def main():
    parser = argparse.ArgumentParser(description="Run BallQueryDBSCAN on PLY files")
    parser.add_argument("--path", type=str, required=True, help="Path to PLY file")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--fix_init", action="store_true", help="Fix initialization for COLMAP/DTU")
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps parameter (wasserstein distance threshold). If not provided, will be estimated.")
    parser.add_argument("--min_pts", type=int, default=20, help="DBSCAN min_pts parameter")
    parser.add_argument("--search_multiplier", type=float, default=1.0, help="Search radius multiplier for ball_query")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--output", type=str, default=None, help="Output file to save results")
    parser.add_argument("--max_points", type=int, default=None, help="Maximum number of points to process (for testing)")
    parser.add_argument("--visualize", action="store_true", help="Create 3D scatter plot visualization of clusters")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of points to sample for eps estimation")
    parser.add_argument("--skip_eps_estimation", action="store_true", help="Skip eps estimation even if eps is not provided")
    
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
        
        # Print data statistics
        print(f"Data statistics:")
        print(f"  Point cloud bounds: {gaussians.xyz.min(dim=0)[0]} to {gaussians.xyz.max(dim=0)[0]}")
        print(f"  Scale range: {gaussians.scaling.min():.3f} to {gaussians.scaling.max():.3f}")
        print(f"  Opacity range: {gaussians.opacity.min():.3f} to {gaussians.opacity.max():.3f}")
        
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
    
    # Initialize DBSCAN with a temporary eps if we need to estimate it
    eps = args.eps if args.eps is not None else 1.0
    dbscan = BallQueryDBSCAN(
        eps=eps,
        min_pts=args.min_pts,
        search_radius_multiplier=args.search_multiplier
    )
    
    # Estimate eps if not provided
    if args.eps is None and not args.skip_eps_estimation:
        print(f"\nEstimating eps parameter using k-distance graph...")
        print(f"Sampling {args.n_samples} points for estimation...")
        suggested_eps = dbscan.plot_k_distance_graph(
            gaussians,
            k=args.min_pts,
            n_samples=args.n_samples
        )
        print(f"Suggested eps from k-distance graph: {suggested_eps:.4f}")
        
        # Update eps
        dbscan.eps = suggested_eps
        eps = suggested_eps
    
    print(f"\nBall Query DBSCAN parameters: eps={eps}, min_pts={args.min_pts}, search_multiplier={args.search_multiplier}")
    
    try:
        print(f"Starting clustering process...")
        start_time = time.time()
        
        labels = dbscan.fit(gaussians)
        
        clustering_time = time.time() - start_time
        print(f"Clustering completed in {clustering_time:.2f} seconds")
        
        # Analyze results
        print(f"Analyzing clustering results...")
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
            print(f"Cluster size distribution: {sorted(results['cluster_sizes'], reverse=True)[:10]}...")
        print(f"Processing speed: {gaussians.xyz.shape[0] / clustering_time:.0f} points/second")
        
        # Enhanced visualization
        visualize_clusters_simple(gaussians, labels)
        
        # Save results if requested
        if args.output:
            save_cluster_results(gaussians, labels, args.output)
        
        # Create visualization if requested
        if args.visualize:
            print(f"\nCreating 3D visualization...")
            create_cluster_visualization(gaussians, labels)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successfully clustered {gaussians.xyz.shape[0]} Gaussian primitives")
        print(f"✓ Found {results['n_clusters']} clusters with {results['n_noise_points']} noise points")
        print(f"✓ Processing speed: {gaussians.xyz.shape[0] / clustering_time:.0f} points/second")
        if args.output:
            print(f"✓ Results saved to: {args.output}")
        if args.visualize:
            print(f"✓ 3D visualization created")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 