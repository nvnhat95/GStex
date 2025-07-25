# Ball Query DBSCAN for Gaussian Splats

This repository contains a ball query-based implementation of DBSCAN clustering specifically designed for Gaussian Splats/Primitives. The implementation uses PyTorch3D's `ball_query` for efficient neighbor search and the Wasserstein distance as the distance metric between Gaussian primitives.

## Overview

The ball query-based DBSCAN implementation removes the complexity of octree-based neighbor search from the original `dbscan.py` and provides a cleaner, more straightforward approach to clustering Gaussian Splats using PyTorch3D's optimized spatial queries.

### Key Features

- **Ball query neighbor search**: Uses PyTorch3D's `ball_query` instead of octree structures
- **Wasserstein distance metric**: Proper distance metric for Gaussian primitives considering both position and covariance
- **Automatic parameter estimation**: K-distance graph analysis for optimal eps estimation
- **GPU acceleration**: Full CUDA support for large point clouds
- **Easy-to-use interface**: Simple API for loading PLY files and running clustering
- **Comprehensive analysis**: Built-in cluster analysis and visualization tools

## Files

- `dbscan_ballquery.py`: Main implementation of BallQueryDBSCAN class
- `demo_ballquery_dbscan.py`: Demo script showing how to use the implementation with PLY files
- `README_DBSCAN.md`: This documentation file

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install pytorch3d
pip install plyfile
pip install tqdm
pip install numpy
pip install matplotlib  # For k-distance plots and visualization
pip install scipy      # For k-distance analysis
```

You'll also need the nerfstudio package for rotation utilities:
```bash
pip install nerfstudio
```

## Usage

### Basic Usage

```python
from dbscan_ballquery import BallQueryDBSCAN, GaussianPrimitives

# Initialize DBSCAN with automatic eps estimation
min_pts = 20  # Minimum points per cluster
dbscan = BallQueryDBSCAN(eps=1.0, min_pts=min_pts)  # Initial eps will be updated

# Estimate good eps value using k-distance graph
suggested_eps = dbscan.plot_k_distance_graph(gaussians, k=min_pts)
dbscan.eps = suggested_eps

# Run clustering (gaussians is a GaussianPrimitives object)
labels = dbscan.fit(gaussians)

# Analyze results
results = dbscan.analyze_clusters(labels)
print(f"Found {results['n_clusters']} clusters")
```

### Using with PLY Files

```python
from demo_ballquery_dbscan import load_ply
from dbscan_ballquery import BallQueryDBSCAN

# Load PLY file
gaussians = load_ply("path/to/your/file.ply", sh_degree=3)

# Run DBSCAN with automatic eps estimation
dbscan = BallQueryDBSCAN(eps=1.0, min_pts=20)
suggested_eps = dbscan.plot_k_distance_graph(gaussians)
dbscan.eps = suggested_eps
labels = dbscan.fit(gaussians)
```

### Command Line Usage

Use the demo script to run DBSCAN on PLY files:

```bash
# Automatic eps estimation (recommended)
python demo_ballquery_dbscan.py --path /path/to/your/file.ply --min_pts 20

# Manual eps specification
python demo_ballquery_dbscan.py --path /path/to/your/file.ply --eps 1.5 --min_pts 20
```

#### Command Line Arguments

- `--path`: Path to PLY file (required)
- `--sh_degree`: Spherical harmonics degree (default: 3)
- `--fix_init`: Fix initialization for COLMAP/DTU datasets
- `--eps`: DBSCAN eps parameter - Wasserstein distance threshold (default: None, will be estimated)
- `--min_pts`: DBSCAN min_pts parameter (default: 20)
- `--search_multiplier`: Search radius multiplier for ball_query (default: 2.0)
- `--device`: Device to use - cuda/cpu/auto (default: auto)
- `--output`: Output file to save results
- `--max_points`: Maximum number of points to process (for testing)
- `--n_samples`: Number of points to sample for eps estimation (default: 1000)
- `--skip_eps_estimation`: Skip eps estimation even if eps is not provided

### Example Commands

```bash
# Automatic eps estimation
python demo_ballquery_dbscan.py --path scene.ply --min_pts 15

# Manual eps specification
python demo_ballquery_dbscan.py --path scene.ply --eps 1.5 --min_pts 20 --output results.txt

# Automatic eps estimation with more samples
python demo_ballquery_dbscan.py --path scene.ply --min_pts 20 --n_samples 2000

# Test with subset of points
python demo_ballquery_dbscan.py --path large_scene.ply --max_points 5000

# For COLMAP/DTU datasets
python demo_ballquery_dbscan.py --path colmap_scene.ply --fix_init
```

## Algorithm Details

### DBSCAN Algorithm

The implementation follows the standard DBSCAN algorithm:

1. **Parameter Estimation**: (Optional) Use k-distance graph to find optimal eps
2. **Initialization**: All points start as unclassified (-2)
3. **Core Point Detection**: For each unclassified point, find neighbors within `eps` distance
4. **Cluster Formation**: If a point has ≥ `min_pts` neighbors, start a new cluster
5. **Cluster Expansion**: Recursively add neighbors and their neighbors to the cluster
6. **Noise Classification**: Points that don't belong to any cluster are marked as noise (-1)

### K-Distance Graph Analysis

The k-distance graph helps find an optimal eps value:

1. For each point, find its k-th nearest neighbor (k = min_pts)
2. Plot these k-distances in sorted order
3. Find the "elbow" point in the curve
4. Use this distance as eps

Benefits:
- Data-driven parameter selection
- Reduces trial and error
- Adapts to different scales and densities
- Helps avoid too many noise points

### Wasserstein Distance

The distance between two Gaussian primitives is computed using the 2-Wasserstein distance:

```
W²(G₁, G₂) = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(√Σ₂ Σ₁ √Σ₂))
```

Where:
- `μ₁, μ₂` are the means (positions)
- `Σ₁, Σ₂` are the covariance matrices (built from scaling and rotation)

### Neighbor Search Strategy

1. **Initial Search**: Use PyTorch3D's `ball_query` with radius = `eps * search_multiplier`
2. **Distance Filtering**: Compute Wasserstein distances for all candidates
3. **Final Filtering**: Return only neighbors with Wasserstein distance ≤ `eps`

The `search_multiplier` parameter accounts for the fact that Euclidean distance (used by `ball_query`) may differ significantly from Wasserstein distance.

## Parameters

### Core Parameters

- **`eps`**: Maximum Wasserstein distance for points to be considered neighbors
  - Can be automatically estimated using k-distance graph (recommended)
  - Smaller values → more, tighter clusters
  - Larger values → fewer, looser clusters
  - Typical range: 0.5 - 5.0 depending on scene scale

- **`min_pts`**: Minimum number of points required to form a cluster
  - Smaller values → more clusters (including small ones)
  - Larger values → fewer, denser clusters
  - Typical range: 5 - 50 depending on point cloud density
  - Also used as k in k-distance analysis

### Advanced Parameters

- **`search_radius_multiplier`**: Multiplier for initial ball_query radius
  - Higher values → more thorough search but slower
  - Lower values → faster but may miss neighbors
  - Default: 2.0 (good for most cases)

- **`n_samples`**: Number of points to use for eps estimation
  - Higher values → more accurate estimation but slower
  - Lower values → faster but might miss structure
  - Default: 1000 (good balance)

## Performance Considerations

### Memory Usage
- Each point requires neighbor search, which can be memory-intensive for large datasets
- Use `--max_points` to test with subsets first
- Consider using CPU for very large datasets if GPU memory is limited

### Speed Optimization
- GPU acceleration provides significant speedup
- PyTorch3D's `ball_query` is much faster than naive neighbor search
- Progress bars show clustering progress
- Eps estimation uses sampling to reduce computation time

### Scalability
- Time complexity: O(n²) in worst case, but typically much better due to spatial locality
- Space complexity: O(n) for storing labels and neighbor lists
- Tested with point clouds up to 100K+ points

## Output Analysis

The `analyze_clusters()` method provides comprehensive statistics:

```python
results = {
    'n_clusters': int,           # Number of clusters found
    'n_noise_points': int,       # Number of noise points
    'n_total_points': int,       # Total number of points
    'noise_ratio': float,        # Percentage of noise points
    'cluster_sizes': list,       # Size of each cluster
    'avg_cluster_size': float,   # Average cluster size
    'largest_cluster_size': int, # Size of largest cluster
    'smallest_cluster_size': int # Size of smallest cluster
}
```

## Troubleshooting

### Common Issues

1. **"Ball query failed"**: PyTorch3D installation issue or CUDA compatibility
   - Fallback: Uses all points as candidates (slower but works)
   - Solution: Reinstall PyTorch3D with correct CUDA version

2. **Out of memory**: Point cloud too large for GPU
   - Solution: Use `--max_points` to test with subset
   - Solution: Use `--device cpu` to run on CPU

3. **No clusters found**: Parameters too strict
   - Solution: Use automatic eps estimation
   - Solution: Increase `eps` or decrease `min_pts`
   - Solution: Check point cloud scale and adjust `eps` accordingly

4. **Too many small clusters**: Parameters too loose
   - Solution: Use automatic eps estimation with higher min_pts
   - Solution: Decrease `eps` or increase `min_pts`

5. **Too many noise points (>90%)**:
   - Solution: Use automatic eps estimation (recommended)
   - Solution: Increase `eps` value
   - Solution: Check k-distance plot for better parameter selection

### Parameter Tuning

1. **Use automatic eps estimation**: Let the algorithm find a good eps value
2. **Start with defaults**: `min_pts=20`, `n_samples=1000`
3. **Visualize k-distance plot**: Understand the distance distribution
4. **Adjust min_pts**: Fine-tune cluster density requirements
5. **If needed, manual eps**: Use k-distance plot as a guide

## Comparison with Original Implementation

### Ball Query Version (This Implementation)
- ✅ Automatic parameter estimation
- ✅ Cleaner codebase (no octree complexity)
- ✅ Uses battle-tested PyTorch3D operations
- ✅ More readable and maintainable
- ✅ Built-in analysis and visualization
- ✅ Leverages optimized spatial queries from PyTorch3D
- ❌ May be slower for very large datasets
- ❌ Less memory-efficient for sparse data

### Original Version (dbscan.py)
- ✅ Potentially faster for large, sparse datasets
- ✅ More memory-efficient octree structure
- ❌ Manual parameter tuning required
- ❌ Complex octree implementation
- ❌ Harder to debug and modify
- ❌ More prone to bugs in spatial data structures

## Contributing

To contribute improvements or report issues:

1. Test with various PLY files and parameter combinations
2. Profile performance on large datasets
3. Report any edge cases or failure modes
4. Suggest parameter tuning guidelines for specific use cases 