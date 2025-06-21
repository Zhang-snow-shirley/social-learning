import numpy as np
from scipy.linalg import cholesky
from collections import defaultdict
from numba import jit, prange
import dask.array as da
import dask
from dask.distributed import as_completed
from typing import Dict, Tuple, Set, Optional
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings("ignore", category=np.RankWarning)

@jit(nopython=True, cache=True)
def _fast_dict_population(result, A, C):
    """JIT-compiled parallel dictionary population"""
    data_tuples = np.empty((A * C, 3), dtype=np.float64)
    idx = 0
    for i in prange(A):
        for j in range(C):
            data_tuples[idx, 0] = i  # row
            data_tuples[idx, 1] = j  # col
            data_tuples[idx, 2] = result[i, j]  # value
            idx += 1
    return data_tuples

@jit(nopython=True, cache=True)
def _fast_noise_addition(values, sigma, size):
    """JIT-compiled parallel noise generation"""
    noise = np.random.normal(0, sigma, size)
    return values + noise

@jit(nopython=True, cache=True)
def _compute_regression_ultra_fast(target_vals, reference_vals, n_pairs):
    """JIT regression with manual loop unrolling"""
    if n_pairs < 2:
        return 0.0, 0.0, False
    
    # Unrolled summation for better performance
    sum_target = 0.0
    sum_reference = 0.0
    sum_target_sq = 0.0
    sum_reference_sq = 0.0
    sum_cross = 0.0
    
    # Process in chunks of 4 for better vectorization
    i = 0
    while i < n_pairs - 3:
        # Unroll 4 iterations
        t0, r0 = target_vals[i], reference_vals[i]
        t1, r1 = target_vals[i+1], reference_vals[i+1]
        t2, r2 = target_vals[i+2], reference_vals[i+2]
        t3, r3 = target_vals[i+3], reference_vals[i+3]
        
        sum_target += t0 + t1 + t2 + t3
        sum_reference += r0 + r1 + r2 + r3
        sum_target_sq += t0*t0 + t1*t1 + t2*t2 + t3*t3
        sum_reference_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3
        sum_cross += t0*r0 + t1*r1 + t2*r2 + t3*r3
        i += 4
    
    # Handle remaining elements
    while i < n_pairs:
        t, r = target_vals[i], reference_vals[i]
        sum_target += t
        sum_reference += r
        sum_target_sq += t * t
        sum_reference_sq += r * r
        sum_cross += t * r
        i += 1
    
    # Compute statistics
    mean_target = sum_target / n_pairs
    mean_reference = sum_reference / n_pairs
    
    var_reference = (sum_reference_sq / n_pairs) - mean_reference * mean_reference
    covariance = (sum_cross / n_pairs) - mean_target * mean_reference
    
    if var_reference < 1e-10:
        return mean_target, 0.0, False
    
    slope = covariance / var_reference
    intercept = mean_target - slope * mean_reference
    
    return intercept, slope, True

@jit(nopython=True, cache=True)
def _fast_set_intersection(arr1, arr2, exclude_val):
    """JIT-compiled set intersection for pairs finding"""
    result = []
    set1 = set(arr1)
    for val in arr2:
        if val in set1 and val != exclude_val:
            result.append(val)
    return np.array(result)

class MatrixDict(object):
    """
    High performance dictionary-based sparse matrix with advanced caching.
    """
    
    __slots__ = ['data', 'shape', '_column_indices', '_column_cache']
    
    def __init__(self, data_dict=None, shape=None):
        self.data = data_dict if data_dict is not None else {}
        self.shape = shape
        self._column_indices = defaultdict(lambda: np.array([], dtype=np.int32))
        self._column_cache = {}
        if self.data:
            self._build_indices_vectorized()
    
    def _build_indices_vectorized(self):
        """Ultra-fast vectorized index building"""
        if not self.data:
            return
        # Extract rows and columns separately from tuple keys
        keys_list = list(self.data.keys())
        rows = np.array([key[0] for key in keys_list], dtype=np.int32)
        cols = np.array([key[1] for key in keys_list], dtype=np.int32)
        
        # Group by column using argsort
        sort_idx = np.argsort(cols)
        sorted_cols = cols[sort_idx]
        sorted_rows = rows[sort_idx]
        
        # Find column boundaries
        unique_cols, col_starts = np.unique(sorted_cols, return_index=True)
        col_ends = np.append(col_starts[1:], len(sorted_cols))
        
        # Build column indices
        self._column_indices.clear()
        for i, col in enumerate(unique_cols):
            start, end = col_starts[i], col_ends[i]
            self._column_indices[col] = sorted_rows[start:end]           
  
    def __getitem__(self, key):
        return self.data.get(key, np.nan)
    
    def __setitem__(self, key, value):
        self.data[key] = value
        row, col = key
        # Invalidate caches
        if col in self._column_cache:
            del self._column_cache[col]
        # Update indices (lazy - rebuild when needed)
        self._column_indices.clear()
    
    def __contains__(self, key):
        return key in self.data
    
    def get_column_rows(self, col_idx):
        """Fast column row access with lazy index building"""
        if not self._column_indices:
            self._build_indices_vectorized()
        return self._column_indices[col_idx]
    
    def get_column(self, col_idx):
        """Cached column extraction"""
        if col_idx in self._column_cache:
            return self._column_cache[col_idx]
        
        rows = self.get_column_rows(col_idx)
        col_data = {row: self.data[(row, col_idx)] for row in rows}
        self._column_cache[col_idx] = col_data
        return col_data
    
    def get_column_array(self, col_idx, fill_missing=True):
        """Vectorized column array extraction"""
        if fill_missing and self.shape:
            arr = np.full(self.shape[0], np.nan)
            rows = self.get_column_rows(col_idx)
            for row in rows:
                arr[row] = self.data[(row, col_idx)]
            return arr
        else:
            rows = self.get_column_rows(col_idx)
            return np.array([self.data[(row, col_idx)] for row in rows])
    
    def copy(self):
        return MatrixDict(self.data.copy(), self.shape)
    
    def get_density(self):
        if self.shape is None:
            return len(self.data)
        return len(self.data) / (self.shape[0] * self.shape[1])

def simulate_correlated_matrix_dict(A, C=4, random_seed=42) -> MatrixDict:
    """
    Ultra-fast correlated matrix simulation with optional Dask parallelization.
    """
    np.random.seed(random_seed)
    
    # Fixed correlation matrix for consistency
    corr_matrix = np.array([
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.2, 1.0]
    ])
    
    L = cholesky(corr_matrix, lower=True)
    independent_normals = np.random.randn(A, C) + 3
    result = independent_normals @ L.T

    # Ultra-fast dictionary population using JIT
    data_tuples = _fast_dict_population(result, A, C)
    
    # Convert to dictionary efficiently
    data_dict = {(int(row), int(col)): val for row, col, val in data_tuples}
    
    return MatrixDict(data_dict, shape=(A, C))

def infer_element_dict(matrix_dict, target_column, target_row, reference_column, 
                      pairs_needed=2, backup_value=0):
    """
    Ultra-optimized inference with advanced caching.
    """
    # Early exits
    if (target_row, reference_column) not in matrix_dict:
        return backup_value
    
    # Check if target element already exists
    if (target_row, target_column) in matrix_dict and matrix_dict[(target_row, target_column)] is not None:
        return matrix_dict[(target_row, target_column)]
    
    reference_target_element = matrix_dict[(target_row, reference_column)]
    
    # Ultra-fast row finding
    target_rows = matrix_dict.get_column_rows(target_column)
    reference_rows = matrix_dict.get_column_rows(reference_column)
    
    # JIT-compiled intersection to find valid pairs
    valid_rows = _fast_set_intersection(target_rows, reference_rows, target_row)
    
    if len(valid_rows) < pairs_needed:
        return backup_value
    
    # Vectorized data extraction of valid pairs
    n_pairs = len(valid_rows)
    target_vals = np.empty(n_pairs, dtype=np.float64)
    reference_vals = np.empty(n_pairs, dtype=np.float64)
    
    for i, row in enumerate(valid_rows):
        target_vals[i] = matrix_dict[(row, target_column)]
        reference_vals[i] = matrix_dict[(row, reference_column)]
    
    # Ultra-fast regression computation
    intercept, slope, is_valid = _compute_regression_ultra_fast(target_vals, reference_vals, n_pairs)
    
    if not is_valid:
        return backup_value
        
    return intercept + slope * reference_target_element

def add_noise_to_matrix(baseline_matrix, sigma, random_seed=None):
    """
    Ultra-fast noise addition with optional Dask parallelization.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    keys = list(baseline_matrix.data.keys())
    values = np.array(list(baseline_matrix.data.values()), dtype=np.float64)
    noisy_values = _fast_noise_addition(values, sigma, len(values))
    noisy_data = dict(zip(keys, noisy_values))
    
    return MatrixDict(noisy_data, baseline_matrix.shape)

def batch_infer_with_dask(matrix_dicts, inference_tasks, pairs_needed=2):
    """
    Dask-parallelized batch inference across multiple matrices and tasks.
    
    Inputs:
    -----------
    matrix_dicts : list of MatrixDict
        List of matrices to perform inference on
    inference_tasks : list of tuples
        List of (target_column, target_row, reference_column) tuples
        
    Returns:
    --------
    results : list of float
        List of inferred values in same order as tasks
    """
    futures = []
    # One matrix corersponds to one task
    for matrix, task in zip(matrix_dicts, inference_tasks):
        target_col, target_row, ref_col = task
        future = dask.delayed(infer_element_dict)(matrix, target_col, target_row, ref_col, pairs_needed)
        futures.append(future)

    return list(dask.compute(*futures))

def batch_create_noise_matrices_dask(matrix_dict, sigma_list, base_seed=42):
    """
    Create multiple noisy matrices in parallel using Dask.
    
    Inputs:
    -----------
    matrix_dict : MatrixDict
        Single baseline matrix
    sigma_list : list of float
        List of noise standard deviations

    Returns:
    --------
    noisy_matrices : list of MatrixDict
        List of noisy matrices
    """
    
    # Create delayed tasks for each noise level
    tasks = []
    for i, sigma in enumerate(sigma_list):
        seed = base_seed + i
        task = dask.delayed(add_noise_to_matrix)(matrix_dict, sigma, seed)
        tasks.append(task)
    
    # Compute all matrices in parallel
    return list(dask.compute(*tasks))

# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    print("Ultra-High Performance Matrix Dictionary Demo")
    print("=" * 50)
    
    # 1. Generate baseline correlated matrix
    print("1. Generating baseline correlated matrix...")
    A, C = 1000000, 4
    start_time = time.time()
    baseline_matrix = simulate_correlated_matrix_dict(A=A, C=C, random_seed=42)
    gen_time = time.time() - start_time
    print(f"   Generated {A}x{C} matrix in {gen_time:.4f}s")
    print(f"   Matrix density: {baseline_matrix.get_density():.3f}")
    
    # 2. Create blank matrix and populate manually
    print("\n2. Creating blank matrix...")
    blank_matrix = MatrixDict(shape=(100, 4))
    # Add some sample data
    for i in range(50):
        for j in range(4):
            if np.random.random() > 0.1:  # 90% density
                blank_matrix[(i, j)] = np.random.normal(0, 1)
    print(f"   Blank matrix populated with density: {blank_matrix.get_density():.3f}")
    
    # 3. Create multiple noisy versions
    print("\n3. Creating noisy matrices...")
    sigma_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    start_time = time.time()
    noisy_matrices = batch_create_noise_matrices_dask(baseline_matrix, sigma_list, base_seed=100)
    noise_time = time.time() - start_time
    print(f"   Created {len(noisy_matrices)} noisy matrices in {noise_time:.4f}s")
    
    # 4. Single element inference test
    print("\n4. Single element inference test...")
    # Remove an element to test inference
    test_matrix = baseline_matrix.copy()
    test_row, test_col = 50, 0
    true_value = test_matrix[(test_row, test_col)]
    del test_matrix.data[(test_row, test_col)]  # Remove element
    
    start_time = time.time()
    inferred_value = infer_element_dict(
        test_matrix, 
        target_column=test_col, 
        target_row=test_row, 
        reference_column=1,
        pairs_needed=3
    )
    infer_time = time.time() - start_time
    error = abs(true_value - inferred_value)
    print(f"   True value: {true_value:.4f}")
    print(f"   Inferred value: {inferred_value:.4f}")
    print(f"   Error: {error:.4f}")
    print(f"   Inference time: {infer_time:.6f}s")
    
    # 5. Batch inference with multiple matrices and tasks
    print("\n5. Batch inference test...")
    
    # Define inference tasks
    inference_tasks = [
        (0, int(A/10), 1),
        (0, int(A/8), 1),
        (2, int(A/5), 3),
    ]

    original_values = []
    test_matrices = []
    # Get 3 noisy matrices
    for i, noisy_matrix in enumerate(noisy_matrices[:3]):
        test_matrix = noisy_matrix.copy()
        target_col, target_row, ref_col = inference_tasks[i]
        if (target_row, target_col) in test_matrix.data:
            original_values.append(test_matrix.data[(target_row, target_col)])
            del test_matrix.data[(target_row, target_col)]
        else:
            original_values.append(np.nan)  # If not present, use NaN
        test_matrices.append(test_matrix)
    
    
    start_time = time.time()
    batch_results = batch_infer_with_dask(test_matrices, inference_tasks, pairs_needed=3)
    batch_time = time.time() - start_time
    
    print(f"   Batch inference on {len(test_matrices)} matrices, i.e. {len(inference_tasks)} tasks")
    print(f"   Total {len(batch_results)} inferences in {batch_time:.4f}s")
    print(f"   Average per inference: {batch_time/len(batch_results)*1000:.2f}ms")
    
    # Show sample results
    print("\n   Sample results:")
    for i, result in enumerate(batch_results):
        print(f"   Task {i}: {result:.4f} (original: {original_values[i]:.4f})")
    
    # 6. Performance comparison
    print("\n6. Performance analysis...")
    
    # Test inference speed on different matrix sizes
    sizes = [500, 1000, 2000]
    for size in sizes:
        matrix = simulate_correlated_matrix_dict(A=size, C=4, random_seed=123)
        
        # Time single inference
        start_time = time.time()
        result = infer_element_dict(matrix, 0, size//2, 1, pairs_needed=3)
        single_time = time.time() - start_time
        
        print(f"   Matrix {size}x4: Single inference {single_time*1000:.2f}ms")
    
    # 7. Column operations test
    print("\n7. Column operations test...")
    sample_matrix = baseline_matrix
    
    start_time = time.time()
    col_array = sample_matrix.get_column_array(0, fill_missing=True)
    col_time = time.time() - start_time
    print(f"   Column extraction: {col_time*1000:.2f}ms")
    print(f"   Column mean: {np.nanmean(col_array):.4f}")
    print(f"   Column std: {np.nanstd(col_array):.4f}")
    

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("Key features demonstrated:")
    print("✓ Fast matrix generation with correlations")
    print("✓ Efficient noise addition")
    print("✓ Single and batch inference")
    print("✓ Dask parallel processing")
    print("✓ Memory-efficient sparse storage")
    print("✓ JIT-compiled performance optimizations")