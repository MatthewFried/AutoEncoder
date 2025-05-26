import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, set_start_method, Manager
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA, SparsePCA, FactorAnalysis
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. UMAP initialization method will fall back to PCA.")
import pandas as pd
import os
import time
import uuid
import warnings
from scipy.stats import entropy
from scipy.linalg import svd
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ===== SETUP AND UTILITY FUNCTIONS =====

def setup_device(gpu_id=None):
    """Set up and return the appropriate device (GPU or CPU)."""
    if not torch.cuda.is_available():
        #print("CUDA not available. Using CPU.")
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        #print("No GPUs detected. Using CPU.")
        return torch.device('cpu')

    if gpu_id is not None:
        gpu_id = gpu_id % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        print(f"Using device: {device}")
        return device
    else:
        device = torch.device('cuda:0')
        print(f"Using default device: {device}")
        return device

def clean_gpu_memory(device=None):
    """Clean GPU memory to avoid fragmentation."""
    if device is not None and device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_available_datasets():
    """Check available datasets in the current directory."""
    datasets = {}
    for prefix in ['fashion_mnist', 'mnist', 'diabetes', 'cifar10', 'dsprites', 'wine']:
        x_train_path = f'{prefix}_x_train.npy'
        if os.path.exists(x_train_path):
            try:
                x_sample = np.load(x_train_path, mmap_mode='r')
                datasets[prefix] = {'available': True, 'x_shape': x_sample.shape, 'x_path': x_train_path}
                print(f"Dataset {prefix} available: {x_sample.shape[0]} samples, {x_sample.shape[1]} features")
            except Exception as e:
                print(f"Error loading {prefix}: {e}")
                datasets[prefix] = {'available': False, 'error': str(e)}
        else:
            print(f"Dataset {prefix} files not found")
            datasets[prefix] = {'available': False, 'error': 'Files not found'}
    return datasets

def sanitize_metrics(metrics):
    """Ensure metrics have valid numerical values."""
    sanitized = {}
    fallback_defaults = {
        'sparsity': 0.5,
        'modularity': 0.5,
        'factor_vae_score': 0.5,
        'sap_score': 0.5,
        'variance_ratio': 0.5,
        'mi_ksg': 0.5,
        'total_correlation': 0.5,
        'recon_error': 1.0
    }

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()

        if np.isnan(value) or np.isinf(value) or not (0 <= value <= 1):
            sanitized[key] = fallback_defaults.get(key, 0.5)
            print(f"Warning: {key} invalid ({value}), using fallback: {sanitized[key]}")
        else:
            sanitized[key] = value
    return sanitized

# ===== METRICS AND EVALUATION FUNCTIONS =====

def safe_mi_ksg_estimator(x, y, k=3):
    """Robust KSG mutual information estimator."""
    try:
        # Ensure inputs are NumPy arrays on CPU
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        x, y = x.flatten(), y.flatten()
        if len(x) < k + 1 or x.shape != y.shape:
            print("MI KSG: Invalid input shapes")
            return 0.5

        # Add small noise for numerical stability
        x = x + np.random.normal(0, 1e-10, x.shape)
        y = y + np.random.normal(0, 1e-10, y.shape)

        n_samples = x.shape[0]
        xy = np.column_stack([x, y])

        # Find nearest neighbors in joint space
        nn_joint = NearestNeighbors(metric='chebyshev').fit(xy)
        dist_joint = nn_joint.kneighbors(xy, k + 1)[0][:, k]

        # Find points within epsilon radius in marginal spaces
        nn_x = NearestNeighbors(metric='chebyshev').fit(x.reshape(-1, 1))
        nn_y = NearestNeighbors(metric='chebyshev').fit(y.reshape(-1, 1))

        nx = np.array([len(nn_x.radius_neighbors(x[i].reshape(1, -1), radius=dist_joint[i])[0]) for i in range(n_samples)])
        ny = np.array([len(nn_y.radius_neighbors(y[i].reshape(1, -1), radius=dist_joint[i])[0]) for i in range(n_samples)])

        # Ensure counts are at least 1
        nx = np.maximum(nx, 1)
        ny = np.maximum(ny, 1)

        # Calculate MI
        mi = np.mean(np.log(n_samples) + np.log(k) - np.log(nx) - np.log(ny))

        # Normalize and clamp
        return max(0.0, min(1.0, mi / np.log(n_samples)))
    except Exception as e:
        print(f"MI KSG error: {e}")
        return 0.5

def sparsity_score(k_matrix):
    """Improved sparsity score with numerical stability."""
    try:
        # Get device from input tensor
        device = k_matrix.device

        # Add small noise for numerical stability
        k_matrix = k_matrix + 1e-6 * torch.randn_like(k_matrix)

        l1_norm = torch.sum(torch.abs(k_matrix))
        l2_norm = torch.sqrt(torch.sum(k_matrix ** 2) + 1e-6)
        n_elements = float(torch.numel(k_matrix))

        # Create tensor on the same device
        n_elements_tensor = torch.tensor(n_elements, dtype=torch.float, device=device)

        sparsity = 1.0 - (l1_norm / (l2_norm * torch.sqrt(n_elements_tensor) + 1e-6))

        # Create min/max tensors on the same device
        min_val = torch.tensor(0.1, device=device)
        max_val = torch.tensor(0.9, device=device)
        sparsity_val = torch.clamp(sparsity, min_val, max_val)

        if torch.isnan(sparsity_val) or torch.isinf(sparsity_val):
            print(f"Sparsity is NaN or Inf")
            return torch.tensor(0.5, device=device)

        return sparsity_val

    except Exception as e:
        print(f"Sparsity error: {e}")
        # Ensure return value is on the same device
        return torch.tensor(0.5, device=device if device is not None else 'cpu')

def robust_total_correlation(z, num_factors, latent_dim, max_samples=5000):
    """Calculate total correlation between latent factors."""
    try:
        # Get the device of the input tensor
        device = z.device

        # Check for NaNs or Infs
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("TC Debug: Input contains NaN or Inf values!")
            z = torch.nan_to_num(z, nan=0.5, posinf=1.0, neginf=0.0)

        # Reshape correctly
        z_reshaped = z.view(-1, num_factors, latent_dim)
        n_samples = z_reshaped.shape[0]

        # Skip computation for num_factors=1 (TC is always 0 for single factor)
        if num_factors <= 1:
            print("TC Debug: Single factor, TC is 0")
            return torch.tensor(0.0, device=device)

        if n_samples < num_factors * 10:
            print(f"Total Correlation: Insufficient samples ({n_samples} < {num_factors * 10})")
            return torch.tensor(0.5, device=device)

        if n_samples > max_samples:
            indices = torch.randperm(n_samples, device=device)[:max_samples]
            z_reshaped = z_reshaped[indices]
            n_samples = max_samples

        # Min-max normalization (keeping on original device)
        z_min, _ = z_reshaped.min(dim=0, keepdim=True)
        z_max, _ = z_reshaped.max(dim=0, keepdim=True)
        z_range = z_max - z_min + 1e-6
        z_reshaped = (z_reshaped - z_min) / z_range

        # Add small noise to prevent exact zeros
        z_reshaped = z_reshaped + torch.randn_like(z_reshaped) * 1e-5

        # Move to CPU for histogram computation
        z_cpu = z_reshaped.detach().cpu()

        # Pairwise TC computation
        tc_scores = []
        # Use adaptive bin count based on sample size
        bin_count = min(30, max(10, int(np.sqrt(n_samples / 5))))
        #print(f"TC Debug: Using {bin_count} bins for histograms")

        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                try:
                    z_i = z_cpu[:, i, :].flatten()
                    z_j = z_cpu[:, j, :].flatten()

                    # Calculate histograms on CPU
                    hist_i, bin_edges_i = np.histogram(z_i.numpy(), bins=bin_count, range=(0.0, 1.0), density=True)
                    hist_j, bin_edges_j = np.histogram(z_j.numpy(), bins=bin_count, range=(0.0, 1.0), density=True)

                    # Remove zeros for log stability
                    hist_i = hist_i + 1e-10
                    hist_j = hist_j + 1e-10

                    # Normalize
                    hist_i = hist_i / np.sum(hist_i)
                    hist_j = hist_j / np.sum(hist_j)

                    # Calculate entropies
                    entropy_i = -np.sum(hist_i * np.log2(hist_i))
                    entropy_j = -np.sum(hist_j * np.log2(hist_j))

                    # Joint histogram - simple 2D binning
                    joint_hist, _, _ = np.histogram2d(
                        z_i.numpy(), z_j.numpy(),
                        bins=bin_count,
                        range=[[0, 1], [0, 1]]
                    )

                    # Normalize and handle zeros
                    joint_hist = joint_hist / np.sum(joint_hist) + 1e-10

                    # Joint entropy
                    joint_entropy = -np.sum(joint_hist * np.log2(joint_hist))

                    # MI calculation
                    mi = entropy_i + entropy_j - joint_entropy

                    # Normalize to [0, 1]
                    max_entropy = np.log2(bin_count)
                    mi_normalized = mi / max_entropy

                    tc_pair = max(0.0, min(0.95, mi_normalized))
                    tc_scores.append(tc_pair)

                except Exception as e:
                    print(f"TC Debug: Error in pair ({i},{j}): {e}")
                    continue

        if tc_scores:
            tc = np.mean(tc_scores)
            return torch.tensor(tc, device=device)  # Return on the original device

        # Gaussian approximation fallback
        #print("TC Debug: Using Gaussian approximation fallback")

        # Compute on GPU if possible
        try:
            z_flat = z_reshaped.reshape(n_samples, -1)

            # Add regularization for numerical stability
            eps = 1e-3 * torch.eye(z_flat.shape[1], device=device)

            # Compute covariance with explicit formula
            z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
            cov_matrix = (z_centered.T @ z_centered) / (n_samples - 1) + eps

            # Compute log determinant
            log_det_cov = torch.logdet(cov_matrix)

            # Compute marginal variances
            marginal_vars = torch.var(z_reshaped, dim=0, unbiased=True).flatten()
            marginal_vars = torch.clamp(marginal_vars, min=1e-5)
            log_det_marginals = torch.sum(torch.log(marginal_vars))

            # Calculate TC
            tc = 0.5 * (log_det_marginals - log_det_cov)

            # Scale to [0,1]
            tc_scaled = 0.95 * torch.tanh(tc / np.log(n_samples))
            tc_value = max(0.0, min(0.95, tc_scaled.item()))

            return torch.tensor(tc_value, device=device)

        except Exception as e:
            #print(f"TC Debug Gaussian fallback error: {e}")
            return torch.tensor(0.5, device=device)

    except Exception as e:
        print(f"TC error: {e}")
        return torch.tensor(0.5, device=device)

def robust_modularity_score(z, num_factors, latent_dim):
    """Modularity score for latent factors."""
    try:
        device = z.device
        z_reshaped = z.view(-1, num_factors, latent_dim)

        # Special case for single factor
        if num_factors <= 1:
            return torch.tensor(1.0, device=device)

        # Check for NaNs or Infs
        if torch.isnan(z_reshaped).any() or torch.isinf(z_reshaped).any():
            print("Modularity Debug: Input contains NaN or Inf values!")
            z_reshaped = torch.nan_to_num(z_reshaped, nan=0.5, posinf=1.0, neginf=0.0)

        modularity = 0.0
        count = 0

        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                z_i = z_reshaped[:, i, :].flatten()
                z_j = z_reshaped[:, j, :].flatten()

                # Robust normalization
                z_i_mean, z_i_std = z_i.mean(), z_i.std() + 1e-8
                z_j_mean, z_j_std = z_j.mean(), z_j.std() + 1e-8

                z_i = (z_i - z_i_mean) / z_i_std
                z_j = (z_j - z_j_mean) / z_j_std

                # Clip to prevent extreme values
                z_i = torch.clamp(z_i, -10.0, 10.0)
                z_j = torch.clamp(z_j, -10.0, 10.0)

                # Compute correlation
                corr = torch.abs(torch.mean(z_i * z_j))

                if not torch.isnan(corr) and not torch.isinf(corr):
                    corr_val = corr.item()
                    # Correlation indicates dependence, so we take (1 - correlation) as modularity
                    modularity += 1.0 - min(1.0, max(0.0, corr_val))
                    count += 1

        if count == 0:
            print("Modularity Debug: No valid correlations computed")
            return torch.tensor(0.5, device=device)

        result = modularity / count

        # Final sanity check
        if result < 0 or result > 1 or np.isnan(result) or np.isinf(result):
            print(f"Modularity Debug: Invalid final result: {result}")
            return torch.tensor(0.5, device=device)

        return torch.tensor(result, device=device)

    except Exception as e:
        print(f"Modularity error: {e}")
        return torch.tensor(0.5, device=device)

def robust_factor_vae_score(z, num_factors, latent_dim, n_samples=2000):
    """Improved Factor VAE score using ElasticNet."""
    try:
        # Ensure z is on CPU before NumPy conversion
        if isinstance(z, torch.Tensor):
            if z.device.type != 'cpu':
                z = z.detach().cpu()

        # Special case for single factor
        if num_factors <= 1:
            return 0.5  # Default value for single factor

        z_reshaped = z.view(-1, num_factors, latent_dim).detach().numpy()

        if z_reshaped.shape[0] < num_factors or z_reshaped.shape[2] != latent_dim:
            print("FactorVAE: Invalid shape")
            return 0.5

        if z_reshaped.shape[0] > n_samples:
            indices = np.random.choice(z_reshaped.shape[0], n_samples, replace=False)
            z_reshaped = z_reshaped[indices]

        # Standardize the data
        scaler = StandardScaler()
        z_reshaped = scaler.fit_transform(z_reshaped.reshape(z_reshaped.shape[0], -1)).reshape(z_reshaped.shape)

        scores = []
        for j in range(num_factors):
            for k in range(latent_dim):
                target = z_reshaped[:, j, k]
                # Get all data from other factors
                other = z_reshaped[:, [i for i in range(num_factors) if i != j], :].reshape(z_reshaped.shape[0], -1)

                if other.size == 0:
                    continue

                # Split data for training and testing
                X_train, X_test, y_train, y_test = train_test_split(other, target, test_size=0.2, random_state=42)

                # Train ElasticNet model
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
                model.fit(X_train, y_train)

                # Score is predictability (R²)
                score = model.score(X_test, y_test)

                # Higher independence (lower predictability) is better
                scores.append(max(0.0, min(1.0, 1.0 - score)))

        return np.mean(scores) if scores else 0.5
    except Exception as e:
        print(f"FactorVAE error: {e}")
        return 0.5

def robust_sap_score(z, x_data, num_factors, latent_dim):
    """Improved SAP score using mutual information."""
    try:
        # Ensure inputs are on CPU
        if isinstance(z, torch.Tensor):
            if z.device.type != 'cpu':
                z = z.detach().cpu()
        if isinstance(x_data, torch.Tensor):
            if x_data.device.type != 'cpu':
                x_data = x_data.detach().cpu()

        # Special case for single factor
        if num_factors <= 1:
            return 0.5  # Default value for single factor

        z_reshaped = z.view(-1, num_factors, latent_dim).numpy()
        x_np = x_data.numpy().reshape(x_data.shape[0], -1)

        # Use a subset of features as proxies for true factors
        n_proxy = min(50, x_np.shape[1])
        proxy_indices = np.random.choice(x_np.shape[1], n_proxy, replace=False)

        sap_scores = []
        for j in range(num_factors):
            for k in range(latent_dim):
                latent = z_reshaped[:, j, k]

                # Calculate mutual information with each proxy
                mi_scores = []
                for p in proxy_indices:
                    if x_np[:, p].var() > 1e-6:  # Skip if variance is too low
                        mi = mutual_info_regression(latent.reshape(-1, 1), x_np[:, p])[0]
                        mi_scores.append(mi)

                if len(mi_scores) > 1:
                    # Sort MI scores
                    sorted_mi = sorted(mi_scores, reverse=True)
                    # Gap between highest and second highest MI score indicates disentanglement
                    gap = (sorted_mi[0] - sorted_mi[1]) / (sorted_mi[0] + 1e-8)
                    sap_scores.append(max(0.0, min(1.0, gap)))

        return np.mean(sap_scores) if sap_scores else 0.5
    except Exception as e:
        print(f"SAP score error: {e}")
        return 0.5

def encode_data(x_data, k_matrix):
    """Encode data with k_matrix."""
    device = x_data.device

    # Ensure k_matrix is on the same device as x_data
    if k_matrix.device != device:
        k_matrix = k_matrix.to(device)

    num_factors = k_matrix.shape[0]
    batch_size = 1024
    all_z = []

    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i + batch_size]

            # Process each factor separately for memory efficiency
            batch_z_factors = []
            for j in range(num_factors):
                z_factor = torch.matmul(batch_x, k_matrix[j])
                batch_z_factors.append(z_factor)

            batch_z = torch.stack(batch_z_factors, dim=1)
            all_z.append(batch_z)

    # Concatenate all batches
    z = torch.cat(all_z, dim=0)
    return z

def evaluate_k_matrix(x_data, k_matrix, num_factors, latent_dim, device):
    """Evaluate K matrix with comprehensive metrics."""
    try:
        # Ensure consistent device placement
        x_data = x_data.to(device)
        k_matrix = k_matrix.to(device)

        # Initialize metrics dictionary
        metrics = {
            'recon_error': 1.0,
            'mi_ksg': 0.5,
            'sparsity': 0.5,
            'total_correlation': 0.5,
            'modularity': 0.5,
            'factor_vae_score': 0.5,
            'sap_score': 0.5,
            'variance_ratio': 0.5
        }

        # Special case for num_factors=1
        if num_factors <= 1:
            print("Evaluation: Single factor case, special handling")
            metrics['total_correlation'] = 0.0  # No correlation with self
            metrics['mi_ksg'] = 0.0  # No mutual information with self
            metrics['modularity'] = 1.0  # Fully modular with self

        # Check for NaNs or Infs in input
        if torch.isnan(k_matrix).any() or torch.isinf(k_matrix).any():
            print("Evaluation Debug: K matrix contains NaN or Inf values!")
            k_matrix = torch.nan_to_num(k_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # Normalize k_matrix
        k_norm = torch.norm(k_matrix.view(num_factors, -1, latent_dim), dim=1, keepdim=True)
        k_matrix = k_matrix / (k_norm + 1e-8)

        # Encode data
        z = encode_data(x_data, k_matrix)

        # Reconstruction
        batch_size = 1024
        all_recon = []

        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch_x = x_data[i:i + batch_size]
                batch_indices = slice(i, min(i + batch_size, len(x_data)))
                batch_z = z[batch_indices]

                # Reconstruct incrementally
                batch_recon = torch.zeros_like(batch_x)

                for j in range(num_factors):
                    # Get the z values for this factor
                    z_j = batch_z[:, j]
                    # Add the reconstruction for this factor
                    batch_recon += torch.matmul(z_j, k_matrix[j].T)

                all_recon.append(batch_recon)

        # Concatenate all batches
        reconstructed = torch.cat(all_recon, dim=0)

        # Calculate reconstruction error
        recon_error = F.mse_loss(reconstructed, x_data)
        data_var = torch.var(x_data)
        metrics['recon_error'] = min(1.0, recon_error.item() / (data_var.item() + 1e-8))

        # Sample a subset for metric computation
        sample_size = min(2000, z.shape[0])
        sample_indices = torch.randperm(z.shape[0], device=device)[:sample_size]
        z_sampled = z[sample_indices]
        x_sampled = x_data[sample_indices]

        # If single factor, just compute remaining metrics
        if num_factors <= 1:
            metrics['sparsity'] = sparsity_score(k_matrix).item()

            try:
                # Move to CPU for SVD
                x_flat = x_sampled.reshape(sample_size, -1).detach().cpu().numpy()
                z_flat = z_sampled.reshape(sample_size, -1).detach().cpu().numpy()

                # Handle NaNs
                if np.isnan(x_flat).any() or np.isinf(x_flat).any():
                    x_flat = np.nan_to_num(x_flat, nan=0.0)
                if np.isnan(z_flat).any() or np.isinf(z_flat).any():
                    z_flat = np.nan_to_num(z_flat, nan=0.0)

                # Add noise for stability
                x_flat = x_flat + 1e-8 * np.random.randn(*x_flat.shape)
                z_flat = z_flat + 1e-8 * np.random.randn(*z_flat.shape)

                # Compute SVD for data
                u_x, s_x, vt_x = svd(x_flat, full_matrices=False)
                total_variance_x = np.sum(s_x ** 2)

                # Compute SVD for latent
                u_z, s_z, vt_z = svd(z_flat, full_matrices=False)
                total_variance_z = np.sum(s_z ** 2)

                # Variance ratio
                metrics['variance_ratio'] = min(0.95, total_variance_z / (total_variance_x + 1e-6))
            except Exception as e:
                print(f"Variance ratio calculation error: {e}")
                metrics['variance_ratio'] = 0.5

            return sanitize_metrics(metrics)

        # For multi-factor case, compute all metrics
        try:
            # Compute mutual information between factors using KSG estimator
            # Move to CPU for KSG computation
            z_cpu = z_sampled.detach().cpu()
            mi_scores = []

            for i in range(num_factors):
                for j in range(i + 1, num_factors):
                    # Take the first dimension of each factor for simplicity
                    mi_scores.append(safe_mi_ksg_estimator(
                        z_cpu[:, i, 0].flatten(),
                        z_cpu[:, j, 0].flatten()
                    ))

            metrics['mi_ksg'] = np.mean(mi_scores) if mi_scores else 0.5
        except Exception as e:
            print(f"MI KSG calculation error: {e}")
            metrics['mi_ksg'] = 0.5

        try:
            # Compute sparsity
            metrics['sparsity'] = sparsity_score(k_matrix).item()
        except Exception as e:
            print(f"Sparsity calculation error: {e}")
            metrics['sparsity'] = 0.5

        try:
            # Compute total correlation
            tc_result = robust_total_correlation(z_sampled, num_factors, latent_dim)
            metrics['total_correlation'] = tc_result.item() if isinstance(tc_result, torch.Tensor) else tc_result
        except Exception as e:
            print(f"Total correlation calculation error: {e}")
            metrics['total_correlation'] = 0.5

        try:
            # Compute modularity
            mod_result = robust_modularity_score(z_sampled, num_factors, latent_dim)
            metrics['modularity'] = mod_result.item() if isinstance(mod_result, torch.Tensor) else mod_result
        except Exception as e:
            print(f"Modularity calculation error: {e}")
            metrics['modularity'] = 0.5

        try:
            # Compute FactorVAE score (needs CPU)
            z_cpu = z_sampled.detach().cpu()
            metrics['factor_vae_score'] = robust_factor_vae_score(z_cpu, num_factors, latent_dim)
        except Exception as e:
            print(f"FactorVAE score calculation error: {e}")
            metrics['factor_vae_score'] = 0.5

        try:
            # Compute SAP score (needs CPU)
            z_cpu = z_sampled.detach().cpu()
            x_cpu = x_sampled.detach().cpu()
            metrics['sap_score'] = robust_sap_score(z_cpu, x_cpu, num_factors, latent_dim)
        except Exception as e:
            print(f"SAP score calculation error: {e}")
            metrics['sap_score'] = 0.5

        try:
            # Compute variance ratio
            x_flat = x_sampled.reshape(sample_size, -1).detach().cpu().numpy()
            z_flat = z_sampled.reshape(sample_size, -1).detach().cpu().numpy()

            # Handle NaNs
            if np.isnan(x_flat).any() or np.isinf(x_flat).any():
                x_flat = np.nan_to_num(x_flat, nan=0.0)
            if np.isnan(z_flat).any() or np.isinf(z_flat).any():
                z_flat = np.nan_to_num(z_flat, nan=0.0)

            # Add noise for stability
            x_flat = x_flat + 1e-8 * np.random.randn(*x_flat.shape)
            z_flat = z_flat + 1e-8 * np.random.randn(*z_flat.shape)

            # Compute SVD
            u_x, s_x, vt_x = svd(x_flat, full_matrices=False)
            total_variance_x = np.sum(s_x ** 2)

            u_z, s_z, vt_z = svd(z_flat, full_matrices=False)
            total_variance_z = np.sum(s_z ** 2)

            metrics['variance_ratio'] = min(0.95, total_variance_z / (total_variance_x + 1e-6))
        except Exception as e:
            print(f"Variance ratio calculation error: {e}")
            metrics['variance_ratio'] = 0.5

        return sanitize_metrics(metrics)

    except Exception as e:
        print(f"Evaluation error: {e}")
        return sanitize_metrics(metrics)


def safe_mine_mutual_information(z, num_factors, latent_dim):
    """Mutual information estimation using correlation."""
    try:
        # Always work with tensors on the original device
        device = z.device

        if z.shape[0] < 100:
            print("MINE: Insufficient samples")
            return torch.tensor(0.5, device=device)

        # Handle single factor case
        if num_factors <= 1:
            return torch.tensor(0.0, device=device)

        # Reshape for proper calculation
        z_reshaped = z.view(-1, num_factors, latent_dim)
        mi_scores = []

        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                # Extract factors and standardize
                z_i = z_reshaped[:, i, :].flatten()
                z_j = z_reshaped[:, j, :].flatten()

                z_i = (z_i - z_i.mean()) / (z_i.std() + 1e-8)
                z_j = (z_j - z_j.mean()) / (z_j.std() + 1e-8)

                # Calculate correlation and convert to MI estimate
                cross_corr = torch.abs(torch.mean(z_i * z_j))
                mi_scores.append(0.5 * (1.0 + torch.tanh(cross_corr)).item())

        return np.mean(mi_scores) if mi_scores else 0.5
    except Exception as e:
        print(f"MINE error: {e}")
        return 0.5

def create_kernel_pca_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using Kernel PCA."""
    try:
        # Always move to CPU for sklearn operations
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 5000:
            indices = np.random.choice(x_np.shape[0], 5000, replace=False)
            x_np = x_np[indices]

        # Standardize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_np)

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_scaled.shape))

        # Set gamma based on variance
        gamma = 1.0 / (x_scaled.var() * x_scaled.shape[1]) if x_scaled.var() > 0 else 0.1

        # Run Kernel PCA
        kpca = KernelPCA(
            n_components=total_components,
            kernel='rbf',
            gamma=gamma,
            fit_inverse_transform=True,
            random_state=42
        )
        transformed = kpca.fit_transform(x_scaled)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Project back to input space
                k_comp = np.zeros((x_np.shape[1], latent_dim))
                actual_dim = end_idx - start_idx

                # Approximation of kernel eigenvectors in input space
                k_comp[:, :actual_dim] = np.dot(x_scaled.T, transformed[:, start_idx:end_idx])

                # If we don't have enough components, pad with random values
                if actual_dim < latent_dim:
                    k_comp[:, actual_dim:] = np.random.randn(x_np.shape[1], latent_dim - actual_dim)

                k = torch.tensor(k_comp, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"KernelPCA initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_sparse_dictionary_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using Sparse Dictionary Learning."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Run Dictionary Learning
        dict_learning = DictionaryLearning(
            n_components=total_components,
            alpha=1.0,
            max_iter=200,
            random_state=42
        )
        dict_learning.fit(x_np)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use dictionary components
                k = torch.tensor(dict_learning.components_[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_np.shape[1], latent_dim - k.shape[1])
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"SparseDictionary initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_clustered_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using feature clustering."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(x_np.T)
        dist_matrix = 1 - np.abs(np.nan_to_num(corr_matrix))

        # Determine number of clusters
        n_clusters = min(num_factors, x_np.shape[1])

        # Run KMeans
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(dist_matrix)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Get indices for this cluster
            if i < n_clusters:
                cluster_indices = np.where(cluster_labels == i)[0]
            else:
                cluster_indices = np.array([])

            # Initialize k matrix
            k = torch.zeros(x_np.shape[1], latent_dim)

            if len(cluster_indices) > 0:
                # Set values for features in this cluster
                for j in range(latent_dim):
                    # Select a subset of the cluster features for each latent dimension
                    indices = cluster_indices[np.random.choice(
                        len(cluster_indices),
                        max(1, len(cluster_indices) // latent_dim),
                        replace=False
                    )]
                    k[indices, j] = 1.0

                # Add small noise for stability
                k = k + torch.randn_like(k) * 0.01
            else:
                # If no features in cluster, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"Clustered initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_ensemble_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using an ensemble of methods."""
    try:
        # Define methods and weights
        methods = [
            create_pca_k_matrix,
            create_svd_k_matrix,
            create_nmf_k_matrix,
            create_ica_k_matrix
        ]
        weights = [0.3, 0.3, 0.2, 0.2]

        # Apply each method
        k_matrices = []
        for method in methods:
            try:
                k = method(x_data, num_factors, latent_dim, device)
                k_matrices.append(k)
            except Exception as e:
                print(f"Error in ensemble method {method.__name__}: {e}")
                # Use random as fallback
                k_matrices.append(create_random_k_matrix(x_data, num_factors, latent_dim, device))

        # Weighted sum
        ensemble_k = sum(w * k for w, k in zip(weights, k_matrices))

        # Orthogonalize factors
        for i in range(num_factors):
            q, r = torch.linalg.qr(ensemble_k[i])
            ensemble_k[i] = q[:, :latent_dim]

        return ensemble_k
    except Exception as e:
        print(f"Ensemble initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_sparse_pca_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using Sparse PCA."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Run Sparse PCA
        sparse_pca = SparsePCA(
            n_components=total_components,
            alpha=1.0,
            random_state=42
        )
        sparse_pca.fit(x_np)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use sparse PCA components
                k = torch.tensor(sparse_pca.components_[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_np.shape[1], latent_dim - k.shape[1])
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"SparsePCA initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_factor_analysis_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using Factor Analysis."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Run Factor Analysis
        fa = FactorAnalysis(
            n_components=total_components,
            random_state=42
        )
        fa.fit(x_np)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use Factor Analysis components
                k = torch.tensor(fa.components_[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_np.shape[1], latent_dim - k.shape[1])
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"FactorAnalysis initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

# ===== K MATRIX INITIALIZATION METHODS =====

# New UMAP method
def create_umap_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using UMAP."""
    try:
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to PCA")
            return create_pca_k_matrix(x_data, num_factors, latent_dim, device)

        # Move to CPU for UMAP processing
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 5000:
            indices = np.random.choice(x_np.shape[0], 5000, replace=False)
            x_np = x_np[indices]

        # Standardize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_np)

        # Determine total components
        total_components = min(num_factors * latent_dim, min(x_scaled.shape))

        # Run UMAP - reduce to total components first
        reducer = umap.UMAP(
            n_components=total_components,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        embedding = reducer.fit_transform(x_scaled)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Train a linear model to map from input to UMAP embedding
                # This gives us the transformation matrix (K matrix)
                k_components = np.zeros((x_np.shape[1], latent_dim))
                actual_dim = end_idx - start_idx

                # For each UMAP dimension, train a linear model
                for j in range(actual_dim):
                    umap_dim = embedding[:, start_idx + j]
                    model = ElasticNet(alpha=0.01, l1_ratio=0.5)
                    model.fit(x_scaled, umap_dim)
                    # Extract weights as K matrix components
                    k_components[:, j] = model.coef_

                # If we don't have enough components, pad with random values
                if actual_dim < latent_dim:
                    k_components[:, actual_dim:] = np.random.randn(x_np.shape[1], latent_dim - actual_dim)

                k = torch.tensor(k_components, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"UMAP initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

# VAE class for initialization
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def create_vae_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using VAE."""
    try:
        # Reshape input data
        x_reshaped = x_data.reshape(x_data.shape[0], -1)
        input_dim = x_reshaped.shape[1]

        # Sample for efficiency
        if x_reshaped.shape[0] > 5000:
            indices = torch.randperm(x_reshaped.shape[0])[:5000]
            x_sampled = x_reshaped[indices].to(device)
        else:
            x_sampled = x_reshaped.to(device)

        # Scale data to [-1, 1] for tanh activation
        x_min = torch.min(x_sampled, dim=0, keepdim=True)[0]
        x_max = torch.max(x_sampled, dim=0, keepdim=True)[0]
        x_range = x_max - x_min + 1e-6
        x_scaled = 2 * ((x_sampled - x_min) / x_range) - 1

        # Train separate VAEs for each factor
        k_matrices = []

        for i in range(num_factors):
            # Define and train VAE
            hidden_dim = min(512, input_dim * 2)
            vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
            optimizer = optim.Adam(vae.parameters(), lr=0.001)

            # Train for a limited number of epochs
            vae.train()
            batch_size = 256
            num_epochs = 10  # Keep this low for efficiency

            for epoch in range(num_epochs):
                total_loss = 0
                dataset = TensorDataset(x_scaled)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for batch in dataloader:
                    batch_x = batch[0]
                    optimizer.zero_grad()

                    # Forward pass
                    recon_x, mu, log_var = vae(batch_x)

                    # Reconstruction loss
                    recon_loss = F.mse_loss(recon_x, batch_x)

                    # KL divergence
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                    # Total loss
                    loss = recon_loss + 0.01 * kl_loss

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            # Extract encoder weights
            # For a VAE, the K matrix is related to the encoder weights
            with torch.no_grad():
                # Extract weights from encoder
                encoder_layers = [m for m in vae.encoder if isinstance(m, nn.Linear)]
                mu_layer = vae.mu

                # Compute the effective K matrix using the trained weights
                # This is an approximation of the encoder transformation
                k_base = torch.zeros(input_dim, latent_dim, device=device)

                # Forward pass through each layer to calculate the effective transformation
                # Starting with identity
                identity = torch.eye(input_dim, device=device)
                h = identity

                # Apply transformations
                for layer in encoder_layers:
                    h = F.leaky_relu(layer(h), 0.2)

                # Final transformation to mu
                k = mu_layer(h)

                # Normalize columns
                k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
                k_matrices.append(k)

        # Stack matrices and return
        result = torch.stack(k_matrices)
        return result
    except Exception as e:
        print(f"VAE initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_pca_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using PCA."""
    try:
        # Always move to CPU for sklearn operations
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Calculate PCA
        pca = PCA(n_components=total_components, random_state=42)
        pca.fit(x_np)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Get components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use PCA components
                k = torch.tensor(pca.components_[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_np.shape[1], latent_dim - k.shape[1])
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to specified device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"PCA initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_svd_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using SVD."""
    try:
        # Reshape input data
        x_data_reshaped = x_data.reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_data_reshaped.shape[0] > 10000:
            indices = torch.randperm(x_data_reshaped.shape[0])[:10000]
            x_sampled = x_data_reshaped[indices].to(device)
        else:
            x_sampled = x_data_reshaped.to(device)

        # Compute SVD on device
        U, S, V = torch.svd(x_sampled)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, V.shape[1])

            if end_idx > start_idx:
                # Use SVD components
                k = V[:, start_idx:end_idx]
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_data_reshaped.shape[1], latent_dim, device=device)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_data_reshaped.shape[1], latent_dim - k.shape[1], device=device)
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack the matrices
        result = torch.stack(k_matrices)
        return result
    except Exception as e:
        print(f"SVD initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_nmf_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using NMF."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # NMF requires non-negative values
        x_np = x_np - x_np.min() + 1e-6

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Run NMF
        nmf = NMF(n_components=total_components, init='nndsvd', max_iter=1000, random_state=42)
        W = nmf.fit_transform(x_np)
        H = nmf.components_

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use NMF components
                k = torch.tensor(H[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization (non-negative)
                k = torch.abs(torch.randn(x_np.shape[1], latent_dim))

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.abs(torch.randn(x_np.shape[1], latent_dim - k.shape[1]))
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"NMF initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_ica_k_matrix(x_data, num_factors, latent_dim, device):
    """Initialize K matrix using ICA."""
    try:
        # Move to CPU for sklearn
        x_np = x_data.cpu().numpy().reshape(x_data.shape[0], -1)

        # Sample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_np = x_np[indices]

        # Determine number of components
        total_components = min(num_factors * latent_dim, min(x_np.shape))

        # Run ICA
        ica = FastICA(n_components=total_components, random_state=42, max_iter=500)
        ica.fit(x_np)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            # Extract components for this factor
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                # Use ICA components
                k = torch.tensor(ica.components_[start_idx:end_idx].T, dtype=torch.float32)
            else:
                # If we run out of components, use random initialization
                k = torch.randn(x_np.shape[1], latent_dim)

            # If we don't have enough components, pad with random values
            if k.shape[1] < latent_dim:
                padding = torch.randn(x_np.shape[1], latent_dim - k.shape[1])
                k = torch.cat([k, padding], dim=1)

            # Normalize columns
            k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-6)
            k_matrices.append(k)

        # Stack and move to device
        result = torch.stack(k_matrices).to(device)
        return result
    except Exception as e:
        print(f"ICA initialization error: {e}")
        return create_random_k_matrix(x_data, num_factors, latent_dim, device)

def create_random_k_matrix(x_data, num_factors, latent_dim, device):
    """Create random orthogonal K matrices."""
    try:
        # Get input feature dimension
        n_features = x_data.shape[1]

        # Initialize on the correct device
        k_matrices = []

        for i in range(num_factors):
            # Create random matrix
            k = torch.randn(n_features, latent_dim, device=device)

            # Make orthogonal to previous factors
            for prev_k in k_matrices:
                k = k - torch.mm(prev_k, torch.mm(prev_k.t(), k))

            # QR decomposition for orthogonalization
            if torch.linalg.matrix_rank(k) > 0:  # Check if matrix is not all zeros
                q, r = torch.linalg.qr(k)
                k = q[:, :latent_dim]
            else:
                # If rank is 0, just use random normalized matrix
                k = torch.randn(n_features, latent_dim, device=device)
                k = k / (torch.norm(k, dim=0, keepdim=True) + 1e-8)

            k_matrices.append(k)

        return torch.stack(k_matrices)
    except Exception as e:
        print(f"Random initialization error: {e}")
        # Ultimate fallback
        return torch.randn(num_factors, x_data.shape[1], latent_dim, device=device)

# ===== REFINE K MATRIX =====

def refine_k_matrix(x_data, k_matrix, num_factors, latent_dim, device, epochs=100):
    """Refine K matrix with robust optimization."""
    try:
        #print(f"Refining K matrix on {device}...")

        # Make a copy for training and ensure it's on the correct device
        k_matrix = k_matrix.clone().to(device).requires_grad_(True)

        # Setup optimizer with conservative learning rate
        optimizer = optim.Adam([k_matrix], lr=5e-5, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

        # Setup data loading
        batch_size = min(1024, len(x_data))
        dataset = TensorDataset(x_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Track best matrix seen so far
        best_k_matrix = k_matrix.clone().detach()
        best_loss = float('inf')
        patience, patience_counter = 30, 0

        for epoch in range(epochs):
            recon_loss_epoch = 0.0
            tc_loss_epoch = 0.0
            ortho_loss_epoch = 0.0
            batches_processed = 0

            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                optimizer.zero_grad()

                # Check input shape
                if x.dim() == 1:  # If 1D tensor, reshape to 2D
                    x = x.unsqueeze(0)

                # Reshape k_matrix for computation
                try:
                    k_reshaped = k_matrix.view(num_factors, -1, latent_dim)
                except RuntimeError as e:
                    #print(f"Refinement error: shape issue with k_matrix: {e}")
                    #print(f"k_matrix shape: {k_matrix.shape}, num_factors: {num_factors}, latent_dim: {latent_dim}")
                    return k_matrix.detach()

                # Compute latent representations
                z_factors = []
                for j in range(num_factors):
                    try:
                        z_factor = torch.matmul(x, k_reshaped[j])
                        z_factors.append(z_factor)
                    except RuntimeError as e:
                        #print(f"Error in factor {j} computation: {e}")
                        #print(f"x shape: {x.shape}, k_reshaped[{j}] shape: {k_reshaped[j].shape}")
                        return k_matrix.detach()

                z = torch.stack(z_factors, dim=1)

                # Compute reconstruction
                recon = torch.zeros_like(x)
                for j in range(num_factors):
                    recon += torch.matmul(z_factors[j], k_reshaped[j].T)

                # Basic reconstruction loss
                recon_loss = F.mse_loss(recon, x)

                # Additional losses for better disentanglement

                # Variance penalty - encourage each latent dimension to have variance
                z_var = torch.var(z, dim=0).mean()
                variance_penalty = 0.1 * torch.clamp(1.0 - z_var, min=0.0)

                # Sparsity loss - encourage sparse k_matrix
                sparsity_loss = 0.01 * torch.mean(torch.abs(k_matrix))

                # Orthogonality loss - encourage factors to be orthogonal
                ortho_loss = 0.0
                for i in range(num_factors):
                    for j in range(i + 1, num_factors):
                        ortho_loss += torch.norm(torch.mm(k_reshaped[i].t(), k_reshaped[j]))
                ortho_loss = 0.01 * ortho_loss

                # Total correlation loss - encourage independence within factors
                tc_loss = 0.0
                z_reshaped = z.view(-1, num_factors, latent_dim)

                for i in range(num_factors):
                    # Calculate correlation matrix for each factor's dimensions
                    z_factor = z_reshaped[:, i, :]
                    z_centered = z_factor - z_factor.mean(0, keepdim=True)
                    cov = torch.mm(z_centered.t(), z_centered) / (z_centered.shape[0] - 1)
                    # Normalize to get correlation
                    var = torch.diag(cov).view(-1, 1)
                    corr = cov / torch.sqrt(var * var.t() + 1e-8)
                    # Sum absolute off-diagonal elements
                    tc_loss += torch.sum(torch.abs(corr * (1 - torch.eye(latent_dim, device=device))))

                tc_loss = 0.01 * tc_loss

                # Modularity loss - encourage between-factor independence
                modularity_loss = 0.0
                for i in range(num_factors):
                    for j in range(i + 1, num_factors):
                        z_i = z_reshaped[:, i, :]
                        z_j = z_reshaped[:, j, :]
                        z_i_centered = z_i - z_i.mean(0, keepdim=True)
                        z_j_centered = z_j - z_j.mean(0, keepdim=True)
                        cross_corr = torch.mm(z_i_centered.t(), z_j_centered) / (z_i_centered.shape[0] - 1)
                        modularity_loss += torch.norm(cross_corr)

                modularity_loss = 0.01 * modularity_loss

                # Total loss
                total_loss = (3.0 * recon_loss +
                             variance_penalty +
                             sparsity_loss +
                             ortho_loss +
                             tc_loss +
                             modularity_loss)

                # Check for numerical stability
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN or Inf loss detected at epoch {epoch}, batch {batches_processed}")
                    continue

                # Backward and optimize
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([k_matrix], max_norm=1.0)
                optimizer.step()

                # Track metrics
                recon_loss_epoch += recon_loss.item()
                tc_loss_epoch += tc_loss.item() if isinstance(tc_loss, torch.Tensor) else tc_loss
                ortho_loss_epoch += ortho_loss.item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
                batches_processed += 1

                # Clean up to prevent memory fragmentation
                del x, z, z_factors, recon
                clean_gpu_memory(device)

            # Average losses
            if batches_processed > 0:
                recon_loss_epoch /= batches_processed
                tc_loss_epoch /= batches_processed
                ortho_loss_epoch /= batches_processed

                # Update learning rate
                scheduler.step(recon_loss_epoch)

                # Track best model
                if recon_loss_epoch < best_loss:
                    best_loss = recon_loss_epoch
                    best_k_matrix = k_matrix.clone().detach()
                    patience_counter = 0
                    #print(f"Epoch {epoch}: New best loss: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        #print(f"Early stopping at epoch {epoch}")
                        break


        # Normalize the best matrix
        k_norm = torch.norm(best_k_matrix.view(num_factors, -1, latent_dim), dim=1, keepdim=True)
        k_matrix_normalized = best_k_matrix / (k_norm + 1e-6)

        # Clean up
        clean_gpu_memory(device)

        return k_matrix_normalized.detach()

    except Exception as e:
        #print(f"Refinement error: {e}")
        # Return the original matrix as fallback
        return k_matrix.detach()

# ===== PROCESS SINGLE METHOD =====

def process_method(gpu_id, method_name, method_func, x_data, num_factors_list, latent_dim_list, result_queue, log_queue=None, dataset_name=None):
    """Process a single method on a specified GPU."""
    try:
        # Set up device
        device = setup_device(gpu_id)

        # Function to log
        def log_message(msg):
            if log_queue:
                full_msg = f"[{method_name}] {msg}"
                log_queue.put(full_msg)
            else:
                print(f"[{method_name}] {msg}")

        # Track results for this method
        method_scores = []

        # Maximum time allowed for processing a single configuration
        max_time_per_config = 15 * 60  # 15 minutes

        # Try each combination of factors and dimensions
        for num_factors in num_factors_list:
            for latent_dim in latent_dim_list:
                try:
                    config_info = f"[Dataset: {dataset_name}, Factors: {num_factors}, Dims: {latent_dim}]"
                    
                    # Clean GPU memory before starting
                    clean_gpu_memory(device)

                    # Start timer for timeout
                    start_time = time.time()

                    # Use a subset for initialization (faster)
                    sample_size = min(5000, len(x_data))
                    indices = torch.randperm(len(x_data))[:sample_size]
                    x_sample = x_data[indices].to(device)

                    k_matrix = method_func(x_sample, num_factors, latent_dim, device)

                    # Check for timeout
                    if time.time() - start_time > max_time_per_config:
                        log_message(f"{config_info} Initialization timed out")
                        continue

                    k_refined = refine_k_matrix(x_sample, k_matrix, num_factors, latent_dim, device, epochs=100)

                    # Check for timeout
                    if time.time() - start_time > max_time_per_config:
                        log_message(f"{config_info} Refinement timed out")
                        continue

                    metrics = evaluate_k_matrix(x_sample, k_refined, num_factors, latent_dim, device)

                    # Calculate combined score
                    combined_score = (
                        (1.0 - metrics['mi_ksg']) * 0.2 +
                        metrics['modularity'] * 0.2 +
                        (1.0 - metrics['total_correlation']) * 0.2 +
                        metrics['factor_vae_score'] * 0.2 +
                        metrics['sap_score'] * 0.2
                    )

                    # Move k_matrix to CPU for storage
                    k_refined_cpu = k_refined.detach().cpu()

                    # Add to results
                    result = {
                        'method': method_name,
                        'num_factors': num_factors,
                        'latent_dim': latent_dim,
                        'metrics': metrics,
                        'combined_score': combined_score,
                        'k_matrix': k_refined_cpu
                    }

                    method_scores.append(result)

                    # Log results with context information
                    log_message(f"{config_info} Combined score: {combined_score:.4f}")
                    log_message(f"{config_info} Metrics: {metrics}")

                    # Clean up
                    clean_gpu_memory(device)

                except Exception as e:
                    log_message(f"Error in {dataset_name}, {num_factors} factors, {latent_dim} dims: {e}")
                    clean_gpu_memory(device)

        # Put results in queue
        result_queue.put((method_name, method_scores))
        log_message(f"Completed all configurations for {dataset_name}")

    except Exception as e:
        log_message(f"Process method error for {dataset_name}: {e}")
        # Ensure we put something in the queue to avoid hanging
        result_queue.put((method_name, []))

    finally:
        # Final cleanup
        clean_gpu_memory(device)

def log_handler(log_queue, log_file=None):
    """Process log messages from the queue."""
    try:
        if log_file:
            f = open(log_file, 'w')

        while True:
            message = log_queue.get()
            if message == "DONE":
                break

            # Format message with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] {message}"

            # Print to console
            print(formatted)

            # Write to file if provided
            if log_file:
                f.write(formatted + "\n")
                f.flush()

    except Exception as e:
        print(f"Error in log handler: {e}")

    finally:
        if log_file and 'f' in locals():
            f.close()

# ===== MAIN RUNNER FUNCTION =====

def run_universal_k_analysis(dataset_names=None):
    """Run Universal K analysis with GPU parallelization."""
    try:
        print("Starting Universal K Analysis...")

        # Set CUDA launch blocking for better error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # Create a unique run ID
        run_id = uuid.uuid4()

        # Check available GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Found {num_gpus} GPUs")

        # For each GPU, print its name and memory
        if num_gpus > 0:
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB memory")

        # Check available datasets
        available_datasets = get_available_datasets()
        if dataset_names is None:
            dataset_names = [name for name, info in available_datasets.items() if info['available']]

        # Define all methods
        all_methods = [
            ('PCA', create_pca_k_matrix),
            ('SVD', create_svd_k_matrix),
            ('NMF', create_nmf_k_matrix),
            ('ICA', create_ica_k_matrix),
            ('UMAP', create_umap_k_matrix),
            ('VAE', create_vae_k_matrix),
            ('Random', create_random_k_matrix),
            ('KernelPCA', create_kernel_pca_k_matrix),
            ('SparseDictionary', create_sparse_dictionary_k_matrix),
            ('Clustered', create_clustered_k_matrix),
            ('SparsePCA', create_sparse_pca_k_matrix),
            ('FactorAnalysis', create_factor_analysis_k_matrix)
        ]

        # Store results
        results = {}

        # Process each dataset
        for dataset_name in dataset_names:
            if not available_datasets.get(dataset_name, {}).get('available', False):
                print(f"Dataset {dataset_name} not available, skipping")
                continue

            print(f"\nProcessing {dataset_name}")

            try:
                # Load dataset
                x_train_path = available_datasets[dataset_name]['x_path']
                x_train = torch.tensor(np.load(x_train_path), dtype=torch.float32)

                # Standardize data
                x_mean = x_train.mean(dim=0, keepdim=True)
                x_std = x_train.std(dim=0, keepdim=True) + 1e-6
                x_train = (x_train - x_mean) / x_std

                # Sample for efficiency
                max_samples = 10000
                if x_train.shape[0] > max_samples:
                    indices = torch.randperm(x_train.shape[0])[:max_samples]
                    x_subset = x_train[indices]
                else:
                    x_subset = x_train

                print(f"Data shape: {x_subset.shape}")

                # Configurations to try - limiting to 3 and 5 factors as requested
                factors_to_try = [3, 5]
                dims_to_try = [8, 16]

                # Store results for this dataset
                method_results = {}

                # Setup Manager for shared data
                manager = Manager()
                result_queue = manager.Queue()
                log_queue = manager.Queue()

                # Start log handler process
                log_process = Process(
                    target=log_handler,
                    args=(log_queue, f"{dataset_name}_log_{run_id}.txt")
                )
                log_process.daemon = True
                log_process.start()

                # Group methods for parallel processing
                # Process at most num_gpus methods at a time
                method_groups = [all_methods[i:i + num_gpus] for i in range(0, len(all_methods), max(1, num_gpus))]

                for group_idx, method_group in enumerate(method_groups):
                    print(f"Processing method group {group_idx + 1}/{len(method_groups)}")

                    # Create and start processes
                    processes = []
                    active_methods = []

                    for idx, (method_name, method_func) in enumerate(method_group):
                        # Assign GPU ID (cycle through available GPUs)
                        gpu_id = idx % num_gpus if num_gpus > 0 else None

                        # Create process
                        p = Process(
                            target=process_method,
                            args=(gpu_id, method_name, method_func, x_subset, factors_to_try, dims_to_try, result_queue, log_queue)
                        )
                        p.daemon = True
                        processes.append(p)
                        active_methods.append(method_name)

                        # Start process
                        p.start()
                        print(f"Started {method_name} on GPU {gpu_id if gpu_id is not None else 'N/A'}")

                    # Wait for processes to complete with timeout
                    try:
                        # Set maximum wait time
                        max_wait_time = 30 * 60  # 30 minutes per group
                        start_time = time.time()

                        # Join processes with timeout
                        for p in processes:
                            remaining_time = max(0, max_wait_time - (time.time() - start_time))
                            p.join(remaining_time)

                            # Check for timeout
                            if time.time() - start_time > max_wait_time:
                                print("Timeout exceeded, terminating processes")
                                for p in processes:
                                    if p.is_alive():
                                        p.terminate()
                                break

                        # Collect results
                        collected_results = {}
                        for _ in range(len(active_methods)):
                            try:
                                # Get results with timeout
                                method_name, method_scores = result_queue.get(block=True, timeout=10)

                                # Process valid results
                                if method_scores:
                                    for result in method_scores:
                                        if method_name not in collected_results:
                                            collected_results[method_name] = []
                                        collected_results[method_name].append(result)
                                    print(f"Processed {method_name} results")
                            except Exception as e:
                                print(f"Error collecting result: {e}")

                        # Update method results
                        method_results.update(collected_results)

                    finally:
                        # Ensure all processes are terminated
                        for p in processes:
                            if p.is_alive():
                                p.terminate()

                        # Clean up
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Signal log handler to finish
                log_queue.put("DONE")
                log_process.join(timeout=5)
                if log_process.is_alive():
                    log_process.terminate()

                # Store results for this dataset
                if method_results:
                    results[dataset_name] = method_results

                    # Save results to CSV
                    df_rows = []
                    for method, configs in method_results.items():
                        for config in configs:
                            row = {
                                'dataset': dataset_name,
                                'method': config['method'],
                                'num_factors': config['num_factors'],
                                'latent_dim': config['latent_dim'],
                                'combined_score': config['combined_score']
                            }
                            # Add metrics
                            for metric_name, metric_value in config['metrics'].items():
                                row[metric_name] = metric_value

                            df_rows.append(row)

                    if df_rows:
                        df = pd.DataFrame(df_rows)
                        output_file = f'{dataset_name}_results_{run_id}.csv'
                        df.to_csv(output_file, index=False)
                        print(f"Results saved to {output_file}")

                        # Print summary of best methods for this dataset
                        print(f"\n=== {dataset_name} Results Summary ===")
                        for num_factors in factors_to_try:
                            print(f"\nFactor count: {num_factors}")

                            # Filter for this factor count
                            factor_df = df[df['num_factors'] == num_factors]

                            # Sort by combined score
                            sorted_df = factor_df.sort_values('combined_score', ascending=False)

                            # Print top methods
                            for i, row in sorted_df.head(3).iterrows():
                                print(f"  {row['method']} (dim={row['latent_dim']}): score={row['combined_score']:.4f}")
                                print(f"    Metrics: modularity={row['modularity']:.4f}, recon_error={row['recon_error']:.4f}, "
                                     f"total_correlation={row['total_correlation']:.4f}, factor_vae_score={row['factor_vae_score']:.4f}")

            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")

        # Save all results to CSV
        all_rows = []
        for dataset_name, methods in results.items():
            for method_name, configs in methods.items():
                for config in configs:
                    row = {
                        'dataset': dataset_name,
                        'method': config['method'],
                        'num_factors': config['num_factors'],
                        'latent_dim': config['latent_dim'],
                        'combined_score': config['combined_score'],
                    }
                    # Add metrics
                    for metric_name, metric_value in config['metrics'].items():
                        row[metric_name] = metric_value

                    all_rows.append(row)

        if all_rows:
            all_df = pd.DataFrame(all_rows)
            output_file = f'universal_k_results_{run_id}.csv'
            all_df.to_csv(output_file, index=False)
            print(f"All results saved to {output_file}")

            # Print overall summary
            print("\n=== Overall Results Summary ===")
            for num_factors in factors_to_try:
                print(f"\nFactor count: {num_factors}")

                # Group by method and get average scores
                factor_df = all_df[all_df['num_factors'] == num_factors]
                avg_scores = factor_df.groupby(['method'])['combined_score'].mean().reset_index()

                # Sort by combined score
                sorted_avg = avg_scores.sort_values('combined_score', ascending=False)

                # Print top methods
                for i, row in sorted_avg.head(3).iterrows():
                    print(f"  {row['method']}: average score={row['combined_score']:.4f}")

        return results

    except Exception as e:
        print(f"Error in run_universal_k_analysis: {e}")
        return {}

# ===== MAIN ENTRY POINT =====

def main():
    """Main function to run the universal K analysis."""
    try:
        # Set multiprocessing start method
        # 'spawn' is required for CUDA compatibility
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            print("Context already set, continuing...")

        # Record start time
        start_time = time.time()

        # Run analysis
        results = run_universal_k_analysis()

        # Print completion time
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        return results

    except Exception as e:
        print(f"Error in main: {e}")
        return None

if __name__ == "__main__":
    main()