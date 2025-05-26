#!/usr/bin/env python3
"""
Universal K Matrix - Configuration and Core Utilities
Comprehensive implementation for publication-quality results
"""
#!/usr/bin/env python3
"""
Universal K Matrix - All Required Imports
Complete import list with installation commands
"""


import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
# =============================================================================
# ALL IMPORTS - FIXED FOR AWS PYTORCH AMI
# =============================================================================
import sys
import time
import os
import json
import random
import warnings
import threading
import traceback
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from queue import Queue as ThreadQueue
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

# Multiprocessing
from multiprocessing import Queue as MPQueue, Process, Manager, Lock

# Scientific computing
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as torch_mp
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression

# Scipy
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.linalg import qr
from scipy.sparse.linalg import svds

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Concurrent processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Check for optional packages
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    print("Warning: scikit-posthocs not available. Some statistical tests will be skipped.")
    HAS_POSTHOCS = False

print("All imports successful!")



logger = logging.getLogger("UniversalKMatrix")
warnings.filterwarnings('ignore')

# =============================================================================
# COMPREHENSIVE CONFIGURATION
# =============================================================================

CONFIG = {
    # Hardware Configuration
    "use_all_gpus": True,
    "max_workers": None,  # None = auto-detect
    "device_type": "cuda",  # "cuda" or "cpu"

    # Experiment Configuration
    "experiment_types": ["universal_k", "sota_baseline", "enhanced_sota"],
    "cross_validation_folds": 5,
    "random_seeds": [42],  # Multiple seeds for robustness 123, 456, 789, 1011

    # Dataset Configuration
    "dataset_names": None,  # None = auto-detect all datasets
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "stratify_classification": True,

    # Universal K Matrix Configuration
    "k_methods": ["PCA", "FactorAnalysis", "Clustered", "Random"],
    "factors_to_try": [3, 5],
    "latent_dimensions": [8],
    "k_refinement_epochs": 100,
    "k_refinement_lr": 1e-4,

    # SOTA Baseline Configuration
    "baseline_methods": ["VIB", "BetaVAE", "SparseAutoencoder", "StandardAutoencoder"],
    "baseline_hyperparams": {
        "VIB": {"beta_values": [0.1, 1.0, 4.0]},
        "BetaVAE": {"beta_values": [0.5, 2.0, 4.0]},
        "SparseAutoencoder": {"sparsity_weights": [0.001, 0.01, 0.1]},
        "StandardAutoencoder": {"dropout_rates": [0.1, 0.3, 0.5]}
    },

    # Training Configuration - STANDARDIZED FOR FAIR COMPARISON
    "standard_architecture": {
        "hidden_dim": 256,
        "intermediate_dim": 128,
        "dropout_rate": 0.3,
        "activation": "LeakyReLU"
    },
    "training_config": {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 20,
        "min_delta": 1e-4,
        "gradient_clip": 1.0
    },

    # Knowledge Distillation Configuration
    "distillation_config": {
        "temperature": 4.0,
        "alpha": 0.7,  # Weight for distillation loss
        "beta": 0.3    # Weight for task loss
    },

    # Evaluation Configuration
    "evaluation_config": {
        "metrics_sample_size": 500,
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
        "statistical_tests": True
    },

    # Output Configuration
    "output_dir": "comprehensive_results",
    "save_models": True,
    "save_intermediate": True,
    "verbose": True,
    "log_level": "INFO"
}


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Set up comprehensive logging."""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("UniversalKMatrix")
    logger.setLevel(getattr(logging, log_level))

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_log_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# =============================================================================
# DEVICE AND GPU UTILITIES
# =============================================================================

class DeviceManager:
    """Manages GPU devices and memory efficiently."""

    def __init__(self, use_all_gpus: bool = True):
        self.use_all_gpus = use_all_gpus
        self.available_devices = self._detect_devices()
        self.device_usage = {device: 0 for device in self.available_devices}

    def _detect_devices(self) -> List[torch.device]:
        """Detect available devices."""
        devices = []

        if torch.cuda.is_available() and self.use_all_gpus:
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f'cuda:{i}'))
        elif torch.cuda.is_available():
            devices.append(torch.device('cuda:0'))
        else:
            devices.append(torch.device('cpu'))

        return devices

    def get_least_used_device(self) -> torch.device:
        """Get the device with least current usage."""
        return min(self.device_usage, key=self.device_usage.get)

    def allocate_device(self) -> torch.device:
        """Allocate a device and track usage."""
        device = self.get_least_used_device()
        self.device_usage[device] += 1
        return device

    def release_device(self, device: torch.device):
        """Release a device and clean memory."""
        if device in self.device_usage:
            self.device_usage[device] = max(0, self.device_usage[device] - 1)
        self.clean_device_memory(device)

    @staticmethod
    def clean_device_memory(device: torch.device):
        """Clean device memory."""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'available_devices': len(self.available_devices),
            'device_list': [str(d) for d in self.available_devices],
            'cuda_available': torch.cuda.is_available(),
            'device_usage': {str(k): v for k, v in self.device_usage.items()}
        }

        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_devices'] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]

        return info




# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_experiment_id() -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}_{random.randint(1000, 9999)}"

# =============================================================================
# NUMERICAL STABILITY UTILITIES
# =============================================================================

def safe_tensor_operation(tensor: torch.Tensor,
                         operation: str = "normalize",
                         eps: float = 1e-8,
                         nan_value: float = 0.0,
                         inf_value: float = 1.0) -> torch.Tensor:
    """Perform tensor operations with numerical stability."""
    # Handle NaN and Inf
    tensor = torch.nan_to_num(tensor, nan=nan_value, posinf=inf_value, neginf=-inf_value)

    if operation == "normalize":
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        return tensor / (norm + eps)
    elif operation == "standardize":
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        return (tensor - mean) / (std + eps)
    elif operation == "clamp":
        return torch.clamp(tensor, min=-10.0, max=10.0)
    else:
        return tensor

def robust_matrix_operations(matrix: torch.Tensor,
                           operation: str = "svd") -> Tuple[torch.Tensor, ...]:
    """Perform robust matrix operations."""
    # Add small noise for numerical stability
    matrix = matrix + torch.randn_like(matrix) * 1e-10

    if operation == "svd":
        try:
            U, S, V = torch.linalg.svd(matrix)
            # Ensure positive singular values
            S = torch.clamp(S, min=1e-10)
            return U, S, V
        except Exception:
            # Fallback to CPU if GPU fails
            matrix_cpu = matrix.cpu()
            U, S, V = torch.linalg.svd(matrix_cpu)
            S = torch.clamp(S, min=1e-10)
            return U.to(matrix.device), S.to(matrix.device), V.to(matrix.device)
    elif operation == "qr":
        try:
            Q, R = torch.qr(matrix)
            return Q, R
        except Exception:
            matrix_cpu = matrix.cpu()
            Q, R = torch.qr(matrix_cpu)
            return Q.to(matrix.device), R.to(matrix.device)
    else:
        return (matrix,)

# =============================================================================
# COMPREHENSIVE METRICS UTILITIES
# =============================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation for all experiment types."""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # AUC metrics (if probabilities available)
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['auc_roc_macro'] = roc_auc_score(y_true, y_proba,
                                                           multi_class='ovr', average='macro')
                    metrics['auc_roc_micro'] = roc_auc_score(y_true, y_proba,
                                                           multi_class='ovr', average='micro')
            except Exception:
                metrics['auc_roc'] = 0.5

        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {}

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # Additional metrics
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['explained_variance'] = 1 - (ss_res / (ss_tot + 1e-10))

        # Mean absolute percentage error
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        return metrics

    @staticmethod
    def calculate_disentanglement_metrics(z: torch.Tensor,
                                        x_data: torch.Tensor,
                                        num_factors: int,
                                        latent_dim: int) -> Dict[str, float]:
        """Calculate comprehensive disentanglement metrics."""
        metrics = {}

        try:
            # Move to CPU for sklearn operations
            z_np = z.detach().cpu().numpy()
            x_np = x_data.detach().cpu().numpy()

            # Reshape for factor analysis
            if z_np.ndim == 3:
                z_reshaped = z_np  # Already in (samples, factors, dims) format
            else:
                z_reshaped = z_np.reshape(-1, num_factors, latent_dim)

            # Sparsity score
            metrics['sparsity'] = MetricsCalculator._calculate_sparsity(z_np)

            # Modularity score
            metrics['modularity'] = MetricsCalculator._calculate_modularity(z_reshaped, num_factors)

            # Total correlation
            metrics['total_correlation'] = MetricsCalculator._calculate_total_correlation(z_reshaped)

            # Factor VAE score
            metrics['factor_vae_score'] = MetricsCalculator._calculate_factor_vae_score(z_reshaped, num_factors, latent_dim)

            # SAP score
            metrics['sap_score'] = MetricsCalculator._calculate_sap_score(z_reshaped, x_np, num_factors, latent_dim)

            # Mutual information gap
            metrics['mig_score'] = MetricsCalculator._calculate_mig_score(z_reshaped, x_np, num_factors, latent_dim)

        except Exception as e:
            logger = logging.getLogger("UniversalKMatrix")
            logger.warning(f"Error calculating disentanglement metrics: {e}")
            metrics = {
                'sparsity': 0.5, 'modularity': 0.5, 'total_correlation': 0.5,
                'factor_vae_score': 0.5, 'sap_score': 0.5, 'mig_score': 0.5
            }

        return metrics

    @staticmethod
    def _calculate_sparsity(z: np.ndarray) -> float:
        """Calculate sparsity using Gini coefficient."""
        z_flat = z.flatten()
        z_abs = np.abs(z_flat)
        z_sorted = np.sort(z_abs)
        n = len(z_sorted)
        cumsum = np.cumsum(z_sorted)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.5

    @staticmethod
    def _calculate_modularity(z: np.ndarray, num_factors: int) -> float:
        """Calculate modularity between factors."""
        if num_factors <= 1:
            return 1.0

        correlations = []
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                z_i = z[:, i, :].flatten()
                z_j = z[:, j, :].flatten()

                # Standardize
                z_i = (z_i - np.mean(z_i)) / (np.std(z_i) + 1e-8)
                z_j = (z_j - np.mean(z_j)) / (np.std(z_j) + 1e-8)

                corr = np.abs(np.corrcoef(z_i, z_j)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)

        return 1 - np.mean(correlations) if correlations else 0.5

    @staticmethod
    def _calculate_total_correlation(z: np.ndarray) -> float:
        """Calculate total correlation using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression

            factors = []
            for i in range(z.shape[1]):
                factor_data = z[:, i, :].flatten()
                factors.append(factor_data)

            if len(factors) < 2:
                return 0.0

            mi_scores = []
            for i in range(len(factors)):
                for j in range(i + 1, len(factors)):
                    mi = mutual_info_regression(
                        factors[i].reshape(-1, 1), factors[j]
                    )[0]
                    mi_scores.append(mi)

            return np.mean(mi_scores) if mi_scores else 0.5

        except Exception:
            return 0.5

    @staticmethod
    def _calculate_factor_vae_score(z: np.ndarray, num_factors: int, latent_dim: int) -> float:
        """Calculate Factor VAE disentanglement score."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            if num_factors <= 1:
                return 1.0

            scores = []
            for k in range(latent_dim):
                # Create targets for each latent dimension
                targets = []
                features = []

                for i in range(num_factors):
                    factor_data = z[:, i, k]
                    targets.extend([i] * len(factor_data))
                    features.extend(factor_data)

                if len(set(targets)) > 1:
                    X = np.array(features).reshape(-1, 1)
                    y = np.array(targets)

                    # Standardize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Train classifier
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(X_scaled, y)

                    score = clf.score(X_scaled, y)
                    scores.append(score)

            return np.mean(scores) if scores else 0.5

        except Exception:
            return 0.5

    @staticmethod
    def _calculate_sap_score(z: np.ndarray, x: np.ndarray, num_factors: int, latent_dim: int) -> float:
        """Calculate SAP (Separated Attribute Predictability) score."""
        try:
            from sklearn.feature_selection import mutual_info_regression

            # Use subset of features as proxies
            n_features = min(50, x.shape[1])
            feature_indices = np.random.choice(x.shape[1], n_features, replace=False)

            sap_scores = []
            for i in range(num_factors):
                for j in range(latent_dim):
                    latent_code = z[:, i, j]

                    mi_scores = []
                    for feat_idx in feature_indices:
                        feature = x[:, feat_idx]
                        if np.std(feature) > 1e-6:
                            mi = mutual_info_regression(
                                latent_code.reshape(-1, 1), feature
                            )[0]
                            mi_scores.append(mi)

                    if len(mi_scores) > 1:
                        mi_scores = sorted(mi_scores, reverse=True)
                        gap = (mi_scores[0] - mi_scores[1]) / (mi_scores[0] + 1e-8)
                        sap_scores.append(gap)

            return np.mean(sap_scores) if sap_scores else 0.5

        except Exception:
            return 0.5

    @staticmethod
    def _calculate_mig_score(z: np.ndarray, x: np.ndarray, num_factors: int, latent_dim: int) -> float:
        """Calculate Mutual Information Gap (MIG) score."""
        try:
            from sklearn.feature_selection import mutual_info_regression

            # Use subset of features
            n_features = min(20, x.shape[1])
            feature_indices = np.random.choice(x.shape[1], n_features, replace=False)

            mig_scores = []
            for feat_idx in feature_indices:
                feature = x[:, feat_idx]
                if np.std(feature) > 1e-6:
                    mi_values = []

                    for i in range(num_factors):
                        for j in range(latent_dim):
                            latent_code = z[:, i, j]
                            mi = mutual_info_regression(
                                latent_code.reshape(-1, 1), feature
                            )[0]
                            mi_values.append(mi)

                    if len(mi_values) > 1:
                        mi_values = sorted(mi_values, reverse=True)
                        gap = (mi_values[0] - mi_values[1]) / (mi_values[0] + 1e-8)
                        mig_scores.append(gap)

            return np.mean(mig_scores) if mig_scores else 0.5

        except Exception:
            return 0.5



#!/usr/bin/env python3
"""
Universal K Matrix - Data Loading and Preprocessing
Comprehensive data handling with proper splits and preprocessing
"""

# =============================================================================
# DATASET DISCOVERY AND LOADING
# =============================================================================

class DatasetManager:
    """Comprehensive dataset management with automatic discovery and preprocessing."""

    def __init__(self, data_dir: str = ".", config: Dict[str, Any] = None):
        self.data_dir = data_dir
        self.config = config or {}
        self.available_datasets = self._discover_datasets()
        self.dataset_info = {}

    def _discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available datasets in the directory."""
        datasets = {}

        # Common dataset prefixes
        dataset_prefixes = [
            'mnist', 'fashion_mnist', 'cifar10', 'cifar100',
            'diabetes', 'wine', 'breast_cancer', 'iris',
            'dsprites', 'celeba', 'svhn', 'omniglot',
            'reuters', 'imdb', 'amazon', 'yelp'
        ]

        for prefix in dataset_prefixes:
            dataset_info = self._check_dataset_files(prefix)
            if dataset_info['available']:
                datasets[prefix] = dataset_info
                logger.info(f"Found dataset: {prefix} with X shape {dataset_info.get('x_shape', 'unknown')}")

        # Also check for generic patterns
        for file in os.listdir(self.data_dir):
            if file.endswith('_x_train.npy'):
                prefix = file.replace('_x_train.npy', '')
                if prefix not in datasets:
                    dataset_info = self._check_dataset_files(prefix)
                    if dataset_info['available']:
                        datasets[prefix] = dataset_info
                        logger.info(f"Found dataset: {prefix} with X shape {dataset_info.get('x_shape', 'unknown')}")

        return datasets

    def _check_dataset_files(self, prefix: str) -> Dict[str, Any]:
        """Check if dataset files exist and get basic info."""
        info = {'available': False, 'files': {}}

        # Possible file patterns
        x_patterns = [f'{prefix}_x_train.npy', f'{prefix}_X.npy', f'{prefix}_data.npy']
        y_patterns = [f'{prefix}_y_train.npy', f'{prefix}_Y.npy', f'{prefix}_labels.npy', f'{prefix}_targets.npy']

        # Find X file
        x_file = None
        for pattern in x_patterns:
            if os.path.exists(os.path.join(self.data_dir, pattern)):
                x_file = pattern
                break

        if x_file:
            try:
                # Load to get shape info
                x_path = os.path.join(self.data_dir, x_file)
                x_sample = np.load(x_path, mmap_mode='r')

                info['available'] = True
                info['files']['x_file'] = x_file
                info['x_shape'] = x_sample.shape
                info['x_path'] = x_path

                # Find Y file
                y_file = None
                for pattern in y_patterns:
                    if os.path.exists(os.path.join(self.data_dir, pattern)):
                        y_file = pattern
                        break

                if y_file:
                    y_path = os.path.join(self.data_dir, y_file)
                    y_sample = np.load(y_path, mmap_mode='r')
                    info['files']['y_file'] = y_file
                    info['y_shape'] = y_sample.shape
                    info['y_path'] = y_path
                else:
                    # No labels found - will create dummy labels
                    info['y_shape'] = (x_sample.shape[0], 1)
                    info['y_path'] = None

            except Exception as e:
                logger.warning(f"Error loading dataset {prefix}: {e}")
                info['available'] = False
                info['error'] = str(e)

        return info

    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.available_datasets.keys())

    def load_dataset(self, dataset_name: str,
                    preprocess: bool = True,
                    return_raw: bool = False) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        """
        Load and preprocess a dataset.

        Returns:
            x_data: Feature tensor
            y_data: Target tensor
            is_classification: Whether this is a classification task
            metadata: Dataset metadata
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {list(self.available_datasets.keys())}")

        dataset_info = self.available_datasets[dataset_name]

        # Load X data
        x_data = np.load(dataset_info['x_path'])

        # Load Y data
        if dataset_info['y_path']:
            y_data = np.load(dataset_info['y_path'])
        else:
            # Create dummy regression targets
            y_data = np.random.randn(x_data.shape[0], 1)
            logger.warning(f"No labels found for {dataset_name}, created dummy regression targets")

        # Store raw data if requested
        if return_raw:
            raw_x, raw_y = x_data.copy(), y_data.copy()

        # Flatten X data if multi-dimensional
        if x_data.ndim > 2:
            original_shape = x_data.shape
            x_data = x_data.reshape(x_data.shape[0], -1)
            logger.info(f"Flattened X data from {original_shape} to {x_data.shape}")

        # Ensure Y data is proper shape
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)

        # Determine if classification task
        is_classification = self._determine_task_type(y_data)

        # Preprocess if requested
        if preprocess:
            x_data, y_data, preprocessing_info = self._preprocess_data(
                x_data, y_data, is_classification
            )
        else:
            preprocessing_info = {}

        # Convert to tensors
        x_tensor = torch.tensor(x_data, dtype=torch.float32)

        if is_classification:
            # Always ensure integer labels for classification
            if y_data.dtype not in [np.int32, np.int64]:
                y_data = y_data.astype(np.int64)
            y_tensor = torch.tensor(y_data.squeeze(), dtype=torch.long)
        else:
            # Always ensure float targets for regression
            y_data = y_data.astype(np.float32)
            y_tensor = torch.tensor(y_data, dtype=torch.float32)

        # Create metadata
        metadata = {
            'dataset_name': dataset_name,
            'original_shape': dataset_info['x_shape'],
            'processed_shape': x_tensor.shape,
            'target_shape': y_tensor.shape,
            'is_classification': is_classification,
            'n_classes': len(np.unique(y_data)) if is_classification else 1,
            'preprocessing_info': preprocessing_info
        }

        if return_raw:
            metadata['raw_x'] = raw_x
            metadata['raw_y'] = raw_y

        # Cache dataset info
        self.dataset_info[dataset_name] = metadata

        logger.info(f"Loaded {dataset_name}: X={x_tensor.shape}, Y={y_tensor.shape}, "
                   f"Classification={is_classification}, Classes={metadata['n_classes']}")

        return x_tensor, y_tensor, is_classification, metadata

    def _determine_task_type(self, y_data: np.ndarray) -> bool:
        """Determine if this is a classification or regression task."""
        # Check data type
        if y_data.dtype in [np.int32, np.int64]:
            return True

        # Check number of unique values
        unique_values = len(np.unique(y_data))
        total_samples = len(y_data)

        # If less than 10% unique values and less than 100 unique values, likely classification
        if unique_values < 100 and unique_values / total_samples < 0.1:
            return True

        # Check if values are integers
        if np.allclose(y_data, np.round(y_data)):
            return True

        return False

    def _preprocess_data(self, x_data: np.ndarray, y_data: np.ndarray,
                        is_classification: bool) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Comprehensive data preprocessing."""
        preprocessing_info = {}

        # Handle missing values
        if np.isnan(x_data).any():
            logger.warning("Found NaN values in data, filling with column means")
            x_data = np.nan_to_num(x_data, nan=np.nanmean(x_data, axis=0))

        if np.isnan(y_data).any():
            logger.warning("Found NaN values in targets")
            if is_classification:
                # Fill with mode
                from scipy import stats
                try:
                    mode_val = stats.mode(y_data[~np.isnan(y_data.ravel())], keepdims=True).mode[0]
                except:
                    # Fallback for newer scipy versions
                    from scipy.stats import mode
                    mode_val = mode(y_data[~np.isnan(y_data.ravel())], keepdims=True).mode[0]
                y_data = np.nan_to_num(y_data, nan=mode_val)
            else:
                # Fill with mean
                y_data = np.nan_to_num(y_data, nan=np.nanmean(y_data))

        # Feature scaling
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)
        preprocessing_info['feature_scaler'] = scaler

        # Handle infinite values
        x_data = np.nan_to_num(x_data, nan=0.0, posinf=3.0, neginf=-3.0)

        # Target preprocessing for regression
        if not is_classification:
            target_scaler = StandardScaler()
            y_data = target_scaler.fit_transform(y_data)
            preprocessing_info['target_scaler'] = target_scaler
        else:
            # Ensure integer labels for classification
            if y_data.dtype not in [np.int32, np.int64]:
                label_encoder = LabelEncoder()
                y_data = label_encoder.fit_transform(y_data.ravel()).reshape(-1, 1)
                preprocessing_info['label_encoder'] = label_encoder
            y_data = y_data.astype(np.int64)

        # Remove constant features
        feature_vars = np.var(x_data, axis=0)
        non_constant_features = feature_vars > 1e-8
        if not np.all(non_constant_features):
            logger.info(f"Removing {np.sum(~non_constant_features)} constant features")
            x_data = x_data[:, non_constant_features]
            preprocessing_info['feature_mask'] = non_constant_features

        return x_data, y_data, preprocessing_info

# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

class DataSplitter:
    """Handles proper data splitting with stratification and cross-validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        self.stratify = config.get('stratify_classification', True)
        self.cv_folds = config.get('cross_validation_folds', 5)

    def create_train_val_test_split(self, x_data: torch.Tensor, y_data: torch.Tensor,
                                   is_classification: bool, random_state: int = 42) -> Dict[str, torch.Tensor]:
        """Create stratified train/validation/test splits."""
        # Convert to numpy for sklearn
        x_np = x_data.numpy()
        y_np = y_data.numpy()

        if is_classification and self.stratify:
            # Stratified split
            stratify_y = y_np.ravel()

            # First split: train + val vs test
            x_trainval, x_test, y_trainval, y_test = train_test_split(
                x_np, y_np,
                test_size=self.test_split,
                stratify=stratify_y,
                random_state=random_state
            )

            # Second split: train vs val
            val_size_adjusted = self.val_split / (self.train_split + self.val_split)
            x_train, x_val, y_train, y_val = train_test_split(
                x_trainval, y_trainval,
                test_size=val_size_adjusted,
                stratify=y_trainval.ravel(),
                random_state=random_state
            )
        else:
            # Regular split
            x_trainval, x_test, y_trainval, y_test = train_test_split(
                x_np, y_np,
                test_size=self.test_split,
                random_state=random_state
            )

            val_size_adjusted = self.val_split / (self.train_split + self.val_split)
            x_train, x_val, y_train, y_val = train_test_split(
                x_trainval, y_trainval,
                test_size=val_size_adjusted,
                random_state=random_state
            )

        # Convert back to tensors
        splits = {
            'x_train': torch.tensor(x_train, dtype=torch.float32),
            'x_val': torch.tensor(x_val, dtype=torch.float32),
            'x_test': torch.tensor(x_test, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=y_data.dtype),
            'y_val': torch.tensor(y_val, dtype=y_data.dtype),
            'y_test': torch.tensor(y_test, dtype=y_data.dtype)
        }

        logger.info(f"Data splits - Train: {splits['x_train'].shape[0]}, "
                   f"Val: {splits['x_val'].shape[0]}, Test: {splits['x_test'].shape[0]}")

        return splits

    def create_cv_splits(self, x_data: torch.Tensor, y_data: torch.Tensor,
                        is_classification: bool, random_state: int = 42) -> List[Dict[str, torch.Tensor]]:
        """Create cross-validation splits."""
        x_np = x_data.numpy()
        y_np = y_data.numpy()

        if is_classification and self.stratify:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
            splits_generator = cv.split(x_np, y_np.ravel())
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=random_state)
            splits_generator = cv.split(x_np)

        cv_splits = []
        for fold, (train_idx, val_idx) in enumerate(splits_generator):
            split = {
                'fold': fold,
                'x_train': torch.tensor(x_np[train_idx], dtype=torch.float32),
                'x_val': torch.tensor(x_np[val_idx], dtype=torch.float32),
                'y_train': torch.tensor(y_np[train_idx], dtype=y_data.dtype),
                'y_val': torch.tensor(y_np[val_idx], dtype=y_data.dtype)
            }
            cv_splits.append(split)

        logger.info(f"Created {self.cv_folds}-fold cross-validation splits")
        return cv_splits

# =============================================================================
# DATA LOADER UTILITIES
# =============================================================================

class DataLoaderFactory:
    """Factory for creating optimized data loaders."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('training_config', {}).get('batch_size', 128)
        self.num_workers = min(4, os.cpu_count())

    def create_loaders(self, data_splits: Dict[str, torch.Tensor],
                      pin_memory: bool = True) -> Dict[str, DataLoader]:
        """Create optimized data loaders for train/val/test splits."""
        loaders = {}

        # Training loader with shuffling
        train_dataset = TensorDataset(data_splits['x_train'], data_splits['y_train'])
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        # Validation loader
        val_dataset = TensorDataset(data_splits['x_val'], data_splits['y_val'])
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )

        # Test loader
        test_dataset = TensorDataset(data_splits['x_test'], data_splits['y_test'])
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory
        )

        return loaders

    def create_cv_loaders(self, cv_splits: List[Dict[str, torch.Tensor]],
                         pin_memory: bool = True) -> List[Dict[str, DataLoader]]:
        """Create data loaders for cross-validation splits."""
        cv_loaders = []

        for split in cv_splits:
            train_dataset = TensorDataset(split['x_train'], split['y_train'])
            val_dataset = TensorDataset(split['x_val'], split['y_val'])

            loaders = {
                'fold': split['fold'],
                'train': DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=pin_memory,
                    drop_last=True
                ),
                'val': DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=pin_memory
                )
            }
            cv_loaders.append(loaders)

        return cv_loaders

# =============================================================================
# DATA ENCODING UTILITIES
# =============================================================================

def encode_data_with_k_matrix(x_data: torch.Tensor, k_matrix: torch.Tensor,
                             batch_size: int = 64) -> torch.Tensor:
    """Efficiently encode data using K-matrix with batching and CUDA handling."""

    # CUDA memory cleanup at start
    if x_data.device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    device = x_data.device
    original_device = k_matrix.device

    try:
        # Move k_matrix to same device as x_data
        k_matrix = k_matrix.to(device, non_blocking=True)

        # Ensure correct dtypes
        x_data = x_data.to(torch.float32)
        k_matrix = k_matrix.to(torch.float32)

        # Handle different k_matrix shapes and force consistent 3D shape
        if k_matrix.dim() == 2:
            # If 2D, we need to infer the correct 3D shape
            n_features = x_data.shape[1]
            total_elements = k_matrix.numel()

            # Try to find valid factorization
            if total_elements % n_features != 0:
                raise ValueError(f"K-matrix size {k_matrix.shape} incompatible with input features {n_features}")

            remaining = total_elements // n_features

            # Try common factor/latent_dim combinations
            possible_factors = [1, 2, 3, 4, 5, 8, 10]
            num_factors = None
            latent_dim = None

            for factors in possible_factors:
                if remaining % factors == 0:
                    latent_dim = remaining // factors
                    if latent_dim > 0:
                        num_factors = factors
                        break

            if num_factors is None:
                # Fallback: assume square-ish factorization
                num_factors = int(np.sqrt(remaining))
                latent_dim = remaining // num_factors
                if num_factors * latent_dim != remaining:
                    num_factors = 1
                    latent_dim = remaining

            k_matrix = k_matrix.view(num_factors, n_features, latent_dim)

        elif k_matrix.dim() == 1:
            # If 1D, reshape to single factor
            n_features = x_data.shape[1]
            if k_matrix.numel() % n_features != 0:
                raise ValueError(f"1D K-matrix size {k_matrix.numel()} not divisible by features {n_features}")

            latent_dim = k_matrix.numel() // n_features
            k_matrix = k_matrix.view(1, n_features, latent_dim)

        elif k_matrix.dim() == 3:
            # Already correct shape, just validate
            pass
        else:
            raise ValueError(f"K-matrix must be 1D, 2D, or 3D tensor, got {k_matrix.dim()}D")

        # Final shape validation
        if k_matrix.dim() != 3:
            raise ValueError(f"Failed to create 3D K-matrix, final shape: {k_matrix.shape}")

        num_factors, k_features, latent_dim = k_matrix.shape

        if k_features != x_data.shape[1]:
            raise ValueError(f"K-matrix features {k_features} don't match input features {x_data.shape[1]}")

        # Check for valid dimensions
        if num_factors <= 0 or latent_dim <= 0:
            raise ValueError(f"Invalid K-matrix dimensions: factors={num_factors}, latent_dim={latent_dim}")

        print(f"K-matrix shape: {k_matrix.shape}, encoding {x_data.shape[0]} samples")

        # Encode data in batches to manage memory
        all_z = []
        n_samples = x_data.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                # CUDA memory check and cleanup every few batches
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

                end_idx = min(i + batch_size, n_samples)
                batch_x = x_data[i:end_idx]

                # Validate batch
                if batch_x.numel() == 0:
                    continue

                batch_z_factors = []

                # Encode with each factor
                for j in range(num_factors):
                    try:
                        # Matrix multiplication: (batch_size, n_features) @ (n_features, latent_dim)
                        z_factor = torch.matmul(batch_x, k_matrix[j])

                        # Validate output
                        if torch.isnan(z_factor).any() or torch.isinf(z_factor).any():
                            print(f"Warning: Invalid values in factor {j}, replacing with zeros")
                            z_factor = torch.zeros_like(z_factor)

                        batch_z_factors.append(z_factor)

                    except RuntimeError as e:
                        if "out of memory" in str(e) and device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # Retry with smaller batch
                            smaller_batch_size = max(1, batch_size // 4)
                            print(f"CUDA OOM, retrying with batch_size={smaller_batch_size}")
                            return encode_data_with_k_matrix(x_data, k_matrix, smaller_batch_size)
                        else:
                            raise e

                if batch_z_factors:
                    # Stack factors: (batch_size, num_factors, latent_dim)
                    batch_z = torch.stack(batch_z_factors, dim=1)
                    all_z.append(batch_z)

        if not all_z:
            # Fallback: create zero tensor with correct shape
            print("Warning: No valid encodings produced, returning zeros")
            return torch.zeros(x_data.shape[0], num_factors, latent_dim,
                             device=device, dtype=torch.float32)

        # Concatenate all batches: (total_samples, num_factors, latent_dim)
        z = torch.cat(all_z, dim=0)

        # Final validation
        expected_shape = (x_data.shape[0], num_factors, latent_dim)
        if z.shape != expected_shape:
            print(f"Warning: Output shape {z.shape} doesn't match expected {expected_shape}")
            # Reshape if possible
            if z.numel() == np.prod(expected_shape):
                z = z.view(expected_shape)
            else:
                raise ValueError(f"Cannot reshape output to expected shape")

        print(f"Successfully encoded to shape: {z.shape}")

        return z

    except Exception as e:
        print(f"Error in encode_data_with_k_matrix: {e}")
        # Clean up and re-raise
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        raise e

    finally:
        # Final cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Move k_matrix back to original device if needed
        if original_device != device:
            k_matrix = k_matrix.to(original_device, non_blocking=True)

def reconstruct_from_k_encoding(z: torch.Tensor, k_matrix: torch.Tensor,
                               batch_size: int = 1024) -> torch.Tensor:
    """Reconstruct data from K-matrix encoding."""
    device = z.device
    if k_matrix.device != device:
        k_matrix = k_matrix.to(device, non_blocking=True)

    num_factors = k_matrix.shape[0]
    all_recon = []

    with torch.no_grad():
        for i in range(0, len(z), batch_size):
            batch_z = z[i:i + batch_size]
            batch_recon = torch.zeros(batch_z.shape[0], k_matrix.shape[1], device=device)

            for j in range(num_factors):
                z_j = batch_z[:, j]
                batch_recon += torch.matmul(z_j, k_matrix[j].T)

            all_recon.append(batch_recon)

    reconstructed = torch.cat(all_recon, dim=0)
    return reconstructed

#!/usr/bin/env python3
"""
Universal K Matrix - Model Architectures
Standardized neural network architectures for fair comparison
"""


# =============================================================================
# BASE ARCHITECTURE COMPONENTS
# =============================================================================

class StandardizedLinearBlock(nn.Module):
    """Standardized linear block with consistent architecture."""

    def __init__(self, input_dim: int, output_dim: int,
                 activation: str = "LeakyReLU", dropout_rate: float = 0.3,
                 batch_norm: bool = True):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        # Standardized activation
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Optional batch normalization
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x

# =============================================================================
# VARIATIONAL INFORMATION BOTTLENECK (VIB)
# =============================================================================

class VIB(nn.Module):
    """Variational Information Bottleneck with standardized architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 output_dim: int, beta: float = 1.0,
                 architecture_config: Dict[str, Any] = None):
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)
        intermediate_dim = config.get('intermediate_dim', hidden_dim // 2)

        # Encoder
        self.encoder = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            StandardizedLinearBlock(hidden_dim, intermediate_dim, activation, dropout_rate)
        )

        # Latent parameters
        self.mu_layer = nn.Linear(intermediate_dim, latent_dim)
        self.logvar_layer = nn.Linear(intermediate_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            StandardizedLinearBlock(latent_dim, intermediate_dim, activation, dropout_rate),
            StandardizedLinearBlock(intermediate_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        # Task predictor
        self.predictor = nn.Sequential(
            StandardizedLinearBlock(latent_dim, intermediate_dim, activation, dropout_rate),
            nn.Linear(intermediate_dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize final layer weights."""
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Make prediction from latent representation."""
        return self.predictor(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all outputs."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        pred = self.predict(z)

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'pred': pred
        }

    # VIB compute_loss method
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor,
                    outputs: Dict[str, torch.Tensor],
                    is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """Compute VIB loss components with proper dtype handling."""

        device = x.device

        # CUDA cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Extract outputs and ensure correct dtypes
            x_recon = outputs['x_recon'].to(torch.float32)
            mu = outputs['mu'].to(torch.float32)
            logvar = outputs['logvar'].to(torch.float32)
            pred = outputs['pred'].to(torch.float32)

            # Ensure input tensors have correct dtypes
            x = x.to(torch.float32)

            # Handle target tensor dtype based on task type
            if is_classification:
                # Classification: targets should be long integers
                if y.dtype != torch.long:
                    y = y.to(torch.long)

                # Ensure targets are properly shaped
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                elif y.dim() > 1 and y.shape[1] > 1:
                    # Multi-hot encoding, convert to class indices
                    y = torch.argmax(y, dim=1)
            else:
                # Regression: targets should be float
                y = y.to(torch.float32)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)

            # Reconstruction loss (always MSE)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            # Ensure KL loss is valid
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                print("Warning: Invalid KL loss, setting to 0")
                kl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Task loss
            if is_classification:
                if pred.size(1) > 1:
                    # Multi-class classification
                    task_loss = F.cross_entropy(pred, y, reduction='mean')
                else:
                    # Binary classification
                    task_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), y.float(), reduction='mean')
            else:
                # Regression
                if pred.dim() > 1:
                    pred = pred.squeeze()
                task_loss = F.mse_loss(pred, y, reduction='mean')

            # Ensure task loss is valid
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                print("Warning: Invalid task loss, setting to 1.0")
                task_loss = torch.tensor(1.0, device=device, dtype=torch.float32)

            # Total loss with beta weighting
            total_loss = task_loss + recon_loss + self.beta * kl_loss

            # Final validation
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: Invalid total loss, using task loss only")
                total_loss = task_loss

            return {
                'total_loss': total_loss,
                'task_loss': task_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }

        except Exception as e:
            print(f"Error in VIB compute_loss: {e}")
            # Return safe fallback losses
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            fallback_loss = torch.tensor(1.0, device=device, dtype=torch.float32)
            return {
                'total_loss': fallback_loss,
                'task_loss': fallback_loss,
                'recon_loss': fallback_loss,
                'kl_loss': torch.tensor(0.0, device=device, dtype=torch.float32)
            }


# =============================================================================
# BETA-VAE
# =============================================================================

class BetaVAE(nn.Module):
    """Beta-VAE with standardized architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 output_dim: int, beta: float = 4.0,
                 architecture_config: Dict[str, Any] = None):
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)
        intermediate_dim = config.get('intermediate_dim', hidden_dim // 2)

        # Encoder
        self.encoder = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            StandardizedLinearBlock(hidden_dim, intermediate_dim, activation, dropout_rate)
        )

        # Latent parameters
        self.mu_layer = nn.Linear(intermediate_dim, latent_dim)
        self.logvar_layer = nn.Linear(intermediate_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            StandardizedLinearBlock(latent_dim, intermediate_dim, activation, dropout_rate),
            StandardizedLinearBlock(intermediate_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        # Task predictor
        self.predictor = nn.Sequential(
            StandardizedLinearBlock(latent_dim, intermediate_dim, activation, dropout_rate),
            nn.Linear(intermediate_dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        pred = self.predict(z)

        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'pred': pred
        }

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor,
                    outputs: Dict[str, torch.Tensor],
                    is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """Compute Beta-VAE loss with emphasis on disentanglement."""

        device = x.device

        # CUDA cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Extract outputs and ensure correct dtypes
            x_recon = outputs['x_recon'].to(torch.float32)
            mu = outputs['mu'].to(torch.float32)
            logvar = outputs['logvar'].to(torch.float32)
            pred = outputs['pred'].to(torch.float32)

            # Ensure input tensors have correct dtypes
            x = x.to(torch.float32)

            # Handle target tensor dtype
            if is_classification:
                if y.dtype != torch.long:
                    y = y.to(torch.long)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                elif y.dim() > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
            else:
                y = y.to(torch.float32)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # KL divergence loss (emphasized for disentanglement)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

            # Ensure KL loss is valid
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                kl_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Task loss
            if is_classification:
                if pred.size(1) > 1:
                    task_loss = F.cross_entropy(pred, y, reduction='mean')
                else:
                    task_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), y.float(), reduction='mean')
            else:
                if pred.dim() > 1:
                    pred = pred.squeeze()
                task_loss = F.mse_loss(pred, y, reduction='mean')

            # Ensure task loss is valid
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                task_loss = torch.tensor(1.0, device=device, dtype=torch.float32)

            # Total loss with beta weighting (higher beta for more disentanglement)
            total_loss = task_loss + recon_loss + self.beta * kl_loss

            # Final validation
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = task_loss

            return {
                'total_loss': total_loss,
                'task_loss': task_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss
            }

        except Exception as e:
            print(f"Error in BetaVAE compute_loss: {e}")
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            fallback_loss = torch.tensor(1.0, device=device, dtype=torch.float32)
            return {
                'total_loss': fallback_loss,
                'task_loss': fallback_loss,
                'recon_loss': fallback_loss,
                'kl_loss': torch.tensor(0.0, device=device, dtype=torch.float32)
            }

# =============================================================================
# SPARSE AUTOENCODER
# =============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 regularization."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 output_dim: int, sparsity_weight: float = 0.01,
                 architecture_config: Dict[str, Any] = None):
        super().__init__()

        self.sparsity_weight = sparsity_weight
        self.latent_dim = latent_dim

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)

        # Encoder
        self.encoder = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            StandardizedLinearBlock(latent_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        # Task predictor
        self.predictor = nn.Sequential(
            StandardizedLinearBlock(latent_dim, hidden_dim // 2, activation, dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        pred = self.predict(z)

        return {
            'x_recon': x_recon,
            'z': z,
            'pred': pred
        }

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor,
                    outputs: Dict[str, torch.Tensor],
                    is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """Compute Sparse Autoencoder loss."""

        device = x.device

        # CUDA cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Extract outputs and ensure correct dtypes
            x_recon = outputs['x_recon'].to(torch.float32)
            z = outputs['z'].to(torch.float32)
            pred = outputs['pred'].to(torch.float32)

            # Ensure input tensors have correct dtypes
            x = x.to(torch.float32)

            # Handle target tensor dtype
            if is_classification:
                if y.dtype != torch.long:
                    y = y.to(torch.long)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                elif y.dim() > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
            else:
                y = y.to(torch.float32)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # Sparsity loss (L1 regularization on latent codes)
            sparsity_loss = torch.mean(torch.abs(z))

            # Ensure sparsity loss is valid
            if torch.isnan(sparsity_loss) or torch.isinf(sparsity_loss):
                sparsity_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Task loss
            if is_classification:
                if pred.size(1) > 1:
                    task_loss = F.cross_entropy(pred, y, reduction='mean')
                else:
                    task_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), y.float(), reduction='mean')
            else:
                if pred.dim() > 1:
                    pred = pred.squeeze()
                task_loss = F.mse_loss(pred, y, reduction='mean')

            # Ensure task loss is valid
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                task_loss = torch.tensor(1.0, device=device, dtype=torch.float32)

            # Total loss
            total_loss = task_loss + recon_loss + self.sparsity_weight * sparsity_loss

            # Final validation
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = task_loss

            return {
                'total_loss': total_loss,
                'task_loss': task_loss,
                'recon_loss': recon_loss,
                'sparsity_loss': sparsity_loss
            }

        except Exception as e:
            print(f"Error in SparseAutoencoder compute_loss: {e}")
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            fallback_loss = torch.tensor(1.0, device=device, dtype=torch.float32)
            return {
                'total_loss': fallback_loss,
                'task_loss': fallback_loss,
                'recon_loss': fallback_loss,
                'sparsity_loss': torch.tensor(0.0, device=device, dtype=torch.float32)
            }


# =============================================================================
# STANDARD AUTOENCODER (BASELINE)
# =============================================================================

class StandardAutoencoder(nn.Module):
    """Standard Autoencoder with dropout regularization."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 output_dim: int, architecture_config: Dict[str, Any] = None):
        super().__init__()

        self.latent_dim = latent_dim

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)

        # Encoder
        self.encoder = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            StandardizedLinearBlock(latent_dim, hidden_dim, activation, dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        # Task predictor
        self.predictor = nn.Sequential(
            StandardizedLinearBlock(latent_dim, hidden_dim // 2, activation, dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        pred = self.predict(z)

        return {
            'x_recon': x_recon,
            'z': z,
            'pred': pred
        }

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor,
                    outputs: Dict[str, torch.Tensor],
                    is_classification: bool = True) -> Dict[str, torch.Tensor]:
        """Compute Standard Autoencoder loss."""

        device = x.device

        # CUDA cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Extract outputs and ensure correct dtypes
            x_recon = outputs['x_recon'].to(torch.float32)
            pred = outputs['pred'].to(torch.float32)

            # Ensure input tensors have correct dtypes
            x = x.to(torch.float32)

            # Handle target tensor dtype
            if is_classification:
                if y.dtype != torch.long:
                    y = y.to(torch.long)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)
                elif y.dim() > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
            else:
                y = y.to(torch.float32)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # Task loss
            if is_classification:
                if pred.size(1) > 1:
                    task_loss = F.cross_entropy(pred, y, reduction='mean')
                else:
                    task_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), y.float(), reduction='mean')
            else:
                if pred.dim() > 1:
                    pred = pred.squeeze()
                task_loss = F.mse_loss(pred, y, reduction='mean')

            # Ensure task loss is valid
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                task_loss = torch.tensor(1.0, device=device, dtype=torch.float32)

            # Total loss (simple sum for standard autoencoder)
            total_loss = task_loss + recon_loss

            # Final validation
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = task_loss

            return {
                'total_loss': total_loss,
                'task_loss': task_loss,
                'recon_loss': recon_loss
            }

        except Exception as e:
            print(f"Error in StandardAutoencoder compute_loss: {e}")
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            fallback_loss = torch.tensor(1.0, device=device, dtype=torch.float32)
            return {
                'total_loss': fallback_loss,
                'task_loss': fallback_loss,
                'recon_loss': fallback_loss
            }

# =============================================================================
# TEACHER-STUDENT ARCHITECTURES
# =============================================================================

class TeacherModel(nn.Module):
    """Teacher model for knowledge distillation (works on K-encoded data)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 architecture_config: Dict[str, Any] = None):
        super().__init__()

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)
        intermediate_dim = config.get('intermediate_dim', hidden_dim // 2)

        self.network = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            StandardizedLinearBlock(hidden_dim, intermediate_dim, activation, dropout_rate),
            nn.Linear(intermediate_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class StudentModel(nn.Module):
    """Student model for knowledge distillation (works on raw data)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 architecture_config: Dict[str, Any] = None):
        super().__init__()

        config = architecture_config or {}
        activation = config.get('activation', 'LeakyReLU')
        dropout_rate = config.get('dropout_rate', 0.3)
        intermediate_dim = config.get('intermediate_dim', hidden_dim // 2)

        self.network = nn.Sequential(
            StandardizedLinearBlock(input_dim, hidden_dim, activation, dropout_rate),
            StandardizedLinearBlock(hidden_dim, intermediate_dim, activation, dropout_rate),
            nn.Linear(intermediate_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class KMatrixTeacherWrapper(nn.Module):
    """Wrapper that combines K-matrix encoding with teacher model."""

    def __init__(self, k_matrix: torch.Tensor, teacher_model: nn.Module):
        super().__init__()
        self.register_buffer('k_matrix', k_matrix)
        self.teacher_model = teacher_model
        self.num_factors = k_matrix.shape[0]

    def encode_with_k_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input using K-matrix."""
        batch_z_factors = []
        for j in range(self.num_factors):
            z_factor = torch.matmul(x, self.k_matrix[j])
            batch_z_factors.append(z_factor)
        z = torch.stack(batch_z_factors, dim=1)
        return z.reshape(z.shape[0], -1)  # Flatten for teacher

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_encoded = self.encode_with_k_matrix(x)
        return self.teacher_model(z_encoded)

# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating standardized models."""

    @staticmethod
    def create_model(model_type: str, input_dim: int, latent_dim: int,
                    output_dim: int, is_classification: bool,
                    hyperparams: Dict[str, Any] = None,
                    architecture_config: Dict[str, Any] = None) -> nn.Module:
        """Create model with standardized architecture."""

        # Default architecture config
        default_config = {
            'hidden_dim': 256,
            'intermediate_dim': 128,
            'activation': 'LeakyReLU',
            'dropout_rate': 0.3
        }

        if architecture_config:
            default_config.update(architecture_config)

        hidden_dim = default_config['hidden_dim']
        hyperparams = hyperparams or {}

        if model_type == "VIB":
            beta = hyperparams.get('beta', 1.0)
            return VIB(input_dim, hidden_dim, latent_dim, output_dim, beta, default_config)

        elif model_type == "BetaVAE":
            beta = hyperparams.get('beta', 4.0)
            return BetaVAE(input_dim, hidden_dim, latent_dim, output_dim, beta, default_config)

        elif model_type == "SparseAutoencoder":
            sparsity_weight = hyperparams.get('sparsity_weight', 0.01)
            return SparseAutoencoder(input_dim, hidden_dim, latent_dim, output_dim,
                                   sparsity_weight, default_config)

        elif model_type == "StandardAutoencoder":
            return StandardAutoencoder(input_dim, hidden_dim, latent_dim, output_dim, default_config)

        elif model_type == "Teacher":
            return TeacherModel(input_dim, hidden_dim, output_dim, default_config)

        elif model_type == "Student":
            return StudentModel(input_dim, hidden_dim, output_dim, default_config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

# =============================================================================
# KNOWLEDGE DISTILLATION UTILITIES
# =============================================================================






#!/usr/bin/env python3
"""
Universal K Matrix - K-Matrix Initialization and Refinement
Comprehensive K-matrix methods with numerical stability
"""


# =============================================================================
# K-MATRIX INITIALIZATION METHODS
# =============================================================================

class KMatrixInitializer:
    """Comprehensive K-matrix initialization methods."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def initialize_k_matrix(self, method: str, x_data: torch.Tensor,
                          num_factors: int, latent_dim: int,
                          device: torch.device) -> torch.Tensor:
        """Initialize K-matrix using specified method."""

        method_map = {
            'PCA': self._pca_initialization,
            'FactorAnalysis': self._factor_analysis_initialization,
            'ICA': self._ica_initialization,
            'Clustered': self._clustered_initialization,
            'Spectral': self._spectral_initialization,
            'Random': self._random_initialization,
            'Identity': self._identity_initialization,
            'Sparse': self._sparse_initialization
        }

        if method not in method_map:
            logger.warning(f"Unknown K-matrix method {method}, using Random")
            method = 'Random'

        try:
            k_matrix = method_map[method](x_data, num_factors, latent_dim, device)

            # Ensure proper shape and numerical stability
            k_matrix = self._ensure_proper_shape(k_matrix, num_factors, latent_dim, device)
            k_matrix = self._normalize_k_matrix(k_matrix)

            logger.info(f"Initialized K-matrix with {method}: shape {k_matrix.shape}")
            return k_matrix

        except Exception as e:
            logger.error(f"Error in {method} initialization: {e}, falling back to Random")
            return self._random_initialization(x_data, num_factors, latent_dim, device)

    def _pca_initialization(self, x_data: torch.Tensor, num_factors: int,
                          latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize using Principal Component Analysis."""
        x_np = x_data.cpu().numpy()

        # Subsample for efficiency
        if x_np.shape[0] > 10000:
            indices = np.random.choice(x_np.shape[0], 10000, replace=False)
            x_sample = x_np[indices]
        else:
            x_sample = x_np

        # Standardize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_sample)

        # Compute PCA
        total_components = min(num_factors * latent_dim, min(x_scaled.shape))
        pca = PCA(n_components=total_components, random_state=self.random_seed)
        pca.fit(x_scaled)

        # Create K matrices
        k_matrices = []
        components = pca.components_.T  # Shape: (n_features, n_components)

        for i in range(num_factors):
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                k = components[:, start_idx:end_idx]
            else:
                # Not enough components, use random
                k = np.random.randn(x_np.shape[1], latent_dim)

            # Pad if necessary
            if k.shape[1] < latent_dim:
                padding = np.random.randn(k.shape[0], latent_dim - k.shape[1]) * 0.1
                k = np.concatenate([k, padding], axis=1)

            k_matrices.append(torch.tensor(k, dtype=torch.float32, device=device))

        return torch.stack(k_matrices)

    def _factor_analysis_initialization(self, x_data: torch.Tensor, num_factors: int,
                                      latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize using Factor Analysis."""
        x_np = x_data.cpu().numpy()

        # Subsample for efficiency
        if x_np.shape[0] > 5000:
            indices = np.random.choice(x_np.shape[0], 5000, replace=False)
            x_sample = x_np[indices]
        else:
            x_sample = x_np

        # Standardize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_sample)

        # Factor Analysis
        total_components = min(num_factors * latent_dim, min(x_scaled.shape) - 1)
        fa = FactorAnalysis(n_components=total_components, random_state=self.random_seed)
        fa.fit(x_scaled)

        # Create K matrices
        k_matrices = []
        components = fa.components_.T  # Shape: (n_features, n_components)

        for i in range(num_factors):
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                k = components[:, start_idx:end_idx]
            else:
                k = np.random.randn(x_np.shape[1], latent_dim)

            # Pad if necessary
            if k.shape[1] < latent_dim:
                padding = np.random.randn(k.shape[0], latent_dim - k.shape[1]) * 0.1
                k = np.concatenate([k, padding], axis=1)

            k_matrices.append(torch.tensor(k, dtype=torch.float32, device=device))

        return torch.stack(k_matrices)

    def _ica_initialization(self, x_data: torch.Tensor, num_factors: int,
                          latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize using Independent Component Analysis."""
        x_np = x_data.cpu().numpy()

        # Subsample for efficiency
        if x_np.shape[0] > 3000:
            indices = np.random.choice(x_np.shape[0], 3000, replace=False)
            x_sample = x_np[indices]
        else:
            x_sample = x_np

        # Standardize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_sample)

        # ICA
        total_components = min(num_factors * latent_dim, min(x_scaled.shape))
        ica = FastICA(n_components=total_components, random_state=self.random_seed, max_iter=1000)

        try:
            ica.fit(x_scaled)
            components = ica.mixing_.T  # Use mixing matrix
        except Exception:
            # ICA failed, fall back to random
            logger.warning("ICA failed, using random initialization")
            return self._random_initialization(x_data, num_factors, latent_dim, device)

        # Create K matrices
        k_matrices = []
        for i in range(num_factors):
            start_idx = i * latent_dim
            end_idx = min(start_idx + latent_dim, total_components)

            if end_idx > start_idx:
                k = components[:, start_idx:end_idx]
            else:
                k = np.random.randn(x_np.shape[1], latent_dim)

            # Pad if necessary
            if k.shape[1] < latent_dim:
                padding = np.random.randn(k.shape[0], latent_dim - k.shape[1]) * 0.1
                k = np.concatenate([k, padding], axis=1)

            k_matrices.append(torch.tensor(k, dtype=torch.float32, device=device))

        return torch.stack(k_matrices)

    def _clustered_initialization(self, x_data: torch.Tensor, num_factors: int,
                                latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize using feature clustering."""
        x_np = x_data.cpu().numpy()

        # Subsample for efficiency
        if x_np.shape[0] > 5000:
            indices = np.random.choice(x_np.shape[0], 5000, replace=False)
            x_sample = x_np[indices]
        else:
            x_sample = x_np

        # Compute feature correlation matrix
        try:
            corr_matrix = np.corrcoef(x_sample.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)

            # K-means clustering on features
            n_clusters = min(num_factors, x_sample.shape[1])
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)

            # Use feature similarities as input to clustering
            feature_embeddings = 1 - distance_matrix
            cluster_labels = kmeans.fit_predict(feature_embeddings)

        except Exception:
            # Clustering failed, use random
            logger.warning("Feature clustering failed, using random assignment")
            cluster_labels = np.random.randint(0, num_factors, x_sample.shape[1])

        # Create K matrices based on clusters
        k_matrices = []
        for i in range(num_factors):
            k = torch.zeros(x_np.shape[1], latent_dim, device=device)

            # Find features in this cluster
            cluster_features = np.where(cluster_labels == (i % len(np.unique(cluster_labels))))[0]

            if len(cluster_features) > 0:
                # Assign features to latent dimensions
                for j in range(latent_dim):
                    # Select subset of features for this latent dimension
                    n_features_per_dim = max(1, len(cluster_features) // latent_dim)
                    start_feat = (j * n_features_per_dim) % len(cluster_features)
                    end_feat = min(start_feat + n_features_per_dim, len(cluster_features))

                    selected_features = cluster_features[start_feat:end_feat]
                    k[selected_features, j] = 1.0

                # Add small random noise
                k += torch.randn_like(k) * 0.01
            else:
                # No features in cluster, use random
                k = torch.randn(x_np.shape[1], latent_dim, device=device)

            k_matrices.append(k)

        return torch.stack(k_matrices)

    def _spectral_initialization(self, x_data: torch.Tensor, num_factors: int,
                               latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize using Spectral clustering."""
        x_np = x_data.cpu().numpy()

        # Subsample for efficiency
        if x_np.shape[0] > 3000:
            indices = np.random.choice(x_np.shape[0], 3000, replace=False)
            x_sample = x_np[indices]
        else:
            x_sample = x_np

        try:
            # Compute similarity matrix between features
            feature_corr = np.corrcoef(x_sample.T)
            feature_corr = np.nan_to_num(feature_corr, nan=0.0)
            similarity_matrix = np.abs(feature_corr)

            # Spectral clustering
            n_clusters = min(num_factors, x_sample.shape[1])
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=self.random_seed,
                                        affinity='precomputed')
            cluster_labels = spectral.fit_predict(similarity_matrix)

        except Exception:
            logger.warning("Spectral clustering failed, using random assignment")
            cluster_labels = np.random.randint(0, num_factors, x_sample.shape[1])

        # Create K matrices similar to clustered initialization
        k_matrices = []
        for i in range(num_factors):
            k = torch.zeros(x_np.shape[1], latent_dim, device=device)

            cluster_features = np.where(cluster_labels == (i % len(np.unique(cluster_labels))))[0]

            if len(cluster_features) > 0:
                for j in range(latent_dim):
                    n_features_per_dim = max(1, len(cluster_features) // latent_dim)
                    start_feat = (j * n_features_per_dim) % len(cluster_features)
                    end_feat = min(start_feat + n_features_per_dim, len(cluster_features))

                    selected_features = cluster_features[start_feat:end_feat]
                    k[selected_features, j] = 1.0

                k += torch.randn_like(k) * 0.01
            else:
                k = torch.randn(x_np.shape[1], latent_dim, device=device)

            k_matrices.append(k)

        return torch.stack(k_matrices)

    def _random_initialization(self, x_data: torch.Tensor, num_factors: int,
                             latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize with random orthogonal matrices."""
        n_features = x_data.shape[1]
        k_matrices = []

        for i in range(num_factors):
            # Random matrix
            k = torch.randn(n_features, latent_dim, device=device)

            # Orthogonalize against previous factors
            for prev_k in k_matrices:
                # Gram-Schmidt orthogonalization
                for j in range(latent_dim):
                    for prev_j in range(prev_k.shape[1]):
                        k[:, j] -= torch.dot(k[:, j], prev_k[:, prev_j]) * prev_k[:, prev_j]

            # QR decomposition for orthogonalization within factor
            try:
                q, r = torch.linalg.qr(k)
                k = q[:, :latent_dim]
            except Exception:
                # QR failed, just normalize
                k = F.normalize(k, dim=0)

            k_matrices.append(k)

        return torch.stack(k_matrices)

    def _identity_initialization(self, x_data: torch.Tensor, num_factors: int,
                               latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize with identity-like matrices."""
        n_features = x_data.shape[1]
        k_matrices = []

        features_per_factor = n_features // num_factors

        for i in range(num_factors):
            k = torch.zeros(n_features, latent_dim, device=device)

            # Each factor gets a subset of features
            start_feat = i * features_per_factor
            end_feat = min(start_feat + features_per_factor, n_features)

            for j in range(latent_dim):
                feat_idx = (start_feat + j) % n_features
                k[feat_idx, j] = 1.0

            # Add small random noise
            k += torch.randn_like(k) * 0.1

            k_matrices.append(k)

        return torch.stack(k_matrices)

    def _sparse_initialization(self, x_data: torch.Tensor, num_factors: int,
                             latent_dim: int, device: torch.device) -> torch.Tensor:
        """Initialize with sparse matrices."""
        n_features = x_data.shape[1]
        k_matrices = []

        sparsity_rate = 0.1  # 10% non-zero elements

        for i in range(num_factors):
            k = torch.zeros(n_features, latent_dim, device=device)

            # Randomly select features to be non-zero
            n_nonzero = int(n_features * latent_dim * sparsity_rate)
            flat_indices = torch.randperm(n_features * latent_dim)[:n_nonzero]

            row_indices = flat_indices // latent_dim
            col_indices = flat_indices % latent_dim

            k[row_indices, col_indices] = torch.randn(n_nonzero, device=device)

            k_matrices.append(k)

        return torch.stack(k_matrices)

    def _ensure_proper_shape(self, k_matrix: torch.Tensor, num_factors: int,
                           latent_dim: int, device: torch.device) -> torch.Tensor:
        """Ensure K-matrix has proper shape."""
        if k_matrix.shape[0] != num_factors:
            logger.warning(f"Adjusting number of factors from {k_matrix.shape[0]} to {num_factors}")
            if k_matrix.shape[0] < num_factors:
                # Add more factors
                additional = num_factors - k_matrix.shape[0]
                extra_factors = torch.randn(additional, k_matrix.shape[1], k_matrix.shape[2], device=device)
                k_matrix = torch.cat([k_matrix, extra_factors], dim=0)
            else:
                # Remove factors
                k_matrix = k_matrix[:num_factors]

        return k_matrix

    def _normalize_k_matrix(self, k_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize K-matrix for numerical stability."""
        # L2 normalize each column of each factor
        for i in range(k_matrix.shape[0]):
            k_matrix[i] = F.normalize(k_matrix[i], dim=0, p=2)

        # Handle any remaining NaN/Inf
        k_matrix = torch.nan_to_num(k_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        return k_matrix

# =============================================================================
# K-MATRIX REFINEMENT
# =============================================================================

class KMatrixRefiner:
    """Refines K-matrices using gradient-based optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epochs = config.get('k_refinement_epochs', 100)
        self.lr = config.get('k_refinement_lr', 1e-4)
        self.batch_size = config.get('training_config', {}).get('batch_size', 128)
        self.patience = 50
        self.min_delta = 1e-6

    def refine_k_matrix(self, x_data: torch.Tensor, k_matrix: torch.Tensor,
                       num_factors: int, latent_dim: int, device: torch.device) -> torch.Tensor:
        """Refine K-matrix using comprehensive optimization."""

        logger.info(f"Refining K-matrix: {k_matrix.shape} for {self.epochs} epochs")

        # Move to device and enable gradients
        k_matrix = k_matrix.clone().to(device).requires_grad_(True)
        x_data = x_data.to(device)

        # Setup optimizer with advanced scheduling
        optimizer = optim.AdamW([k_matrix], lr=self.lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=20, min_lr=1e-7
        )

        # Create data loader
        dataset = TensorDataset(x_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Training tracking
        best_loss = float('inf')
        best_k_matrix = k_matrix.clone().detach()
        patience_counter = 0
        loss_history = []

        for epoch in range(self.epochs):
            epoch_losses = []

            for batch_idx, (batch_x,) in enumerate(dataloader):
                batch_x = batch_x.to(device)

                # Zero gradients
                optimizer.zero_grad()

                try:
                    # Compute comprehensive loss
                    loss_dict = self._compute_refinement_loss(
                        batch_x, k_matrix, num_factors, latent_dim
                    )
                    total_loss = loss_dict['total_loss']

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logger.warning(f"Invalid loss at epoch {epoch}, batch {batch_idx}")
                        continue

                    # Backward pass
                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_([k_matrix], max_norm=1.0)

                    # Update
                    optimizer.step()

                    # Project to valid space (normalize)
                    with torch.no_grad():
                        k_matrix.data = self._project_k_matrix(k_matrix.data)

                    epoch_losses.append(total_loss.item())

                except Exception as e:
                    logger.warning(f"Error in refinement batch {batch_idx}: {e}")
                    continue

            if not epoch_losses:
                logger.warning(f"No valid batches in epoch {epoch}")
                continue

            # Epoch statistics
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                best_k_matrix = k_matrix.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}: Loss = {avg_loss:.6f}, "
                           f"LR = {optimizer.param_groups[0]['lr']:.2e}")

            if epoch % 5 == 0 and k_matrix.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final processing
        final_k_matrix = self._finalize_k_matrix(best_k_matrix, num_factors, latent_dim)

        logger.info(f"K-matrix refinement completed. Final loss: {best_loss:.6f}")
        return final_k_matrix.detach()

    def _compute_refinement_loss(self, x: torch.Tensor, k_matrix: torch.Tensor,
                              num_factors: int, latent_dim: int) -> Dict[str, torch.Tensor]:
        """Compute comprehensive refinement loss."""
        # Ensure K-matrix has correct shape
        expected_elements = num_factors * x.shape[1] * latent_dim
        if k_matrix.numel() != expected_elements:
            raise ValueError(f"K-matrix has {k_matrix.numel()} elements, expected {expected_elements}")

        k_reshaped = k_matrix.view(num_factors, x.shape[1], latent_dim)

        # Encode data
        z_factors = []
        for j in range(num_factors):
            z_factor = torch.matmul(x, k_reshaped[j])
            z_factors.append(z_factor)
        z = torch.stack(z_factors, dim=1)

        # Reconstruction
        recon = torch.zeros_like(x)
        for j in range(num_factors):
            recon += torch.matmul(z_factors[j], k_reshaped[j].T)

        # Loss components - use k_reshaped consistently
        losses = {}
        losses['recon_loss'] = F.mse_loss(recon, x)
        losses['sparsity_loss'] = 0.01 * torch.mean(torch.abs(k_reshaped))

        # Orthogonality between factors
        ortho_loss = 0.0
        if num_factors > 1:
            for i in range(num_factors):
                for j in range(i + 1, num_factors):
                    overlap = torch.norm(torch.mm(k_reshaped[i].T, k_reshaped[j]))
                    ortho_loss += overlap
            losses['ortho_loss'] = 0.005 * ortho_loss
        else:
            losses['ortho_loss'] = torch.tensor(0.0, device=x.device)

        # Variance preservation
        z_flat = z.view(z.shape[0], -1)
        z_var = torch.var(z_flat, dim=0).mean()
        losses['variance_loss'] = 0.1 * torch.clamp(1.0 - z_var, min=0.0)

        # Total correlation minimization
        if num_factors > 1:
            tc_loss = 0.0
            for i in range(num_factors):
                z_factor = z[:, i, :]
                z_centered = z_factor - z_factor.mean(dim=0, keepdim=True)
                cov = torch.mm(z_centered.T, z_centered) / (z_centered.shape[0] - 1)
                var = torch.diag(cov)
                corr = cov / (torch.sqrt(var.unsqueeze(0) * var.unsqueeze(1)) + 1e-8)
                eye = torch.eye(latent_dim, device=x.device)
                tc_loss += torch.sum(torch.abs(corr * (1 - eye)))
            losses['tc_loss'] = 0.005 * tc_loss
        else:
            losses['tc_loss'] = torch.tensor(0.0, device=x.device)

        # Smoothness regularization
        smoothness_loss = 0.0
        for i in range(num_factors):
            k_factor = k_reshaped[i]
            diff = k_factor[1:] - k_factor[:-1]
            smoothness_loss += torch.sum(diff ** 2)
        losses['smoothness_loss'] = 0.001 * smoothness_loss

        # Total loss
        total_loss = (
            3.0 * losses['recon_loss'] +
            losses['sparsity_loss'] +
            losses['ortho_loss'] +
            losses['variance_loss'] +
            losses['tc_loss'] +
            losses['smoothness_loss']
        )

        losses['total_loss'] = total_loss
        return losses

    def _project_k_matrix(self, k_matrix: torch.Tensor) -> torch.Tensor:
        """Project K-matrix to valid space."""
        # Normalize columns to unit norm
        k_norm = torch.norm(k_matrix.view(k_matrix.shape[0], -1, k_matrix.shape[-1]),
                           dim=1, keepdim=True)
        k_normalized = k_matrix / (k_norm + 1e-8)

        # Handle NaN/Inf
        k_normalized = torch.nan_to_num(k_normalized, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp to reasonable range
        k_normalized = torch.clamp(k_normalized, -10.0, 10.0)

        return k_normalized

    def _finalize_k_matrix(self, k_matrix: torch.Tensor, num_factors: int,
                          latent_dim: int) -> torch.Tensor:
        """Final processing of refined K-matrix."""

        # Ensure proper normalization
        k_matrix = self._project_k_matrix(k_matrix)

        # Optional: Re-orthogonalize factors
        k_reshaped = k_matrix.view(num_factors, -1, latent_dim)

        if num_factors > 1:
            # Gram-Schmidt orthogonalization between factors
            for i in range(1, num_factors):
                for j in range(i):
                    # Remove projection onto previous factors
                    projection = torch.sum(k_reshaped[i] * k_reshaped[j], dim=0, keepdim=True)
                    k_reshaped[i] = k_reshaped[i] - projection * k_reshaped[j]

                # Renormalize
                k_reshaped[i] = F.normalize(k_reshaped[i], dim=0, p=2)

        k_matrix = k_reshaped.view_as(k_matrix)

        return k_matrix

# =============================================================================
# K-MATRIX EVALUATION METRICS
# =============================================================================

class KMatrixEvaluator:
    """Evaluates K-matrix quality using various metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_size = config.get('evaluation_config', {}).get('metrics_sample_size', 2000)

    def evaluate_k_matrix(self, x_data: torch.Tensor, k_matrix: torch.Tensor,
                         num_factors: int, latent_dim: int, device: torch.device) -> Dict[str, float]:
        """Comprehensive K-matrix evaluation."""

        logger.info("Evaluating K-matrix quality...")

        # Move data to device
        x_data = x_data.to(device)
        k_matrix = k_matrix.to(device)

        # Sample data if too large
        if x_data.shape[0] > self.sample_size:
            indices = torch.randperm(x_data.shape[0])[:self.sample_size]
            x_sample = x_data[indices]
        else:
            x_sample = x_data

        # Encode data
        z_encoded = self._encode_with_k_matrix(x_sample, k_matrix, num_factors, latent_dim)

        # Compute metrics
        metrics = {}

        # 1. Reconstruction quality
        metrics.update(self._compute_reconstruction_metrics(x_sample, k_matrix, z_encoded, num_factors, latent_dim))

        # 2. Disentanglement metrics
        metrics.update(self._compute_disentanglement_metrics(z_encoded, x_sample, num_factors, latent_dim))

        # 3. K-matrix properties
        metrics.update(self._compute_matrix_properties(k_matrix, num_factors, latent_dim))

        # 4. Latent space quality
        metrics.update(self._compute_latent_space_metrics(z_encoded, num_factors, latent_dim))

        logger.info(f"K-matrix evaluation completed. Reconstruction error: {metrics.get('recon_error', 'N/A'):.4f}")

        return metrics

    def _encode_with_k_matrix(self, x: torch.Tensor, k_matrix: torch.Tensor,
                             num_factors: int, latent_dim: int) -> torch.Tensor:
        """Encode data using K-matrix."""
        k_reshaped = k_matrix.view(num_factors, -1, latent_dim)

        z_factors = []
        for j in range(num_factors):
            z_factor = torch.matmul(x, k_reshaped[j])
            z_factors.append(z_factor)

        return torch.stack(z_factors, dim=1)  # Shape: (batch, factors, latent_dim)

    def _compute_reconstruction_metrics(self, x: torch.Tensor, k_matrix: torch.Tensor,
                                     z: torch.Tensor, num_factors: int, latent_dim: int) -> Dict[str, float]:
        """Compute reconstruction quality metrics."""
        k_reshaped = k_matrix.view(num_factors, -1, latent_dim)

        # Reconstruct
        recon = torch.zeros_like(x)
        for j in range(num_factors):
            recon += torch.matmul(z[:, j], k_reshaped[j].T)

        # Metrics
        metrics = {}
        metrics['recon_error'] = F.mse_loss(recon, x).item()
        metrics['recon_mae'] = F.l1_loss(recon, x).item()

        # R-squared for reconstruction
        ss_res = torch.sum((x - recon) ** 2).item()
        ss_tot = torch.sum((x - x.mean()) ** 2).item()
        metrics['recon_r2'] = 1 - (ss_res / (ss_tot + 1e-10))

        # Relative error
        x_norm = torch.norm(x)
        recon_norm = torch.norm(recon - x)
        metrics['relative_recon_error'] = (recon_norm / (x_norm + 1e-10)).item()

        return metrics

    def _compute_disentanglement_metrics(self, z: torch.Tensor, x: torch.Tensor,
                                       num_factors: int, latent_dim: int) -> Dict[str, float]:
        """Compute disentanglement quality metrics."""


        metrics = {}

        try:
            disentangle_metrics = MetricsCalculator.calculate_disentanglement_metrics(
                z, x, num_factors, latent_dim
            )
            metrics.update(disentangle_metrics)
        except Exception as e:
            logger.warning(f"Error computing disentanglement metrics: {e}")
            metrics.update({
                'sparsity': 0.5, 'modularity': 0.5, 'total_correlation': 0.5,
                'factor_vae_score': 0.5, 'sap_score': 0.5, 'mig_score': 0.5
            })

        return metrics

    def _compute_matrix_properties(self, k_matrix: torch.Tensor,
                                 num_factors: int, latent_dim: int) -> Dict[str, float]:
        """Compute K-matrix mathematical properties."""
        metrics = {}

        k_reshaped = k_matrix.view(num_factors, -1, latent_dim)

        # Condition numbers
        condition_numbers = []
        for i in range(num_factors):
            try:
                U, S, V = torch.linalg.svd(k_reshaped[i])
                cond_num = (S.max() / (S.min() + 1e-10)).item()
                condition_numbers.append(cond_num)
            except Exception:
                condition_numbers.append(1.0)

        metrics['avg_condition_number'] = np.mean(condition_numbers)
        metrics['max_condition_number'] = np.max(condition_numbers)

        # Orthogonality between factors
        if num_factors > 1:
            ortho_scores = []
            for i in range(num_factors):
                for j in range(i + 1, num_factors):
                    overlap = torch.norm(torch.mm(k_reshaped[i].T, k_reshaped[j])).item()
                    ortho_scores.append(overlap)
            metrics['avg_factor_orthogonality'] = np.mean(ortho_scores)
        else:
            metrics['avg_factor_orthogonality'] = 0.0

        # Sparsity of K-matrix itself
        k_flat = k_matrix.view(-1)
        metrics['k_matrix_sparsity'] = (torch.sum(torch.abs(k_flat) < 1e-6).float() / len(k_flat)).item()

        # Frobenius norm
        metrics['k_matrix_frobenius_norm'] = torch.norm(k_matrix, p='fro').item()

        return metrics

    def _compute_latent_space_metrics(self, z: torch.Tensor,
                                    num_factors: int, latent_dim: int) -> Dict[str, float]:
        """Compute latent space quality metrics."""
        metrics = {}

        z_flat = z.view(z.shape[0], -1)  # Flatten to (batch, total_latent_dims)

        # Variance explained
        total_var = torch.var(z_flat, dim=0).sum().item()
        metrics['total_latent_variance'] = total_var

        # Effective dimensionality (participation ratio)
        eigenvals = torch.linalg.eigvals(torch.cov(z_flat.T)).real
        eigenvals = torch.clamp(eigenvals, min=0)
        participation_ratio = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()
        metrics['effective_dimensionality'] = participation_ratio.item()

        # Factor-wise statistics
        factor_variances = []
        for i in range(num_factors):
            factor_z = z[:, i, :]  # Shape: (batch, latent_dim)
            factor_var = torch.var(factor_z, dim=0).mean().item()
            factor_variances.append(factor_var)

        metrics['avg_factor_variance'] = np.mean(factor_variances)
        metrics['std_factor_variance'] = np.std(factor_variances)

        # Latent space smoothness (local variance)
        if z_flat.shape[0] > 100:
            sample_indices = torch.randperm(z_flat.shape[0])[:100]
            z_sample = z_flat[sample_indices]

            # Compute pairwise distances
            dists = torch.cdist(z_sample, z_sample)

            # Find k nearest neighbors (k=5)
            k = min(5, z_sample.shape[0] - 1)
            _, knn_indices = torch.topk(dists, k + 1, largest=False, dim=1)

            # Compute local variance
            local_variances = []
            for i in range(z_sample.shape[0]):
                neighbors = z_sample[knn_indices[i, 1:]]  # Exclude self
                local_var = torch.var(neighbors, dim=0).mean().item()
                local_variances.append(local_var)

            metrics['latent_space_smoothness'] = np.mean(local_variances)
        else:
            metrics['latent_space_smoothness'] = 0.0

        return metrics


#!/usr/bin/env python3
"""
Universal K Matrix - Training Engine and Optimization
Comprehensive training system with early stopping, checkpointing, and monitoring
"""


# =============================================================================
# KNOWLEDGE DISTILLATION UTILITIES (move this BEFORE TrainingEngine class)
# =============================================================================

def compute_distillation_loss(student_outputs: torch.Tensor,
                            teacher_outputs: torch.Tensor,
                            targets: torch.Tensor,
                            temperature: float = 4.0,
                            alpha: float = 0.7,
                            is_classification: bool = True) -> Dict[str, torch.Tensor]:
    """Compute knowledge distillation loss with proper dtype handling."""

    device = student_outputs.device

    # CUDA cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    try:
        # Ensure all tensors are on the same device and have correct dtypes
        student_outputs = student_outputs.to(torch.float32)
        teacher_outputs = teacher_outputs.to(device).to(torch.float32)

        # Handle targets based on task type
        if is_classification:
            if targets.dtype != torch.long:
                targets = targets.to(torch.long)
            if targets.dim() > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            elif targets.dim() > 1 and targets.shape[1] > 1:
                targets = torch.argmax(targets, dim=1)
        else:
            targets = targets.to(torch.float32)
            if targets.dim() > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

        targets = targets.to(device)

        # Task loss (hard targets)
        if is_classification:
            if student_outputs.size(1) > 1:
                # Multi-class classification
                task_loss = F.cross_entropy(student_outputs, targets, reduction='mean')
            else:
                # Binary classification
                task_loss = F.binary_cross_entropy_with_logits(
                    student_outputs.squeeze(), targets.float(), reduction='mean'
                )
        else:
            # Regression
            if student_outputs.dim() > 1:
                student_outputs_squeezed = student_outputs.squeeze()
            else:
                student_outputs_squeezed = student_outputs
            task_loss = F.mse_loss(student_outputs_squeezed, targets, reduction='mean')

        # Distillation loss (soft targets)
        if is_classification and student_outputs.size(1) > 1:
            # Multi-class classification
            soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
            soft_student = F.log_softmax(student_outputs / temperature, dim=1)
            distillation_loss = -(soft_teacher * soft_student).sum(dim=1).mean() * (temperature ** 2)
        elif is_classification:
            # Binary classification
            teacher_probs = torch.sigmoid(teacher_outputs / temperature)
            student_log_probs = F.logsigmoid(student_outputs / temperature)
            student_log_probs_neg = torch.log(1 - torch.sigmoid(student_outputs / temperature) + 1e-8)
            distillation_loss = -(teacher_probs * student_log_probs +
                                (1 - teacher_probs) * student_log_probs_neg).mean() * (temperature ** 2)
        else:
            # Regression
            teacher_squeezed = teacher_outputs.squeeze() if teacher_outputs.dim() > 1 else teacher_outputs
            student_squeezed = student_outputs.squeeze() if student_outputs.dim() > 1 else student_outputs
            distillation_loss = F.mse_loss(student_squeezed, teacher_squeezed, reduction='mean')

        # Ensure losses are valid
        if torch.isnan(task_loss) or torch.isinf(task_loss):
            task_loss = torch.tensor(1.0, device=device, dtype=torch.float32)

        if torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
            distillation_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * task_loss

        # Final validation
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = task_loss

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss
        }

    except Exception as e:
        print(f"Error in compute_distillation_loss: {e}")
        device = student_outputs.device if hasattr(student_outputs, 'device') else torch.device('cpu')
        fallback_loss = torch.tensor(1.0, device=device, dtype=torch.float32)
        return {
            'total_loss': fallback_loss,
            'task_loss': fallback_loss,
            'distillation_loss': torch.tensor(0.0, device=device, dtype=torch.float32)
        }

# =============================================================================
# TRAINING ENGINE (this comes after compute_distillation_loss)
# =============================================================================



class TrainingEngine:
    """Comprehensive training engine with monitoring and optimization."""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

        # Training configuration
        training_config = config.get('training_config', {})
        self.epochs = training_config.get('epochs', 200)
        self.learning_rate = training_config.get('learning_rate', 1e-3)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        self.patience = training_config.get('patience', 20)
        self.min_delta = training_config.get('min_delta', 1e-4)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)

        # Mixed precision training
        self.use_amp = torch.cuda.is_available() and config.get('use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Monitoring
        self.save_models = config.get('save_models', True)
        self.save_intermediate = config.get('save_intermediate', True)
        self.verbose = config.get('verbose', True)

        # Training history
        self.history = defaultdict(list)

    def train_autoencoder_model(self, model: nn.Module, train_loader: DataLoader,
                              val_loader: DataLoader, is_classification: bool,
                              hyperparams: Dict[str, Any] = None,
                              experiment_id: str = None) -> Dict[str, Any]:
        """Train autoencoder-based model with comprehensive monitoring."""

        logger.info(f"Training {model.__class__.__name__} for {self.epochs} epochs")

        # Setup optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-7, verbose=self.verbose
        )

        # Training tracking
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        training_history = {
            'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': [],
            'learning_rates': [], 'epoch_times': []
        }

        # Training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch_autoencoder(
                model, train_loader, optimizer, is_classification
            )

            # Validation phase
            val_metrics = self._validate_epoch_autoencoder(
                model, val_loader, is_classification
            )

            # Scheduler step
            scheduler.step(val_metrics['total_loss'])

            # Record history
            training_history['train_loss'].append(train_metrics['total_loss'])
            training_history['val_loss'].append(val_metrics['total_loss'])
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            training_history['epoch_times'].append(time.time() - epoch_start_time)

            # Early stopping check
            if val_metrics['total_loss'] < best_val_loss - self.min_delta:
                best_val_loss = val_metrics['total_loss']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Epoch {epoch}/{self.epochs}: "
                    f"Train Loss = {train_metrics['total_loss']:.4f}, "
                    f"Val Loss = {val_metrics['total_loss']:.4f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.2e}, "
                    f"Time = {training_history['epoch_times'][-1]:.2f}s"
                )

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save model if requested
        if self.save_models and experiment_id:
            self._save_model(model, experiment_id, training_history)

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'total_epochs': epoch + 1
        }

    def train_teacher_student(self, teacher_model: nn.Module, student_model: nn.Module,
                            teacher_loader: DataLoader, student_loader: DataLoader,
                            val_loader: DataLoader, is_classification: bool,
                            distillation_config: Dict[str, Any] = None,
                            experiment_id: str = None) -> Dict[str, Any]:
        """Train teacher-student architecture with knowledge distillation."""

        logger.info("Training teacher-student architecture")

        distillation_config = distillation_config or self.config.get('distillation_config', {})
        temperature = distillation_config.get('temperature', 4.0)
        alpha = distillation_config.get('alpha', 0.7)

        # Phase 1: Train teacher model
        logger.info("Phase 1: Training teacher model")
        teacher_results = self.train_autoencoder_model(
            teacher_model, teacher_loader, val_loader, is_classification,
            experiment_id=f"{experiment_id}_teacher" if experiment_id else None
        )

        # Phase 2: Train student with knowledge distillation
        logger.info("Phase 2: Training student model with knowledge distillation")

        # Setup student optimizer
        student_optimizer = optim.AdamW(
            student_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            student_optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-7
        )

        # Student training tracking
        best_val_loss = float('inf')
        best_student_state = None
        patience_counter = 0
        student_history = {
            'train_loss': [], 'val_loss': [], 'distillation_loss': [],
            'task_loss': [], 'learning_rates': [], 'epoch_times': []
        }

        # Teacher in evaluation mode
        teacher_model.eval()

        # Student training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self._train_epoch_distillation(
                student_model, teacher_model, student_loader, student_optimizer,
                is_classification, temperature, alpha
            )

            # Validation phase
            val_metrics = self._validate_epoch_simple(
                student_model, val_loader, is_classification
            )

            # Scheduler step
            student_scheduler.step(val_metrics['loss'])

            # Record history
            student_history['train_loss'].append(train_metrics['total_loss'])
            student_history['val_loss'].append(val_metrics['loss'])
            student_history['distillation_loss'].append(train_metrics['distillation_loss'])
            student_history['task_loss'].append(train_metrics['task_loss'])
            student_history['learning_rates'].append(student_optimizer.param_groups[0]['lr'])
            student_history['epoch_times'].append(time.time() - epoch_start_time)

            # Early stopping check
            if val_metrics['loss'] < best_val_loss - self.min_delta:
                best_val_loss = val_metrics['loss']
                best_student_state = student_model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Student Epoch {epoch}/{self.epochs}: "
                    f"Train Loss = {train_metrics['total_loss']:.4f}, "
                    f"Val Loss = {val_metrics['loss']:.4f}, "
                    f"Distill Loss = {train_metrics['distillation_loss']:.4f}, "
                    f"Task Loss = {train_metrics['task_loss']:.4f}"
                )

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Student early stopping at epoch {epoch}")
                break

        # Restore best student model
        if best_student_state is not None:
            student_model.load_state_dict(best_student_state)

        # Save models if requested
        if self.save_models and experiment_id:
            self._save_model(student_model, f"{experiment_id}_student", student_history)

        return {
            'teacher_model': teacher_model,
            'student_model': student_model,
            'teacher_results': teacher_results,
            'student_best_val_loss': best_val_loss,
            'student_history': student_history,
            'student_epochs': epoch + 1
        }

    def _train_epoch_autoencoder(self, model: nn.Module, train_loader: DataLoader,
                               optimizer: optim.Optimizer, is_classification: bool) -> Dict[str, float]:
        """Train one epoch for autoencoder model."""
        model.train()

        total_loss = 0.0
        component_losses = defaultdict(float)
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = model(batch_x)
                    losses = model.compute_loss(batch_x, batch_y, outputs, is_classification)

                self.scaler.scale(losses['total_loss']).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(batch_x)
                losses = model.compute_loss(batch_x, batch_y, outputs, is_classification)

                losses['total_loss'].backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                optimizer.step()

            # Accumulate losses
            total_loss += losses['total_loss'].item()
            for key, value in losses.items():
                if key != 'total_loss':
                    component_losses[key] += value.item()

            num_batches += 1

            if num_batches % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Average losses
        metrics = {'total_loss': total_loss / num_batches}
        for key, value in component_losses.items():
            metrics[key] = value / num_batches

        return metrics

    def _validate_epoch_autoencoder(self, model: nn.Module, val_loader: DataLoader,
                                  is_classification: bool) -> Dict[str, float]:
        """Validate one epoch for autoencoder model."""
        model.eval()

        total_loss = 0.0
        component_losses = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(batch_x)
                        losses = model.compute_loss(batch_x, batch_y, outputs, is_classification)
                else:
                    outputs = model(batch_x)
                    losses = model.compute_loss(batch_x, batch_y, outputs, is_classification)

                # Accumulate losses
                total_loss += losses['total_loss'].item()
                for key, value in losses.items():
                    if key != 'total_loss':
                        component_losses[key] += value.item()

                num_batches += 1

        # Average losses
        metrics = {'total_loss': total_loss / num_batches}
        for key, value in component_losses.items():
            metrics[key] = value / num_batches

        return metrics

    def _train_epoch_distillation(self, student_model: nn.Module, teacher_model: nn.Module,
                                train_loader: DataLoader, optimizer: optim.Optimizer,
                                is_classification: bool, temperature: float, alpha: float) -> Dict[str, float]:
        #from .models import compute_distillation_loss
        """Train one epoch with knowledge distillation."""
        student_model.train()
        teacher_model.eval()

        total_loss = 0.0
        distillation_loss = 0.0
        task_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    # Teacher forward pass
                    with torch.no_grad():
                        teacher_outputs = teacher_model(batch_x)

                    # Student forward pass
                    student_outputs = student_model(batch_x)

                    # Compute distillation loss

                    losses = compute_distillation_loss(
                        student_outputs, teacher_outputs, batch_y,
                        temperature, alpha, is_classification
                    )

                self.scaler.scale(losses['total_loss']).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), self.gradient_clip)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Teacher forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(batch_x)

                # Student forward pass
                student_outputs = student_model(batch_x)

                # Compute distillation loss

                losses = compute_distillation_loss(
                    student_outputs, teacher_outputs, batch_y,
                    temperature, alpha, is_classification
                )

                losses['total_loss'].backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), self.gradient_clip)

                optimizer.step()

            # Accumulate losses
            total_loss += losses['total_loss'].item()
            distillation_loss += losses['distillation_loss'].item()
            task_loss += losses['task_loss'].item()
            num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'distillation_loss': distillation_loss / num_batches,
            'task_loss': task_loss / num_batches
        }

    def _validate_epoch_simple(self, model: nn.Module, val_loader: DataLoader,
                             is_classification: bool) -> Dict[str, float]:
        """Simple validation for basic models."""
        model.eval()

        total_loss = 0.0
        num_batches = 0

        if is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(batch_x)
                        if is_classification:
                            if outputs.size(1) > 1:
                                loss = criterion(outputs, batch_y.squeeze().long())
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    outputs.squeeze(), batch_y.squeeze().float()
                                )
                        else:
                            loss = criterion(outputs.squeeze(), batch_y.squeeze().float())
                else:
                    outputs = model(batch_x)
                    if is_classification:
                        if outputs.size(1) > 1:
                            loss = criterion(outputs, batch_y.squeeze().long())
                        else:
                            loss = F.binary_cross_entropy_with_logits(
                                outputs.squeeze(), batch_y.squeeze().float()
                            )
                    else:
                        loss = criterion(outputs.squeeze(), batch_y.squeeze().float())

                total_loss += loss.item()
                num_batches += 1

        return {'loss': total_loss / num_batches}

    def _save_model(self, model: nn.Module, experiment_id: str,
                   training_history: Dict[str, Any]):
        """Save model and training history."""
        output_dir = self.config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)

        # Save model state
        model_path = os.path.join(output_dir, f"{experiment_id}_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'config': self.config
        }, model_path)

        # Save training history
        history_path = os.path.join(output_dir, f"{experiment_id}_history.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in training_history.items():
            if isinstance(value, list):
                serializable_history[key] = [
                    float(v) if isinstance(v, (np.float32, np.float64, np.integer)) else v
                    for v in value
                ]
            else:
                serializable_history[key] = value

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        logger.info(f"Saved model and history for {experiment_id}")

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

class HyperparameterOptimizer:
    """Hyperparameter optimization for fair comparison."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_hyperparams = config.get('baseline_hyperparams', {})

    def optimize_hyperparameters(self, model_type: str, train_func: Callable,
                                dataset_info: Dict[str, Any],
                                max_trials: int = 20) -> Dict[str, Any]:
        """Optimize hyperparameters for given model type."""

        logger.info(f"Optimizing hyperparameters for {model_type}")

        # Get hyperparameter space
        param_space = self._get_parameter_space(model_type)

        best_score = -float('inf') if dataset_info['is_classification'] else float('inf')
        best_params = {}

        # Random search (can be replaced with more sophisticated methods)
        for trial in range(max_trials):
            # Sample parameters
            params = self._sample_parameters(param_space)

            try:
                # Train and evaluate model
                result = train_func(params)

                # Extract validation score
                if dataset_info['is_classification']:
                    score = result.get('val_accuracy', result.get('best_val_loss', 0))
                    is_better = score > best_score
                else:
                    score = result.get('val_mse', result.get('best_val_loss', float('inf')))
                    is_better = score < best_score

                if is_better:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"Trial {trial}: New best score = {score:.4f}, params = {params}")
                else:
                    logger.debug(f"Trial {trial}: Score = {score:.4f}")

            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue

        logger.info(f"Hyperparameter optimization completed. Best score: {best_score:.4f}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trials_completed': max_trials
        }

    def _get_parameter_space(self, model_type: str) -> Dict[str, List]:
        """Get hyperparameter space for model type."""

        if model_type not in self.baseline_hyperparams:
            return {}

        return self.baseline_hyperparams[model_type]

    def _sample_parameters(self, param_space: Dict[str, List]) -> Dict[str, Any]:
        """Sample parameters from parameter space."""
        params = {}

        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, dict):
                if param_values.get('type') == 'uniform':
                    low, high = param_values['range']
                    params[param_name] = np.random.uniform(low, high)
                elif param_values.get('type') == 'log_uniform':
                    low, high = param_values['range']
                    params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))

        return params

# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class EvaluationEngine:
    """Comprehensive model evaluation with statistical metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_samples = config.get('evaluation_config', {}).get('bootstrap_samples', 1000)
        self.confidence_level = config.get('evaluation_config', {}).get('confidence_level', 0.95)

    def evaluate_model(self, model: nn.Module, test_loader: DataLoader,
                      is_classification: bool, device: torch.device) -> Dict[str, Any]:
        """Comprehensive model evaluation with confidence intervals."""

        logger.info("Evaluating model performance")

        model.eval()
        model = model.to(device)

        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)

                if is_classification:
                    if outputs.size(1) > 1:
                        probabilities = F.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                    else:
                        probabilities = torch.sigmoid(outputs)
                        predictions = (probabilities > 0.5).long()

                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                else:
                    all_predictions.extend(outputs.squeeze().cpu().numpy())

                all_targets.extend(batch_y.squeeze().cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        if is_classification:
            probabilities = np.array(all_probabilities)

            # Classification metrics

            metrics = MetricsCalculator.calculate_classification_metrics(
                targets, predictions, probabilities
            )

            # Bootstrap confidence intervals for classification
            bootstrap_metrics = self._bootstrap_classification_metrics(
                targets, predictions, probabilities
            )

        else:
            # Regression metrics

            metrics = MetricsCalculator.calculate_regression_metrics(targets, predictions)

            # Bootstrap confidence intervals for regression
            bootstrap_metrics = self._bootstrap_regression_metrics(targets, predictions)

        # Combine metrics with confidence intervals
        final_metrics = {}
        for key, value in metrics.items():
            final_metrics[key] = {
                'value': value,
                'confidence_interval': bootstrap_metrics.get(key, (value, value))
            }

        logger.info(f"Model evaluation completed. Primary metric: {list(metrics.keys())[0]} = {list(metrics.values())[0]:.4f}")

        return {
            'metrics': final_metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities if is_classification else None
        }

    def _bootstrap_classification_metrics(self, targets: np.ndarray,
                                        predictions: np.ndarray,
                                        probabilities: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for classification metrics."""

        n_samples = len(targets)
        bootstrap_results = defaultdict(list)

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_targets = targets[indices]
            boot_predictions = predictions[indices]
            boot_probabilities = probabilities[indices]

            # Compute metrics
            try:
                boot_metrics = MetricsCalculator.calculate_classification_metrics(
                    boot_targets, boot_predictions, boot_probabilities
                )

                for key, value in boot_metrics.items():
                    bootstrap_results[key].append(value)
            except Exception:
                continue

        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.confidence_level

        for key, values in bootstrap_results.items():
            if values:
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                ci_lower = np.percentile(values, lower_percentile)
                ci_upper = np.percentile(values, upper_percentile)

                confidence_intervals[key] = (ci_lower, ci_upper)

        return confidence_intervals

    def _bootstrap_regression_metrics(self, targets: np.ndarray,
                                    predictions: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for regression metrics."""

        n_samples = len(targets)
        bootstrap_results = defaultdict(list)

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_targets = targets[indices]
            boot_predictions = predictions[indices]

            # Compute metrics
            try:

                boot_metrics = MetricsCalculator.calculate_regression_metrics(
                    boot_targets, boot_predictions
                )

                for key, value in boot_metrics.items():
                    bootstrap_results[key].append(value)
            except Exception:
                continue

        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.confidence_level

        for key, values in bootstrap_results.items():
            if values:
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                ci_lower = np.percentile(values, lower_percentile)
                ci_upper = np.percentile(values, upper_percentile)

                confidence_intervals[key] = (ci_lower, ci_upper)

        return confidence_intervals

#!/usr/bin/env python3
"""
Universal K Matrix - Parallel Execution and Experiment Management
True multi-GPU parallelization with comprehensive experiment orchestration
"""


# =============================================================================
# EXPERIMENT DEFINITION
# =============================================================================

class ExperimentType(Enum):
    UNIVERSAL_K = "universal_k"
    SOTA_BASELINE = "sota_baseline"
    ENHANCED_SOTA = "enhanced_sota"

@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""
    experiment_id: str
    experiment_type: ExperimentType
    dataset_name: str
    method_name: str
    hyperparams: Dict[str, Any]
    num_factors: Optional[int] = None
    latent_dim: Optional[int] = None
    baseline_method: Optional[str] = None
    k_method: Optional[str] = None
    random_seed: int = 42
    priority: int = 1  # Higher priority = runs first

@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    experiment_id: str
    experiment_spec: ExperimentSpec
    success: bool
    metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    training_time: float
    gpu_id: Optional[int] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None

# =============================================================================
# EXPERIMENT QUEUE MANAGER
# =============================================================================

class ExperimentQueueManager:
    """Manages experiment queue with prioritization and load balancing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments = []
        self.completed_experiments = []
        self.failed_experiments = []
        self.lock = Lock()

    def generate_experiments(self, dataset_names: List[str]) -> List[ExperimentSpec]:
        """Generate all experiment specifications."""

        experiments = []
        experiment_counter = 0

        # Get configuration
        k_methods = self.config.get('k_methods', ['PCA', 'FactorAnalysis', 'Clustered', 'Random'])
        baseline_methods = self.config.get('baseline_methods', ['VIB', 'BetaVAE', 'SparseAutoencoder', 'StandardAutoencoder'])
        factors_to_try = self.config.get('factors_to_try', [3, 5])
        latent_dimensions = self.config.get('latent_dimensions', [8])
        random_seeds = self.config.get('random_seeds', [42])

        for dataset_name in dataset_names:
            for seed in random_seeds:

                # 1. Universal K Matrix experiments (alone)
                for k_method in k_methods:
                    for num_factors in factors_to_try:
                        for latent_dim in latent_dimensions:
                            experiment_counter += 1
                            experiments.append(ExperimentSpec(
                                experiment_id=f"uk_{experiment_counter:06d}",
                                experiment_type=ExperimentType.UNIVERSAL_K,
                                dataset_name=dataset_name,
                                method_name=k_method,
                                hyperparams={},
                                num_factors=num_factors,
                                latent_dim=latent_dim,
                                k_method=k_method,
                                random_seed=seed,
                                priority=3  # High priority
                            ))

                # 2. SOTA Baseline experiments (alone)
                for baseline_method in baseline_methods:
                    for latent_dim in latent_dimensions:
                        # Get hyperparameter combinations for this baseline
                        hyperparams_list = self._get_hyperparameter_combinations(baseline_method)

                        for hyperparams in hyperparams_list:
                            experiment_counter += 1
                            experiments.append(ExperimentSpec(
                                experiment_id=f"sota_{experiment_counter:06d}",
                                experiment_type=ExperimentType.SOTA_BASELINE,
                                dataset_name=dataset_name,
                                method_name=baseline_method,
                                hyperparams=hyperparams,
                                latent_dim=latent_dim,
                                baseline_method=baseline_method,
                                random_seed=seed,
                                priority=2  # Medium priority
                            ))

                # 3. Enhanced SOTA experiments (Universal K + SOTA)
                for baseline_method in baseline_methods:
                    for k_method in k_methods:
                        for num_factors in factors_to_try:
                            for latent_dim in latent_dimensions:
                                # Sample fewer hyperparameter combinations for enhanced experiments
                                hyperparams_list = self._get_hyperparameter_combinations(baseline_method, max_combinations=2)

                                for hyperparams in hyperparams_list:
                                    experiment_counter += 1
                                    experiments.append(ExperimentSpec(
                                        experiment_id=f"enhanced_{experiment_counter:06d}",
                                        experiment_type=ExperimentType.ENHANCED_SOTA,
                                        dataset_name=dataset_name,
                                        method_name=f"{baseline_method}+{k_method}",
                                        hyperparams=hyperparams,
                                        num_factors=num_factors,
                                        latent_dim=latent_dim,
                                        baseline_method=baseline_method,
                                        k_method=k_method,
                                        random_seed=seed,
                                        priority=1  # Lower priority
                                    ))

        # Sort by priority (higher priority first)
        experiments.sort(key=lambda x: x.priority, reverse=True)

        logger.info(f"Generated {len(experiments)} experiments across {len(dataset_names)} datasets")
        logger.info(f"Universal K: {sum(1 for e in experiments if e.experiment_type == ExperimentType.UNIVERSAL_K)}")
        logger.info(f"SOTA Baseline: {sum(1 for e in experiments if e.experiment_type == ExperimentType.SOTA_BASELINE)}")
        logger.info(f"Enhanced SOTA: {sum(1 for e in experiments if e.experiment_type == ExperimentType.ENHANCED_SOTA)}")

        return experiments

    def _get_hyperparameter_combinations(self, baseline_method: str, max_combinations: int = None) -> List[Dict[str, Any]]:
        """Get hyperparameter combinations for baseline method."""

        baseline_hyperparams = self.config.get('baseline_hyperparams', {})

        if baseline_method not in baseline_hyperparams:
            return [{}]

        method_params = baseline_hyperparams[baseline_method]
        combinations = []

        if baseline_method == 'VIB':
            beta_values = method_params.get('beta_values', [1.0])
            combinations = [{'beta': beta} for beta in beta_values]

        elif baseline_method == 'BetaVAE':
            beta_values = method_params.get('beta_values', [4.0])
            combinations = [{'beta': beta} for beta in beta_values]

        elif baseline_method == 'SparseAutoencoder':
            sparsity_weights = method_params.get('sparsity_weights', [0.01])
            combinations = [{'sparsity_weight': sw} for sw in sparsity_weights]

        elif baseline_method == 'StandardAutoencoder':
            dropout_rates = method_params.get('dropout_rates', [0.3])
            combinations = [{'dropout_rate': dr} for dr in dropout_rates]

        else:
            combinations = [{}]

        # Sample subset if max_combinations is specified
        if max_combinations and len(combinations) > max_combinations:
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]

        return combinations

    def add_experiments(self, experiments: List[ExperimentSpec]):
        """Add experiments to queue."""
        with self.lock:
            self.experiments.extend(experiments)

    def get_next_experiment(self) -> Optional[ExperimentSpec]:
        """Get next experiment from queue."""
        with self.lock:
            if self.experiments:
                return self.experiments.pop(0)
            return None

    def add_completed_experiment(self, result: ExperimentResult):
        """Add completed experiment result."""
        with self.lock:
            if result.success:
                self.completed_experiments.append(result)
            else:
                self.failed_experiments.append(result)

    def get_all_experiments(self) -> List[ExperimentSpec]:
        """Get all experiments as a list"""
        with self.lock:
            return self.experiments.copy()

    def get_progress(self) -> Dict[str, int]:
        """Get current progress statistics."""
        with self.lock:
            return {
                'pending': len(self.experiments),
                'completed': len(self.completed_experiments),
                'failed': len(self.failed_experiments),
                'total': len(self.experiments) + len(self.completed_experiments) + len(self.failed_experiments)
            }

# =============================================================================
# GPU WORKER PROCESS
# =============================================================================

class GPUWorker:
    """Worker process that runs experiments on a specific GPU."""

    def __init__(self, gpu_id: int, config: Dict[str, Any]):
        self.gpu_id = gpu_id
        self.config = config
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu')

        if gpu_id >= 0:
            torch.cuda.set_device(self.device)

        # Initialize components directly (no imports needed)
        self.device_manager = DeviceManager(use_all_gpus=False)
        self.dataset_manager = DatasetManager(config=config)
        self.data_splitter = DataSplitter(config)
        self.dataloader_factory = DataLoaderFactory(config)
        self.model_factory = ModelFactory()
        self.k_matrix_initializer = KMatrixInitializer(random_seed=42)
        self.k_matrix_refiner = KMatrixRefiner(config)
        self.k_matrix_evaluator = KMatrixEvaluator(config)
        self.training_engine = TrainingEngine(config, self.device)
        self.evaluation_engine = EvaluationEngine(config)

        logger.info(f"Initialized GPU worker for device {self.device}")

    def run_worker(self, experiment_specs: List[ExperimentSpec]) -> List[ExperimentResult]:
        """Main worker loop - now takes experiment list instead of queue"""

        # Initialize CUDA context in worker process
        if self.gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.device = torch.device(f'cuda:{self.gpu_id}')
            # Warm up CUDA context
            dummy = torch.cuda.FloatTensor([0])
            del dummy
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        # Set random seeds for this worker
        set_random_seeds(42 + self.gpu_id)

        # Initialize components in worker process
        self.device_manager = DeviceManager(use_all_gpus=False)
        self.dataset_manager = DatasetManager(config=self.config)
        self.data_splitter = DataSplitter(self.config)
        self.dataloader_factory = DataLoaderFactory(self.config)
        self.model_factory = ModelFactory()
        self.k_matrix_initializer = KMatrixInitializer(random_seed=42 + self.gpu_id)
        self.k_matrix_refiner = KMatrixRefiner(self.config)
        self.k_matrix_evaluator = KMatrixEvaluator(self.config)
        self.training_engine = TrainingEngine(self.config, self.device)
        self.evaluation_engine = EvaluationEngine(self.config)

        print(f"Worker {self.gpu_id} initialized on device {self.device}")

        # Process experiments
        results = []
        for exp in experiment_specs:
            try:
                result = self.run_experiment(exp)
                results.append(result)

                # Clean GPU memory after each experiment
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            except Exception as e:
                print(f"Worker {self.gpu_id} error on {exp.experiment_id}: {e}")
                traceback.print_exc()

                # Create failed result
                result = ExperimentResult(
                    experiment_id=exp.experiment_id,
                    experiment_spec=exp,
                    success=False,
                    metrics={},
                    performance_metrics={},
                    training_time=0.0,
                    gpu_id=self.gpu_id,
                    error_message=str(e)
                )
                results.append(result)

        return results

    def run_experiment(self, experiment: ExperimentSpec) -> ExperimentResult:
        """Run a single experiment."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


        # Set random seed for reproducibility
        set_random_seeds(experiment.random_seed)

        start_time = time.time()

        logger.info(f"GPU {self.gpu_id}: Running {experiment.experiment_id} - {experiment.experiment_type.value}")

        try:
            if experiment.experiment_type == ExperimentType.UNIVERSAL_K:
                result = self._run_universal_k_experiment(experiment)
            elif experiment.experiment_type == ExperimentType.SOTA_BASELINE:
                result = self._run_sota_baseline_experiment(experiment)
            elif experiment.experiment_type == ExperimentType.ENHANCED_SOTA:
                result = self._run_enhanced_sota_experiment(experiment)
            else:
                raise ValueError(f"Unknown experiment type: {experiment.experiment_type}")

            training_time = time.time() - start_time

            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                experiment_spec=experiment,
                success=True,
                metrics=result['metrics'],
                performance_metrics=result['performance_metrics'],
                training_time=training_time,
                gpu_id=self.gpu_id,
                model_path=result.get('model_path')
            )

        except Exception as e:
            training_time = time.time() - start_time
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"Experiment {experiment.experiment_id} failed: {e}")

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                experiment_spec=experiment,
                success=False,
                metrics={},
                performance_metrics={},
                training_time=training_time,
                gpu_id=self.gpu_id,
                error_message=str(e)
            )

    def _run_universal_k_experiment(self, experiment: ExperimentSpec) -> Dict[str, Any]:
        """Run Universal K Matrix experiment."""

        # Load dataset
        x_data, y_data, is_classification, metadata = self.dataset_manager.load_dataset(experiment.dataset_name)

        # Create data splits
        data_splits = self.data_splitter.create_train_val_test_split(
            x_data, y_data, is_classification, experiment.random_seed
        )

        # Move to device
        for key in data_splits:
            data_splits[key] = data_splits[key].to(self.device)

        # Initialize K-matrix
        k_matrix = self.k_matrix_initializer.initialize_k_matrix(
            experiment.k_method, data_splits['x_train'],
            experiment.num_factors, experiment.latent_dim, self.device
        )

        # Refine K-matrix
        k_matrix_refined = self.k_matrix_refiner.refine_k_matrix(
            data_splits['x_train'], k_matrix,
            experiment.num_factors, experiment.latent_dim, self.device
        )

        # Evaluate K-matrix
        k_metrics = self.k_matrix_evaluator.evaluate_k_matrix(
            data_splits['x_train'], k_matrix_refined,
            experiment.num_factors, experiment.latent_dim, self.device
        )

        # Encode data with K-matrix


        train_z = encode_data_with_k_matrix(data_splits['x_train'], k_matrix_refined)
        val_z = encode_data_with_k_matrix(data_splits['x_val'], k_matrix_refined)
        test_z = encode_data_with_k_matrix(data_splits['x_test'], k_matrix_refined)

        # Flatten encoded data
        train_z_flat = train_z.reshape(train_z.shape[0], -1)
        val_z_flat = val_z.reshape(val_z.shape[0], -1)
        test_z_flat = test_z.reshape(test_z.shape[0], -1)

        # Create data loaders for encoded data
        encoded_splits = {
            'x_train': train_z_flat, 'x_val': val_z_flat, 'x_test': test_z_flat,
            'y_train': data_splits['y_train'], 'y_val': data_splits['y_val'], 'y_test': data_splits['y_test']
        }
        loaders = self.dataloader_factory.create_loaders(encoded_splits)

        # Train teacher model
        num_classes = metadata['n_classes'] if is_classification else 1
        teacher_model = self.model_factory.create_model(
            'Teacher', train_z_flat.shape[1], experiment.latent_dim,
            num_classes, is_classification, architecture_config=self.config.get('standard_architecture')
        ).to(self.device)

        teacher_results = self.training_engine.train_autoencoder_model(
            teacher_model, loaders['train'], loaders['val'], is_classification,
            experiment_id=f"{experiment.experiment_id}_teacher"
        )

        # Train student model (on original data)
        original_loaders = self.dataloader_factory.create_loaders(data_splits)
        student_model = self.model_factory.create_model(
            'Student', x_data.shape[1], experiment.latent_dim,
            num_classes, is_classification, architecture_config=self.config.get('standard_architecture')
        ).to(self.device)

        # Create teacher wrapper for distillation

        teacher_wrapper = KMatrixTeacherWrapper(k_matrix_refined, teacher_model).to(self.device)

        student_results = self.training_engine.train_teacher_student(
            teacher_wrapper, student_model, loaders['train'], original_loaders['train'],
            original_loaders['val'], is_classification,
            experiment_id=f"{experiment.experiment_id}_student"
        )

        # Evaluate on test set
        teacher_eval = self.evaluation_engine.evaluate_model(
            teacher_model, loaders['test'], is_classification, self.device
        )
        student_eval = self.evaluation_engine.evaluate_model(
            student_model, original_loaders['test'], is_classification, self.device
        )

        return {
            'metrics': k_metrics,
            'performance_metrics': {
                'teacher': {k: v['value'] for k, v in teacher_eval['metrics'].items()},
                'student': {k: v['value'] for k, v in student_eval['metrics'].items()}
            },
            'model_path': None  # Could save models here if needed
        }

    def _run_sota_baseline_experiment(self, experiment: ExperimentSpec) -> Dict[str, Any]:
        """Run SOTA baseline experiment."""

        # Load dataset
        x_data, y_data, is_classification, metadata = self.dataset_manager.load_dataset(experiment.dataset_name)

        # Create data splits
        data_splits = self.data_splitter.create_train_val_test_split(
            x_data, y_data, is_classification, experiment.random_seed
        )

        # Move to device
        for key in data_splits:
            data_splits[key] = data_splits[key].to(self.device)

        # Create data loaders
        loaders = self.dataloader_factory.create_loaders(data_splits)

        # Create model
        num_classes = metadata['n_classes'] if is_classification else 1
        model = self.model_factory.create_model(
            experiment.baseline_method, x_data.shape[1], experiment.latent_dim,
            num_classes, is_classification, experiment.hyperparams,
            architecture_config=self.config.get('standard_architecture')
        ).to(self.device)

        # Train model
        training_results = self.training_engine.train_autoencoder_model(
            model, loaders['train'], loaders['val'], is_classification,
            experiment.hyperparams, experiment_id=experiment.experiment_id
        )

        # Evaluate on test set
        evaluation_results = self.evaluation_engine.evaluate_model(
            model, loaders['test'], is_classification, self.device
        )

        # Extract latent representations for disentanglement metrics
        model.eval()
        test_latents = []
        test_inputs = []

        with torch.no_grad():
            for batch_x, _ in loaders['test']:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)

                if 'z' in outputs:
                    test_latents.append(outputs['z'])
                elif hasattr(model, 'encode'):
                    if experiment.baseline_method in ['VIB', 'BetaVAE']:
                        mu, _ = model.encode(batch_x)
                        test_latents.append(mu)
                    else:
                        z = model.encode(batch_x)
                        test_latents.append(z)

                test_inputs.append(batch_x)

        if test_latents:
            all_latents = torch.cat(test_latents, dim=0)
            all_inputs = torch.cat(test_inputs, dim=0)

            z_reshaped = all_latents.unsqueeze(1)  # Add factor dimension
            disentanglement_metrics = MetricsCalculator.calculate_disentanglement_metrics(
                z_reshaped, all_inputs, 1, experiment.latent_dim
            )
        else:
            disentanglement_metrics = {}

        return {
            'metrics': disentanglement_metrics,
            'performance_metrics': {k: v['value'] for k, v in evaluation_results['metrics'].items()},
            'model_path': None
        }

    def _run_enhanced_sota_experiment(self, experiment: ExperimentSpec) -> Dict[str, Any]:
        """Run Enhanced SOTA experiment (Universal K + SOTA)."""

        # Load dataset
        x_data, y_data, is_classification, metadata = self.dataset_manager.load_dataset(experiment.dataset_name)

        # Create data splits
        data_splits = self.data_splitter.create_train_val_test_split(
            x_data, y_data, is_classification, experiment.random_seed
        )

        # Move to device
        for key in data_splits:
            data_splits[key] = data_splits[key].to(self.device)

        # Step 1: Initialize and refine K-matrix
        k_matrix = self.k_matrix_initializer.initialize_k_matrix(
            experiment.k_method, data_splits['x_train'],
            experiment.num_factors, experiment.latent_dim, self.device
        )

        k_matrix_refined = self.k_matrix_refiner.refine_k_matrix(
            data_splits['x_train'], k_matrix,
            experiment.num_factors, experiment.latent_dim, self.device
        )

        # Step 2: Encode data with K-matrix


        train_z = encode_data_with_k_matrix(data_splits['x_train'], k_matrix_refined)
        val_z = encode_data_with_k_matrix(data_splits['x_val'], k_matrix_refined)
        test_z = encode_data_with_k_matrix(data_splits['x_test'], k_matrix_refined)

        # Flatten encoded data
        train_z_flat = train_z.reshape(train_z.shape[0], -1)
        val_z_flat = val_z.reshape(val_z.shape[0], -1)
        test_z_flat = test_z.reshape(test_z.shape[0], -1)

        # Step 3: Train SOTA method on K-encoded data
        encoded_splits = {
            'x_train': train_z_flat, 'x_val': val_z_flat, 'x_test': test_z_flat,
            'y_train': data_splits['y_train'], 'y_val': data_splits['y_val'], 'y_test': data_splits['y_test']
        }
        encoded_loaders = self.dataloader_factory.create_loaders(encoded_splits)

        # Create enhanced model
        num_classes = metadata['n_classes'] if is_classification else 1
        enhanced_model = self.model_factory.create_model(
            experiment.baseline_method, train_z_flat.shape[1], experiment.latent_dim,
            num_classes, is_classification, experiment.hyperparams,
            architecture_config=self.config.get('standard_architecture')
        ).to(self.device)

        # Train enhanced model
        training_results = self.training_engine.train_autoencoder_model(
            enhanced_model, encoded_loaders['train'], encoded_loaders['val'],
            is_classification, experiment.hyperparams, experiment_id=experiment.experiment_id
        )

        # Evaluate enhanced model
        evaluation_results = self.evaluation_engine.evaluate_model(
            enhanced_model, encoded_loaders['test'], is_classification, self.device
        )

        # Evaluate K-matrix quality
        k_metrics = self.k_matrix_evaluator.evaluate_k_matrix(
            data_splits['x_train'], k_matrix_refined,
            experiment.num_factors, experiment.latent_dim, self.device
        )

        # Extract enhanced latent representations
        enhanced_model.eval()
        test_latents = []

        with torch.no_grad():
            for batch_x, _ in encoded_loaders['test']:
                batch_x = batch_x.to(self.device)
                outputs = enhanced_model(batch_x)

                if 'z' in outputs:
                    test_latents.append(outputs['z'])
                elif hasattr(enhanced_model, 'encode'):
                    if experiment.baseline_method in ['VIB', 'BetaVAE']:
                        mu, _ = enhanced_model.encode(batch_x)
                        test_latents.append(mu)
                    else:
                        z = enhanced_model.encode(batch_x)
                        test_latents.append(z)

        if test_latents:
            all_latents = torch.cat(test_latents, dim=0)
            z_reshaped = all_latents.unsqueeze(1)  # Add factor dimension
            enhanced_disentanglement_metrics = MetricsCalculator.calculate_disentanglement_metrics(
                z_reshaped, test_z_flat, 1, experiment.latent_dim
            )
        else:
            enhanced_disentanglement_metrics = {}

        # Combine K-matrix metrics with enhanced model metrics
        combined_metrics = {**k_metrics, **enhanced_disentanglement_metrics}

        return {
            'metrics': combined_metrics,
            'performance_metrics': {k: v['value'] for k, v in evaluation_results['metrics'].items()},
            'model_path': None
        }

# =============================================================================
# PARALLEL EXPERIMENT ORCHESTRATOR
# =============================================================================

class ParallelExperimentOrchestrator:
    """Orchestrates parallel execution of experiments across multiple GPUs."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_manager = DeviceManager(config.get('use_all_gpus', True))
        self.num_workers = len(self.device_manager.available_devices)
        self.experiment_queue = ExperimentQueueManager(config)

        logger.info(f"Initialized orchestrator with {self.num_workers} workers")
        logger.info(f"Available devices: {[str(d) for d in self.device_manager.available_devices]}")

    def run_comprehensive_experiments(self, dataset_names: List[str]) -> pd.DataFrame:
        """Run comprehensive experiments across all datasets and methods."""

        logger.info("Starting comprehensive experiment suite")

        # Generate all experiments
        experiments = self.experiment_queue.generate_experiments(dataset_names)
        self.experiments = experiments

        # Start progress monitoring
        progress_thread = threading.Thread(target=self._monitor_progress)
        progress_thread.daemon = True
        progress_thread.start()

        # Start GPU workers
        if self.num_workers > 1:
            results = self._run_parallel_experiments()
        else:
            results = self._run_sequential_experiments()

        # Convert results to DataFrame
        results_df = self._create_results_dataframe(results)

        logger.info(f"Completed {len(results)} experiments")
        return results_df

    def _run_parallel_experiments(self) -> List[ExperimentResult]:
        """Fixed parallel execution"""
        cpu_config = self.config.copy()

        # Create CPU-only experiment specs
        cpu_experiments = []
        for exp in self.experiments:
            # Ensure no CUDA tensors in experiment specs
            cpu_exp = ExperimentSpec(
                experiment_id=exp.experiment_id,
                experiment_type=exp.experiment_type,
                dataset_name=exp.dataset_name,
                method_name=exp.method_name,
                hyperparams=exp.hyperparams,
                num_factors=exp.num_factors,
                latent_dim=exp.latent_dim,
                baseline_method=exp.baseline_method,
                k_method=exp.k_method,
                random_seed=exp.random_seed,
                priority=exp.priority
            )
            cpu_experiments.append(cpu_exp)

        print(f"Starting {self.num_workers} parallel workers")

        # Get all experiments from the generated list
        all_experiments = self.experiments

        if not all_experiments:
            print("No experiments to run")
            return []

        print(f"Total experiments to run: {len(all_experiments)}")

        # Split experiments across workers
        experiment_chunks = [[] for _ in range(self.num_workers)]
        for i, exp in enumerate(all_experiments):
            experiment_chunks[i % self.num_workers].append(exp)

        # Use spawn context for CUDA compatibility
        mp_context = mp.get_context('spawn')

        # Clear CUDA cache before starting workers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Start worker processes
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_context) as executor:
            futures = []

            for i, exp_chunk in enumerate(experiment_chunks):
                if exp_chunk:  # Only submit if there are experiments
                    gpu_id = self.device_manager.available_devices[i].index if self.device_manager.available_devices[i].type == 'cuda' else -1
                    future = executor.submit(worker_process_main, gpu_id, self.config, exp_chunk)
                    futures.append(future)

            # Collect results
            all_results = []
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                    print(f"Worker completed with {len(worker_results)} results")
                except Exception as e:
                    print(f"Worker process failed: {e}")
                    traceback.print_exc()

        return all_results

    def _run_sequential_experiments(self) -> List[ExperimentResult]:
        """Run experiments sequentially on single GPU/CPU."""

        logger.info("Running experiments sequentially")

        device = self.device_manager.available_devices[0]
        gpu_id = device.index if device.type == 'cuda' else -1

        # Just use the worker process main function
        return worker_process_main(gpu_id, self.config, self.experiments)

    def _monitor_progress(self):
        """Monitor and log experiment progress."""

        total_experiments = len(self.experiments)
        last_completed = 0

        while True:
            time.sleep(30)  # Update every 30 seconds

            # Get progress from queue manager
            progress = self.experiment_queue.get_progress()
            completed = progress['completed'] + progress['failed']

            if completed != last_completed:
                completion_rate = (completed / total_experiments) * 100 if total_experiments > 0 else 0

                logger.info(f"Progress: {completion_rate:.1f}% complete ({completed}/{total_experiments})")
                last_completed = completed

            # Exit if all experiments completed
            if completed >= total_experiments:
                break

    def _create_results_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Convert experiment results to comprehensive DataFrame."""

        rows = []

        for result in results:
            if not result.success:
                continue

            spec = result.experiment_spec

            # Base row data
            row = {
                'experiment_id': result.experiment_id,
                'experiment_type': spec.experiment_type.value,
                'dataset': spec.dataset_name,
                'method': spec.method_name,
                'baseline_method': spec.baseline_method or '',
                'k_method': spec.k_method or '',
                'num_factors': spec.num_factors or 0,
                'latent_dim': spec.latent_dim or 0,
                'random_seed': spec.random_seed,
                'training_time': result.training_time,
                'gpu_id': result.gpu_id or -1,
                'success': result.success
            }

            # Add hyperparameters
            for key, value in spec.hyperparams.items():
                row[f'hyperparam_{key}'] = value

            # Add metrics
            for key, value in result.metrics.items():
                row[f'metric_{key}'] = value

            # Add performance metrics
            if isinstance(result.performance_metrics, dict):
                if 'teacher' in result.performance_metrics:
                    for key, value in result.performance_metrics['teacher'].items():
                        row[f'teacher_{key}'] = value

                if 'student' in result.performance_metrics:
                    for key, value in result.performance_metrics['student'].items():
                        row[f'student_{key}'] = value

                # Handle flat performance metrics
                for key, value in result.performance_metrics.items():
                    if key not in ['teacher', 'student']:
                        row[f'perf_{key}'] = value

            rows.append(row)

        if not rows:
            logger.warning("No successful experiments to create DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')

        logger.info(f"Created results DataFrame with {len(df)} rows and {len(df.columns)} columns")

        return df

#!/usr/bin/env python3
"""
Universal K Matrix - Results Analysis and Reporting
Comprehensive statistical analysis with single CSV output
"""



# =============================================================================
# COMPREHENSIVE STATISTICAL ANALYZER
# =============================================================================

class ComprehensiveStatisticalAnalyzer:
    """All-in-one statistical analysis with single CSV output."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_level = config.get('evaluation_config', {}).get('confidence_level', 0.95)

    def analyze_and_export(self, results_df: pd.DataFrame, output_dir: str) -> str:
        """Perform comprehensive analysis and export to single CSV."""

        logger.info("Starting comprehensive statistical analysis")

        os.makedirs(output_dir, exist_ok=True)

        # Create comprehensive analysis DataFrame
        analysis_df = self._create_comprehensive_analysis_df(results_df)

        # Export to single CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'comprehensive_analysis_{timestamp}.csv')
        analysis_df.to_csv(output_path, index=False)

        logger.info(f"Comprehensive analysis exported to {output_path}")
        logger.info(f"Analysis contains {len(analysis_df)} rows and {len(analysis_df.columns)} columns")

        return output_path

    def _create_comprehensive_analysis_df(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create single comprehensive analysis DataFrame."""

        analysis_rows = []

        # 1. Overall summary statistics
        analysis_rows.extend(self._get_overall_summary_stats(results_df))

        # 2. Dataset-level statistics
        analysis_rows.extend(self._get_dataset_level_stats(results_df))

        # 3. Method-level statistics
        analysis_rows.extend(self._get_method_level_stats(results_df))

        # 4. Experiment type statistics
        analysis_rows.extend(self._get_experiment_type_stats(results_df))

        # 5. Performance metric statistics
        analysis_rows.extend(self._get_performance_metric_stats(results_df))

        # 6. Disentanglement metric statistics
        analysis_rows.extend(self._get_disentanglement_metric_stats(results_df))

        # 7. Hyperparameter statistics
        analysis_rows.extend(self._get_hyperparameter_stats(results_df))

        # 8. Pairwise comparisons
        analysis_rows.extend(self._get_pairwise_comparisons(results_df))

        # 9. Statistical significance tests
        analysis_rows.extend(self._get_significance_tests(results_df))

        # 10. Correlation analysis
        analysis_rows.extend(self._get_correlation_analysis(results_df))

        return pd.DataFrame(analysis_rows)

    def _get_overall_summary_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Overall summary statistics."""

        rows = []

        # Basic counts
        rows.append({
            'analysis_category': 'overall_summary',
            'analysis_type': 'basic_counts',
            'dataset': 'ALL',
            'method': 'ALL',
            'experiment_type': 'ALL',
            'metric_name': 'total_experiments',
            'count': len(df),
            'mean': len(df),
            'std': 0,
            'min': len(df),
            'max': len(df),
            'median': len(df)
        })

        rows.append({
            'analysis_category': 'overall_summary',
            'analysis_type': 'basic_counts',
            'dataset': 'ALL',
            'method': 'ALL',
            'experiment_type': 'ALL',
            'metric_name': 'unique_datasets',
            'count': len(df['dataset'].unique()),
            'mean': len(df['dataset'].unique()),
            'std': 0,
            'min': len(df['dataset'].unique()),
            'max': len(df['dataset'].unique()),
            'median': len(df['dataset'].unique())
        })

        rows.append({
            'analysis_category': 'overall_summary',
            'analysis_type': 'basic_counts',
            'dataset': 'ALL',
            'method': 'ALL',
            'experiment_type': 'ALL',
            'metric_name': 'unique_methods',
            'count': len(df['method'].unique()),
            'mean': len(df['method'].unique()),
            'std': 0,
            'min': len(df['method'].unique()),
            'max': len(df['method'].unique()),
            'median': len(df['method'].unique())
        })

        # Success rate if available
        if 'success' in df.columns:
            success_rate = df['success'].mean()
            rows.append({
                'analysis_category': 'overall_summary',
                'analysis_type': 'success_rate',
                'dataset': 'ALL',
                'method': 'ALL',
                'experiment_type': 'ALL',
                'metric_name': 'overall_success_rate',
                'count': len(df),
                'mean': success_rate,
                'std': df['success'].std(),
                'min': df['success'].min(),
                'max': df['success'].max(),
                'median': df['success'].median()
            })

        # Training time statistics if available
        if 'training_time' in df.columns:
            time_stats = df['training_time'].describe()
            rows.append({
                'analysis_category': 'overall_summary',
                'analysis_type': 'training_time',
                'dataset': 'ALL',
                'method': 'ALL',
                'experiment_type': 'ALL',
                'metric_name': 'training_time_seconds',
                'count': time_stats['count'],
                'mean': time_stats['mean'],
                'std': time_stats['std'],
                'min': time_stats['min'],
                'max': time_stats['max'],
                'median': time_stats['50%']
            })

        return rows

    def _get_dataset_level_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Dataset-level statistics."""

        rows = []

        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            # Basic dataset statistics
            rows.append({
                'analysis_category': 'dataset_level',
                'analysis_type': 'experiment_count',
                'dataset': dataset,
                'method': 'ALL',
                'experiment_type': 'ALL',
                'metric_name': 'experiments_per_dataset',
                'count': len(dataset_df),
                'mean': len(dataset_df),
                'std': 0,
                'min': len(dataset_df),
                'max': len(dataset_df),
                'median': len(dataset_df)
            })

            # Experiment type distribution
            for exp_type in dataset_df['experiment_type'].unique():
                type_count = len(dataset_df[dataset_df['experiment_type'] == exp_type])
                rows.append({
                    'analysis_category': 'dataset_level',
                    'analysis_type': 'experiment_type_distribution',
                    'dataset': dataset,
                    'method': 'ALL',
                    'experiment_type': exp_type,
                    'metric_name': 'experiments_per_type',
                    'count': type_count,
                    'mean': type_count,
                    'std': 0,
                    'min': type_count,
                    'max': type_count,
                    'median': type_count
                })

            # Performance metrics for this dataset
            numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns
            performance_cols = [col for col in numeric_cols if any(keyword in col.lower()
                               for keyword in ['accuracy', 'mse', 'f1', 'precision', 'recall', 'r2', 'mae', 'rmse'])]

            for perf_col in performance_cols:
                if not dataset_df[perf_col].isna().all():
                    stats = dataset_df[perf_col].describe()
                    rows.append({
                        'analysis_category': 'dataset_level',
                        'analysis_type': 'performance_metrics',
                        'dataset': dataset,
                        'method': 'ALL',
                        'experiment_type': 'ALL',
                        'metric_name': perf_col,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'median': stats['50%']
                    })

        return rows

    def _get_method_level_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Method-level statistics."""

        rows = []

        for method in df['method'].unique():
            method_df = df[df['method'] == method]

            # Basic method statistics
            rows.append({
                'analysis_category': 'method_level',
                'analysis_type': 'experiment_count',
                'dataset': 'ALL',
                'method': method,
                'experiment_type': 'ALL',
                'metric_name': 'experiments_per_method',
                'count': len(method_df),
                'mean': len(method_df),
                'std': 0,
                'min': len(method_df),
                'max': len(method_df),
                'median': len(method_df)
            })

            # Performance metrics for this method
            numeric_cols = method_df.select_dtypes(include=[np.number]).columns
            performance_cols = [col for col in numeric_cols if any(keyword in col.lower()
                               for keyword in ['accuracy', 'mse', 'f1', 'precision', 'recall', 'r2', 'mae', 'rmse'])]

            for perf_col in performance_cols:
                if not method_df[perf_col].isna().all():
                    stats = method_df[perf_col].describe()
                    rows.append({
                        'analysis_category': 'method_level',
                        'analysis_type': 'performance_metrics',
                        'dataset': 'ALL',
                        'method': method,
                        'experiment_type': 'ALL',
                        'metric_name': perf_col,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'median': stats['50%']
                    })

        return rows

    def _get_experiment_type_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Experiment type statistics."""

        rows = []

        for exp_type in df['experiment_type'].unique():
            type_df = df[df['experiment_type'] == exp_type]

            # Basic experiment type statistics
            rows.append({
                'analysis_category': 'experiment_type_level',
                'analysis_type': 'experiment_count',
                'dataset': 'ALL',
                'method': 'ALL',
                'experiment_type': exp_type,
                'metric_name': 'experiments_per_type',
                'count': len(type_df),
                'mean': len(type_df),
                'std': 0,
                'min': len(type_df),
                'max': len(type_df),
                'median': len(type_df)
            })

            # Performance metrics for this experiment type
            numeric_cols = type_df.select_dtypes(include=[np.number]).columns
            performance_cols = [col for col in numeric_cols if any(keyword in col.lower()
                               for keyword in ['accuracy', 'mse', 'f1', 'precision', 'recall', 'r2', 'mae', 'rmse'])]

            for perf_col in performance_cols:
                if not type_df[perf_col].isna().all():
                    stats = type_df[perf_col].describe()
                    rows.append({
                        'analysis_category': 'experiment_type_level',
                        'analysis_type': 'performance_metrics',
                        'dataset': 'ALL',
                        'method': 'ALL',
                        'experiment_type': exp_type,
                        'metric_name': perf_col,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max'],
                        'median': stats['50%']
                    })

        return rows

    def _get_performance_metric_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Performance metric statistics across all dimensions."""

        rows = []

        # Find all performance metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        performance_cols = [col for col in numeric_cols if any(keyword in col.lower()
                           for keyword in ['accuracy', 'mse', 'f1', 'precision', 'recall', 'r2', 'mae', 'rmse'])]

        for perf_col in performance_cols:
            if not df[perf_col].isna().all():
                # Overall statistics
                stats = df[perf_col].describe()
                rows.append({
                    'analysis_category': 'performance_metrics',
                    'analysis_type': 'overall_distribution',
                    'dataset': 'ALL',
                    'method': 'ALL',
                    'experiment_type': 'ALL',
                    'metric_name': perf_col,
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['50%']
                })

                # By dataset
                for dataset in df['dataset'].unique():
                    dataset_data = df[df['dataset'] == dataset][perf_col].dropna()
                    if len(dataset_data) > 0:
                        stats = dataset_data.describe()
                        rows.append({
                            'analysis_category': 'performance_metrics',
                            'analysis_type': 'by_dataset',
                            'dataset': dataset,
                            'method': 'ALL',
                            'experiment_type': 'ALL',
                            'metric_name': perf_col,
                            'count': stats['count'],
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'median': stats['50%']
                        })

                # By experiment type
                for exp_type in df['experiment_type'].unique():
                    type_data = df[df['experiment_type'] == exp_type][perf_col].dropna()
                    if len(type_data) > 0:
                        stats = type_data.describe()
                        rows.append({
                            'analysis_category': 'performance_metrics',
                            'analysis_type': 'by_experiment_type',
                            'dataset': 'ALL',
                            'method': 'ALL',
                            'experiment_type': exp_type,
                            'metric_name': perf_col,
                            'count': stats['count'],
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'median': stats['50%']
                        })

        return rows

    def _get_disentanglement_metric_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Disentanglement metric statistics."""

        rows = []

        # Find disentanglement metrics
        disentangle_cols = [col for col in df.columns if col.startswith('metric_') and
                           any(keyword in col for keyword in ['sparsity', 'modularity', 'total_correlation',
                                                           'factor_vae_score', 'sap_score', 'mig_score'])]

        for disentangle_col in disentangle_cols:
            if not df[disentangle_col].isna().all():
                # Overall statistics
                stats = df[disentangle_col].describe()
                rows.append({
                    'analysis_category': 'disentanglement_metrics',
                    'analysis_type': 'overall_distribution',
                    'dataset': 'ALL',
                    'method': 'ALL',
                    'experiment_type': 'ALL',
                    'metric_name': disentangle_col,
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['50%']
                })

                # By experiment type (most relevant for disentanglement)
                for exp_type in df['experiment_type'].unique():
                    type_data = df[df['experiment_type'] == exp_type][disentangle_col].dropna()
                    if len(type_data) > 0:
                        stats = type_data.describe()
                        rows.append({
                            'analysis_category': 'disentanglement_metrics',
                            'analysis_type': 'by_experiment_type',
                            'dataset': 'ALL',
                            'method': 'ALL',
                            'experiment_type': exp_type,
                            'metric_name': disentangle_col,
                            'count': stats['count'],
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'median': stats['50%']
                        })

        return rows

    def _get_hyperparameter_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Hyperparameter impact statistics."""

        rows = []

        # Find hyperparameter columns
        hyperparam_cols = [col for col in df.columns if col.startswith('hyperparam_')]

        for hyperparam_col in hyperparam_cols:
            param_name = hyperparam_col.replace('hyperparam_', '')

            # Skip if all values are the same
            unique_values = df[hyperparam_col].dropna().unique()
            if len(unique_values) <= 1:
                continue

            # Statistics for each parameter value
            for param_value in unique_values:
                param_df = df[df[hyperparam_col] == param_value]

                rows.append({
                    'analysis_category': 'hyperparameter_analysis',
                    'analysis_type': 'parameter_value_distribution',
                    'dataset': 'ALL',
                    'method': 'ALL',
                    'experiment_type': 'ALL',
                    'metric_name': f'{param_name}_{param_value}',
                    'count': len(param_df),
                    'mean': len(param_df),
                    'std': 0,
                    'min': len(param_df),
                    'max': len(param_df),
                    'median': len(param_df)
                })

        return rows

    def _get_pairwise_comparisons(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Pairwise comparisons between methods and experiment types."""

        rows = []

        # Method pairwise comparisons
        methods = df['method'].unique()
        performance_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                           if any(keyword in col.lower() for keyword in ['accuracy', 'mse', 'f1'])]

        if len(performance_cols) > 0:
            primary_metric = performance_cols[0]

            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    method1_data = df[df['method'] == method1][primary_metric].dropna()
                    method2_data = df[df['method'] == method2][primary_metric].dropna()

                    if len(method1_data) > 0 and len(method2_data) > 0:
                        # Basic comparison statistics
                        mean_diff = method1_data.mean() - method2_data.mean()

                        rows.append({
                            'analysis_category': 'pairwise_comparisons',
                            'analysis_type': 'method_comparison',
                            'dataset': 'ALL',
                            'method': f'{method1}_vs_{method2}',
                            'experiment_type': 'ALL',
                            'metric_name': f'mean_difference_{primary_metric}',
                            'count': len(method1_data) + len(method2_data),
                            'mean': mean_diff,
                            'std': np.sqrt(method1_data.var() + method2_data.var()),
                            'min': min(method1_data.min(), method2_data.min()),
                            'max': max(method1_data.max(), method2_data.max()),
                            'median': np.median(np.concatenate([method1_data.values, method2_data.values]))
                        })

        # Experiment type pairwise comparisons
        exp_types = df['experiment_type'].unique()

        if len(performance_cols) > 0 and len(exp_types) > 1:
            primary_metric = performance_cols[0]

            for i, type1 in enumerate(exp_types):
                for type2 in exp_types[i+1:]:
                    type1_data = df[df['experiment_type'] == type1][primary_metric].dropna()
                    type2_data = df[df['experiment_type'] == type2][primary_metric].dropna()

                    if len(type1_data) > 0 and len(type2_data) > 0:
                        mean_diff = type1_data.mean() - type2_data.mean()

                        rows.append({
                            'analysis_category': 'pairwise_comparisons',
                            'analysis_type': 'experiment_type_comparison',
                            'dataset': 'ALL',
                            'method': 'ALL',
                            'experiment_type': f'{type1}_vs_{type2}',
                            'metric_name': f'mean_difference_{primary_metric}',
                            'count': len(type1_data) + len(type2_data),
                            'mean': mean_diff,
                            'std': np.sqrt(type1_data.var() + type2_data.var()),
                            'min': min(type1_data.min(), type2_data.min()),
                            'max': max(type1_data.max(), type2_data.max()),
                            'median': np.median(np.concatenate([type1_data.values, type2_data.values]))
                        })

        return rows

    def _get_significance_tests(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Statistical significance tests."""

        rows = []

        # Find primary performance metric
        performance_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                           if any(keyword in col.lower() for keyword in ['accuracy', 'mse', 'f1'])]

        if len(performance_cols) == 0:
            return rows

        primary_metric = performance_cols[0]

        # Significance tests between experiment types
        exp_types = df['experiment_type'].unique()

        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            for i, type1 in enumerate(exp_types):
                for type2 in exp_types[i+1:]:
                    type1_data = dataset_df[dataset_df['experiment_type'] == type1][primary_metric].dropna()
                    type2_data = dataset_df[dataset_df['experiment_type'] == type2][primary_metric].dropna()

                    if len(type1_data) > 1 and len(type2_data) > 1:
                        try:
                            # Mann-Whitney U test
                            statistic, p_value = mannwhitneyu(type1_data, type2_data, alternative='two-sided')

                            rows.append({
                                'analysis_category': 'significance_tests',
                                'analysis_type': 'mann_whitney_u',
                                'dataset': dataset,
                                'method': 'ALL',
                                'experiment_type': f'{type1}_vs_{type2}',
                                'metric_name': f'mann_whitney_statistic_{primary_metric}',
                                'count': len(type1_data) + len(type2_data),
                                'mean': float(statistic),
                                'std': 0,
                                'min': float(statistic),
                                'max': float(statistic),
                                'median': float(statistic)
                            })

                            rows.append({
                                'analysis_category': 'significance_tests',
                                'analysis_type': 'mann_whitney_u',
                                'dataset': dataset,
                                'method': 'ALL',
                                'experiment_type': f'{type1}_vs_{type2}',
                                'metric_name': f'mann_whitney_pvalue_{primary_metric}',
                                'count': len(type1_data) + len(type2_data),
                                'mean': float(p_value),
                                'std': 0,
                                'min': float(p_value),
                                'max': float(p_value),
                                'median': float(p_value)
                            })

                            rows.append({
                                'analysis_category': 'significance_tests',
                                'analysis_type': 'mann_whitney_u',
                                'dataset': dataset,
                                'method': 'ALL',
                                'experiment_type': f'{type1}_vs_{type2}',
                                'metric_name': f'is_significant_{primary_metric}',
                                'count': len(type1_data) + len(type2_data),
                                'mean': 1.0 if p_value < 0.05 else 0.0,
                                'std': 0,
                                'min': 1.0 if p_value < 0.05 else 0.0,
                                'max': 1.0 if p_value < 0.05 else 0.0,
                                'median': 1.0 if p_value < 0.05 else 0.0
                            })

                        except Exception as e:
                            logger.warning(f"Statistical test failed for {type1} vs {type2} on {dataset}: {e}")

        return rows

    def _get_correlation_analysis(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Correlation analysis between metrics."""

        rows = []

        # Find numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Filter out ID columns and other non-metric columns
        metric_cols = [col for col in numeric_cols if not any(exclude in col.lower()
                      for exclude in ['id', 'seed', 'gpu', 'time', 'epoch'])]

        if len(metric_cols) < 2:
            return rows

        # Compute correlation matrix
        try:
            corr_matrix = df[metric_cols].corr()

            # Extract correlations
            for i, col1 in enumerate(metric_cols):
                for col2 in metric_cols[i+1:]:
                    if not (pd.isna(corr_matrix.loc[col1, col2])):
                        correlation = corr_matrix.loc[col1, col2]

                        rows.append({
                            'analysis_category': 'correlation_analysis',
                            'analysis_type': 'pearson_correlation',
                            'dataset': 'ALL',
                            'method': 'ALL',
                            'experiment_type': 'ALL',
                            'metric_name': f'{col1}_vs_{col2}',
                            'count': len(df[[col1, col2]].dropna()),
                            'mean': correlation,
                            'std': 0,
                            'min': correlation,
                            'max': correlation,
                            'median': correlation
                        })

        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")

        return rows



def worker_process_main(gpu_id: int, config: Dict[str, Any],
                       experiments: List[ExperimentSpec]) -> List[ExperimentResult]:
    """Main function for worker process - must be at module level for pickling"""
    try:
        # Initialize CUDA context in worker process
        if gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            # Initialize CUDA context
            dummy = torch.cuda.FloatTensor([0])
            del dummy
        else:
            device = torch.device('cpu')

        # Set random seeds for this worker
        set_random_seeds(42 + gpu_id)

        # Initialize worker with fixed constructor
        worker = GPUWorker(0 if gpu_id >= 0 else -1, config)
        return worker.run_worker(experiments)
    except Exception as e:
        print(f"Worker process {gpu_id} failed: {e}")
        traceback.print_exc()
        return []


#!/usr/bin/env python3
"""
Universal K Matrix - Main Experiment Runner
Complete implementation for publication-ready results
"""



def main():
    """Main function to run comprehensive Universal K Matrix experiments."""
    logger = setup_logging(CONFIG['output_dir'], CONFIG['log_level'])

    print("="*80)
    print("UNIVERSAL K MATRIX - COMPREHENSIVE EXPERIMENTAL SUITE")
    print("Publication-Ready Results with Statistical Analysis")
    print("="*80)

    # Setup
    experiment_id = get_experiment_id()

    print("="*60)
    print("STARTING COMPREHENSIVE UNIVERSAL K MATRIX EXPERIMENTS")
    print(f"Experiment ID: {experiment_id}")
    print("="*60)

    # Display configuration
    print("EXPERIMENT CONFIGURATION:")
    print(f"  - Output Directory: {CONFIG['output_dir']}")
    print(f"  - Use All GPUs: {CONFIG['use_all_gpus']}")
    print(f"  - K Methods: {CONFIG['k_methods']}")
    print(f"  - Baseline Methods: {CONFIG['baseline_methods']}")
    print(f"  - Factors to Try: {CONFIG['factors_to_try']}")
    print(f"  - Latent Dimensions: {CONFIG['latent_dimensions']}")
    print(f"  - Random Seeds: {CONFIG['random_seeds']}")
    print(f"  - Cross-Validation Folds: {CONFIG['cross_validation_folds']}")
    print(f"  - Training Epochs: {CONFIG['training_config']['epochs']}")

    # Check GPU availability
    device_manager = DeviceManager(CONFIG['use_all_gpus'])
    device_info = device_manager.get_device_info()
    print("\nHARDWARE CONFIGURATION:")
    for key, value in device_info.items():
        print(f"  - {key}: {value}")

    # Discover datasets
    print("\nDISCOVERING DATASETS:")
    dataset_manager = DatasetManager(config=CONFIG)
    available_datasets = dataset_manager.get_available_datasets()

    if not available_datasets:
        print("ERROR: No datasets found!")
        print("Please ensure your .npy files are in the current directory.")
        print("Expected format: dataset_name_x_train.npy and dataset_name_y_train.npy")
        sys.exit(1)

    print(f"Found {len(available_datasets)} datasets:")
    for dataset_name in available_datasets:
        try:
            _, _, is_classification, metadata = dataset_manager.load_dataset(dataset_name)
            task_type = "Classification" if is_classification else "Regression"
            print(f"  - {dataset_name}: {metadata['processed_shape']} features, {task_type}")
        except Exception as e:
            print(f"  - {dataset_name}: Error loading - {e}")

    # Estimate experiment count
    n_datasets = len(available_datasets)
    n_k_methods = len(CONFIG['k_methods'])
    n_baseline_methods = len(CONFIG['baseline_methods'])
    n_factors = len(CONFIG['factors_to_try'])
    n_latent_dims = len(CONFIG['latent_dimensions'])
    n_seeds = len(CONFIG['random_seeds'])

    # Calculate experiment counts
    universal_k_experiments = n_datasets * n_k_methods * n_factors * n_latent_dims * n_seeds
    sota_experiments = n_datasets * n_baseline_methods * n_latent_dims * n_seeds * 3  # avg 3 hyperparams per method
    enhanced_experiments = n_datasets * n_baseline_methods * n_k_methods * n_factors * n_latent_dims * n_seeds * 2  # avg 2 hyperparams

    total_experiments = universal_k_experiments + sota_experiments + enhanced_experiments

    print("\nEXPERIMENT ESTIMATION:")
    print(f"  - Universal K Matrix experiments: {universal_k_experiments}")
    print(f"  - SOTA Baseline experiments: {sota_experiments}")
    print(f"  - Enhanced SOTA experiments: {enhanced_experiments}")
    print(f"  - TOTAL EXPERIMENTS: {total_experiments}")

    # Estimate time
    avg_time_per_experiment = 120  # seconds (conservative estimate)
    total_time_hours = (total_experiments * avg_time_per_experiment) / 3600
    parallel_time_hours = total_time_hours / len(device_manager.available_devices)

    print(f"  - Estimated total time (sequential): {total_time_hours:.1f} hours")
    print(f"  - Estimated time (parallel on {len(device_manager.available_devices)} devices): {parallel_time_hours:.1f} hours")

    # Confirm execution
    print(f"\nAbout to run {total_experiments} experiments across {n_datasets} datasets.")
    print(f"Estimated time: {parallel_time_hours:.1f} hours on {len(device_manager.available_devices)} devices.")


    start_time = time.time()

    try:
        # Run comprehensive experiments
        print("\nSTARTING PARALLEL EXPERIMENT EXECUTION")
        orchestrator = ParallelExperimentOrchestrator(CONFIG)
        results_df = orchestrator.run_comprehensive_experiments(available_datasets)

        experiment_time = time.time() - start_time
        print(f"\nEXPERIMENTS COMPLETED in {experiment_time/3600:.2f} hours")
        print(f"Generated {len(results_df)} experiment results")

        # Save raw results
        raw_results_path = os.path.join(CONFIG['output_dir'], f'raw_results_{experiment_id}.csv')
        results_df.to_csv(raw_results_path, index=False)
        print(f"Raw results saved to: {raw_results_path}")

        # Perform comprehensive statistical analysis
        print("\nSTARTING COMPREHENSIVE STATISTICAL ANALYSIS")
        analyzer = ComprehensiveStatisticalAnalyzer(CONFIG)
        analysis_path = analyzer.analyze_and_export(results_df, CONFIG['output_dir'])

        analysis_time = time.time() - start_time - experiment_time
        print(f"Statistical analysis completed in {analysis_time:.2f} seconds")
        print(f"Comprehensive analysis saved to: {analysis_path}")

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total execution time: {total_time/3600:.2f} hours")
        print(f"Experiments run: {len(results_df)}")
        print(f"Success rate: {len(results_df[results_df['success'] == True]) / len(results_df) * 100:.1f}%" if 'success' in results_df.columns else "N/A")
        print(f"Raw results: {raw_results_path}")
        print(f"Statistical analysis: {analysis_path}")
        print("\nFiles are ready for publication analysis!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        print("Partial results may be available in the output directory")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR during experiment execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # This should be at the very beginning of main()

    # Rest of main() function...
    main()