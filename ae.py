import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="networkx backend defined more than once: nx-loopback")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch\\.cuda\\.amp\\..*")

import os
import torch
import numpy as np

# ───────────────────────────────────────────────────────────────
# Grab each worker's rank from the env, not from argparse:
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
print(f"[local_rank={local_rank}] using device {device}", flush=True)
# ───────────────────────────────────────────────────────────────

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error
)
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal, ttest_ind, mannwhitneyu, wilcoxon, friedmanchisquare, f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import defaultdict
import sys
import time
import os
import json
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

num_gpus = torch.cuda.device_count()
device_ids = list(range(num_gpus))

# How many Python workers to use (e.g. up to 16 CPUs)
NUM_WORKERS     = min(16, os.cpu_count())
# How many batches each worker can prefetch
PREFETCH_FACTOR = 2

# --- Load Pre-saved Datasets ---
def load_fashion_mnist_from_files():
    """Load Fashion-MNIST from .npy files and move to GPU."""
    x_train = torch.tensor(np.load('fashion_mnist_x_train.npy'), dtype=torch.float32)
    y_train = torch.tensor(np.load('fashion_mnist_y_train.npy'), dtype=torch.long)
    x_test = torch.tensor(np.load('fashion_mnist_x_test.npy'), dtype=torch.float32)
    y_test = torch.tensor(np.load('fashion_mnist_y_test.npy'), dtype=torch.long)
    return (x_train, y_train), (x_test, y_test)

def load_diabetes_from_files():
    """Load UCI Diabetes from .npy files and move to GPU."""
    x_train = torch.tensor(np.load('diabetes_x_train.npy'), dtype=torch.float32)
    y_train = torch.tensor(np.load('diabetes_y_train.npy'), dtype=torch.long)
    x_test = torch.tensor(np.load('diabetes_x_test.npy'), dtype=torch.float32)
    y_test = torch.tensor(np.load('diabetes_y_test.npy'), dtype=torch.long)
    return (x_train, y_train), (x_test, y_test)

# --- Feature Grouping and Shared Structure Estimation (GPU) ---
def group_features(x_train, num_groups):
    if num_groups == 1:
        return [list(range(x_train.shape[1]))]
    # Compute correlation matrix on GPU
    x_train = x_train.float()
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True)
    x_norm = (x_train - x_mean) / (x_std + 1e-6)
    corr_matrix = torch.mm(x_norm.T, x_norm) / x_norm.shape[0]
    distances = 1 - torch.abs(corr_matrix)

    # Perform clustering on CPU (AgglomerativeClustering doesn't support GPU natively)
    distances_np = distances.cpu().numpy()
    clustering = AgglomerativeClustering(n_clusters=num_groups, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distances_np)
    groups = [[] for _ in range(num_groups)]
    for feat_idx, label in enumerate(labels):
        groups[label].append(feat_idx)
    return groups

def estimate_shared_structure(x_train, feature_groups, method='mean'):
    shared_structures = []
    for group in feature_groups:
        group_data = x_train[:, group]
        if method == 'mean':
            shared = group_data.mean(dim=1, keepdim=True)
            shared = shared.expand(-1, len(group))
        elif method == 'pca':
            # Simplified PCA on GPU
            group_data_centered = group_data - group_data.mean(dim=0, keepdim=True)
            cov = torch.mm(group_data_centered.T, group_data_centered) / (group_data_centered.shape[0] - 1)
            _, _, V = torch.svd(cov)
            principal_component = V[:, 0:1]
            shared = torch.mm(group_data_centered, principal_component)
            shared = torch.mm(shared, principal_component.T) + group_data.mean(dim=0, keepdim=True)
        elif method == 'nmf':
            # Approximate NMF on GPU (basic iterative approach)
            group_data_shifted = group_data + torch.abs(group_data.min()) + 1e-6
            W = torch.rand(group_data_shifted.shape[0], 1, device=group_data.device)
            H = torch.rand(1, group_data_shifted.shape[1], device=group_data.device)
            for _ in range(10):  # Simple iteration
                W = W * (torch.mm(group_data_shifted, H.T) / (torch.mm(torch.mm(W, H), H.T) + 1e-6))
                H = H * (torch.mm(W.T, group_data_shifted) / (torch.mm(W.T, torch.mm(W, H)) + 1e-6))
            shared = torch.mm(W, H)
            shared = shared - torch.abs(group_data.min()) - 1e-6
        shared_structures.append(shared)
    return shared_structures

# --- Mutual Information Estimators (GPU) ---
def ksg_mutual_information(z, num_factors, latent_dim_per_factor, k=3):
    start_time = time.time()
    z_reshaped = z.view(-1, num_factors, latent_dim_per_factor)
    subsample_size = 300
    indices = torch.randperm(len(z_reshaped), device=z.device)[:subsample_size]
    z_reshaped = z_reshaped[indices]
    
    # Check variance of z
    z_var = torch.var(z_reshaped, dim=0)
    if torch.any(z_var < 1e-6):
        if dist.get_rank() == 0:
            print(f"[Rank 0] Warning: Low variance in latent space: {z_var.cpu().numpy()}")
        return 0.0
    
    mi_scores = []
    for j in range(num_factors):
        for k_idx in range(j + 1, num_factors):
            x = z_reshaped[:, j, :]
            y = z_reshaped[:, k_idx, :]
            xy = torch.cat([x, y], dim=1)
            dist_xy = torch.cdist(xy, xy)
            dist_x = torch.cdist(x, x)
            dist_y = torch.cdist(y, y)
            _, indices_xy = torch.topk(dist_xy, k + 1, largest=False)
            _, indices_x = torch.topk(dist_x, k + 1, largest=False)
            _, indices_y = torch.topk(dist_y, k + 1, largest=False)
            d_xy = dist_xy[torch.arange(dist_xy.shape[0], device=z.device), indices_xy[:, k]]
            d_xy = torch.clamp(d_xy, min=1e-6)  # Prevent zero distances
            d_x = dist_x[torch.arange(dist_x.shape[0], device=z.device), indices_x[:, k]]
            d_y = dist_y[torch.arange(dist_y.shape[0], device=z.device), indices_y[:, k]]
            psi_k = torch.digamma(torch.tensor(k, device=z.device)).item()
            psi_N = torch.digamma(torch.tensor(len(x), device=z.device)).item()
            psi_x = torch.digamma((dist_x < d_xy.unsqueeze(1)).sum(dim=1).float() + 1e-6).mean().item()
            psi_y = torch.digamma((dist_y < d_xy.unsqueeze(1)).sum(dim=1).float() + 1e-6).mean().item()
            mi = psi_k + psi_N - psi_x - psi_y
            mi_scores.append(max(0, mi))
    mi_value = np.mean(mi_scores) if mi_scores else 0.0
    return mi_value

class MINECritic(nn.Module):
    def __init__(self, input_dim):
        super(MINECritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))

def mine_mutual_information(z, num_factors, latent_dim_per_factor, epochs=10, early_stopping=True, delta=0.001, patience=5):
    start_time = time.time()
    z = z.detach().to(z.device)
    z_reshaped = z.view(-1, num_factors, latent_dim_per_factor)
    subsample_size = 300
    indices = torch.randperm(len(z_reshaped), device=z.device)[:subsample_size]
    z_reshaped = z_reshaped[indices]
    
    # Single critic for all pairs
    critic = MINECritic(latent_dim_per_factor).to(z.device)
    optimizer = optim.Adam(critic.parameters(), lr=0.001)
    scaler = GradScaler()
    
    mi_scores = []
    for j in range(num_factors):
        for k in range(j + 1, num_factors):
            x = z_reshaped[:, j, :].detach().clone().requires_grad_(True)
            y = z_reshaped[:, k, :].detach().clone().requires_grad_(True)

            mi_history = []
            patience_counter = 0
            best_mi = -float('inf')

            for epoch in range(epochs):
                optimizer.zero_grad()
                with autocast():
                    y_shuffled = y[torch.randperm(y.shape[0], device=y.device)]
                    t_joint = critic(x, y)
                    t_marginal = critic(x, y_shuffled)
                    mi = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
                    loss = -mi
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                mi_value = mi.detach().item()
                mi_history.append(mi_value)

                if early_stopping:
                    if mi_value > best_mi:
                        best_mi = mi_value
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if len(mi_history) >= patience + 1:
                        recent_change = abs(mi_history[-1] - mi_history[-(patience + 1)])
                        if recent_change < delta and patience_counter >= patience:
                            break
                if epoch > epochs * 2:
                    break

            mi_scores.append(max(0, best_mi))

    critic = None  # Free memory
    return np.mean(mi_scores) if mi_scores else 0.0

def edge_based_mutual_information(z, num_factors, latent_dim_per_factor):
    start_time = time.time()
    z_reshaped = z.view(-1, num_factors, latent_dim_per_factor)
    subsample_size = 300
    indices = torch.randperm(len(z_reshaped), device=z.device)[:subsample_size]
    z_reshaped = z_reshaped[indices]
    mi_scores = []
    for j in range(num_factors):
        for k in range(j + 1, num_factors):
            x = z_reshaped[:, j, :]
            y = z_reshaped[:, k, :]
            batch_size = x.shape[0]
            logits = torch.matmul(x, y.T)
            labels = torch.arange(batch_size, device=x.device)
            loss = F.cross_entropy(logits, labels)
            mi_lower_bound = -loss.item() + np.log(batch_size)
            mi_scores.append(max(0, mi_lower_bound))
    mi_value = np.mean(mi_scores) if mi_scores else 0.0
    return mi_value

def gaussian_mutual_information(z, num_factors, latent_dim_per_factor):
    start_time = time.time()
    z_reshaped = z.view(-1, num_factors, latent_dim_per_factor)
    subsample_size = 300
    indices = torch.randperm(len(z_reshaped), device=z.device)[:subsample_size]
    z_reshaped = z_reshaped[indices]
    mi_scores = []
    for j in range(num_factors):
        for k in range(j + 1, num_factors):
            x = z_reshaped[:, j, :]
            y = z_reshaped[:, k, :]
            xy = torch.cat([x, y], dim=1)
            cov_xy = torch.cov(xy.T)
            cov_x = torch.cov(x.T)
            cov_y = torch.cov(y.T)
            det_xy = torch.det(cov_xy + 1e-6 * torch.eye(cov_xy.shape[0], device=z.device))
            det_x = torch.det(cov_x + 1e-6 * torch.eye(cov_x.shape[0], device=z.device))
            det_y = torch.det(cov_y + 1e-6 * torch.eye(cov_y.shape[0], device=z.device))
            mi = 0.5 * torch.log(det_x * det_y / det_xy).item()
            mi_scores.append(max(0, mi))
    mi_value = np.mean(mi_scores) if mi_scores else 0.0
    return mi_value

class VariationalBoundMI(nn.Module):
    def __init__(self, input_dim):
        super(VariationalBoundMI, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, y):
        return self.q_net(x)

def variational_bound_mutual_information(z, num_factors, latent_dim_per_factor, epochs=10, early_stopping=True, delta=0.001, patience=5):
    start_time = time.time()
    z = z.detach().to(z.device)
    z_reshaped = z.view(-1, num_factors, latent_dim_per_factor)
    subsample_size = 300
    indices = torch.randperm(len(z_reshaped), device=z.device)[:subsample_size]
    z_reshaped = z_reshaped[indices]
    
    # Single q_model for all pairs
    q_model = VariationalBoundMI(latent_dim_per_factor).to(z.device)
    optimizer = optim.Adam(q_model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    mi_scores = []
    for j in range(num_factors):
        for k in range(j + 1, num_factors):
            x = z_reshaped[:, j, :].detach().clone().requires_grad_(True)
            y = z_reshaped[:, k, :].detach().clone().requires_grad_(True)

            mi_history = []
            patience_counter = 0
            best_mi = -float('inf')

            for epoch in range(epochs):
                optimizer.zero_grad()
                with autocast():
                    q_y_given_x = q_model(x, y)
                    log_q = -F.mse_loss(q_y_given_x, y, reduction='mean')
                    y_shuffled = y[torch.randperm(y.shape[0], device=y.device)]
                    q_y_marginal = q_model(x, y_shuffled)
                    log_p = -F.mse_loss(q_y_marginal, y_shuffled, reduction='mean')
                    mi_lower_bound = log_q - log_p
                    loss = -mi_lower_bound
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                mi_value = mi_lower_bound.detach().item()
                mi_history.append(mi_value)

                if early_stopping:
                    if mi_value > best_mi:
                        best_mi = mi_value
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if len(mi_history) >= patience + 1:
                        recent_change = abs(mi_history[-1] - mi_history[-(patience + 1)])
                        if recent_change < delta and patience_counter >= patience:
                            break
                if epoch > epochs * 2:
                    break

            mi_scores.append(max(0, best_mi))

    q_model = None  # Free memory
    return np.mean(mi_scores) if mi_scores else 0.0

# --- Universal Matrix Autoencoder with Alternative Decoders ---
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=64, nhead=nhead, dim_feedforward=128),
            num_layers=1
        )
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        memory = torch.zeros(batch_size, 1, 64).to(x.device)
        x = self.transformer(x, memory)
        x = x.squeeze(1)
        return self.out(x)

class UniversalMatrixAutoencoder(nn.Module):
    def __init__(self, input_dim, num_factors, latent_dim_per_factor, feature_groups, shared_structures, beta, lambda_recon, decoder_type='k_matrix'):
        super(UniversalMatrixAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.latent_dim_per_factor = latent_dim_per_factor
        self.feature_groups = feature_groups
        self.shared_structures = shared_structures
        self.beta = beta
        self.lambda_recon = lambda_recon
        self.decoder_type = decoder_type

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_factors * latent_dim_per_factor)
        )

        if decoder_type == 'k_matrix':
            self.decoders = [nn.Parameter(torch.randn(len(group), latent_dim_per_factor, device=device)) for group in feature_groups]
            for idx, param in enumerate(self.decoders):
                self.register_parameter(f'decoder_{idx}', param)
        elif decoder_type == 'linear':
            self.decoders = nn.ModuleList([
                nn.Linear(latent_dim_per_factor, len(group)) for group in feature_groups
            ])
        elif decoder_type == 'transformer':
            self.decoders = nn.ModuleList([
                TransformerDecoder(latent_dim_per_factor, len(group)) for group in feature_groups
            ])

    def forward(self, x):
        z = self.encoder(x)
        z_reshaped = z.view(-1, self.num_factors, self.latent_dim_per_factor)

        reconstructions = torch.zeros_like(x, device=x.device)
        for group_idx, group in enumerate(self.feature_groups):
            decoder = self.decoders[group_idx]
            for j in range(self.num_factors):
                i_j = z_reshaped[:, j, :]
                if self.decoder_type == 'k_matrix':
                    recon = i_j @ decoder.T
                else:
                    recon = decoder(i_j)
                reconstructions[:, group] += recon

        return z, reconstructions

    def compute_loss(self, x, z, reconstructions):
        kl_loss = torch.mean(torch.sum(z ** 2, dim=1))
        batch_size = x.shape[0]
        target = torch.zeros_like(x, device=x.device)
        for group_idx, group in enumerate(self.feature_groups):
            shared = self.shared_structures[group_idx][:batch_size].to(x.device)
            target[:, group] = shared
        recon_loss = F.mse_loss(reconstructions, target, reduction='mean')
        total_loss = self.beta * kl_loss + self.lambda_recon * recon_loss
        return total_loss, kl_loss, recon_loss

# --- Teacher and Student Models ---
class TeacherModel(nn.Module):
    def __init__(self, latent_dim, output_dim, task='classification'):
        super(TeacherModel, self).__init__()
        self.task = task
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1) if task == 'classification' else nn.Identity()
        )

    def forward(self, z):
        return self.decoder(z)

class StudentModel(nn.Module):
    def __init__(self, latent_dim, output_dim, task='classification'):
        super(StudentModel, self).__init__()
        self.task = task
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1) if task == 'classification' else nn.Identity()
        )

    def forward(self, z):
        return self.decoder(z)

# --- Enhanced Evaluation Metrics ---
def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    try:
        # For binary classification
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Class-wise metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['class_report'] = class_report
        
        # ROC AUC if probabilities are provided
        if y_prob is not None:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except:
                    metrics['roc_auc'] = None
    except Exception as e:
        print(f"Error in classification metrics: {e}")
    
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    metrics = {}
    
    try:
        # Error metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # Try to calculate MAPE if no zeros in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        
        # Goodness of fit metrics
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Residuals analysis
        residuals = y_true - y_pred
        metrics['residuals_mean'] = np.mean(residuals)
        metrics['residuals_std'] = np.std(residuals)
    except Exception as e:
        print(f"Error in regression metrics: {e}")
    
    return metrics

# --- Statistical Tests ---
def perform_statistical_tests(results_df, metric_col, group_col, alpha=0.05):
    """Perform appropriate statistical tests to compare groups."""
    unique_groups = results_df[group_col].unique()
    n_groups = len(unique_groups)
    test_results = {}
    
    if n_groups < 2:
        return {"error": "Need at least 2 groups for comparison"}
    
    # Extract data by group
    group_data = [results_df[results_df[group_col] == g][metric_col].values for g in unique_groups]
    group_means = {g: results_df[results_df[group_col] == g][metric_col].mean() for g in unique_groups}
    test_results['group_means'] = group_means
    
    # Check sample sizes
    sample_sizes = [len(data) for data in group_data]
    min_samples = min(sample_sizes)
    test_results['sample_sizes'] = dict(zip(unique_groups, sample_sizes))
    
    # For only two groups
    if n_groups == 2:
        # t-test
        try:
            t_stat, p_value = ttest_ind(group_data[0], group_data[1], equal_var=False)
            test_results['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        except Exception as e:
            test_results['t_test_error'] = str(e)
        
        # Mann-Whitney U test (non-parametric alternative)
        try:
            u_stat, p_value = mannwhitneyu(group_data[0], group_data[1])
            test_results['mann_whitney'] = {
                'u_statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        except Exception as e:
            test_results['mann_whitney_error'] = str(e)
            
    # For more than two groups
    if n_groups > 2:
        # One-way ANOVA
        try:
            f_stat, p_value = f_oneway(*group_data)
            test_results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
        except Exception as e:
            test_results['anova_error'] = str(e)
        
        # Friedman test (non-parametric alternative)
        if min_samples >= 2:
            try:
                # Prepare data for Friedman test
                # We need equal size groups, so we'll use the minimum
                truncated_data = [data[:min_samples] for data in group_data]
                friedman_data = np.array(truncated_data).T  # Transpose so rows are samples
                f_stat, p_value = friedmanchisquare(*truncated_data)
                test_results['friedman'] = {
                    'chi2_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha
                }
            except Exception as e:
                test_results['friedman_error'] = str(e)
        
        # If ANOVA is significant, run post-hoc Tukey HSD test
        if test_results.get('anova', {}).get('significant', False):
            try:
                # Prepare data for Tukey's test
                all_data = []
                all_groups = []
                for i, group in enumerate(unique_groups):
                    all_data.extend(group_data[i])
                    all_groups.extend([group] * len(group_data[i]))
                
                # Run Tukey's test
                tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=alpha)
                tukey_results = {
                    'group1': [],
                    'group2': [],
                    'mean_diff': [],
                    'p_value': [],
                    'significant': [],
                    'lower_ci': [],
                    'upper_ci': []
                }
                
                for i in range(len(tukey.groupsunique)):
                    for j in range(i+1, len(tukey.groupsunique)):
                        idx = tukey.pairindices.index((i, j))
                        tukey_results['group1'].append(tukey.groupsunique[i])
                        tukey_results['group2'].append(tukey.groupsunique[j])
                        tukey_results['mean_diff'].append(tukey.meandiffs[idx])
                        tukey_results['p_value'].append(tukey.pvalues[idx])
                        tukey_results['significant'].append(tukey.pvalues[idx] < alpha)
                        tukey_results['lower_ci'].append(tukey.confint[idx][0])
                        tukey_results['upper_ci'].append(tukey.confint[idx][1])
                
                test_results['tukey_hsd'] = tukey_results
            except Exception as e:
                test_results['tukey_hsd_error'] = str(e)
    
    return test_results

# --- Cross Validation with Statistical Comparison ---
def k_fold_cross_validation(models_config, x, y, n_splits=5, task='classification'):
    """
    Perform k-fold cross validation for multiple model configurations and analyze results statistically.
    
    Args:
        models_config: List of model configuration dictionaries
        x: Input data
        y: Target data
        n_splits: Number of folds for cross-validation
        task: 'classification' or 'regression'
        
    Returns:
        DataFrame with results and statistical test outcomes
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_train_fold, x_val_fold = x[train_idx], x[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Convert to PyTorch tensors
        x_train_tensor = torch.tensor(x_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)
        
        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        val_ds = TensorDataset(x_val_tensor, y_val_tensor)
        
        # Create DataLoaders with DDP for training
        sampler = DistributedSampler(train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=1024, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        for config_idx, config in enumerate(models_config):
            # Extract configuration
            num_k = config['num_k']
            shared_method = config['shared_method']
            decoder_type = config['decoder_type']
            beta = config['beta']
            lambda_recon = config['lambda_recon']
            alpha = config['alpha']
            latent_dim_per_factor = config['latent_dim_per_factor']
            
            # Create feature groups and shared structures
            feature_groups = group_features(x_train_tensor, num_k)
            shared_structs = estimate_shared_structure(x_train_tensor, feature_groups, method=shared_method)
            
            # Initialize and wrap models
            input_dim = x.shape[1]
            output_dim = len(np.unique(y)) if task == 'classification' else 1
            
            ae = UniversalMatrixAutoencoder(
                input_dim,
                num_k,
                latent_dim_per_factor,
                feature_groups,
                shared_structs,
                beta=beta,
                lambda_recon=lambda_recon,
                decoder_type=decoder_type
            ).to(device)
            ae = DDP(ae, device_ids=[local_rank])
            
            latent_size = num_k * latent_dim_per_factor
            
            teacher = TeacherModel(latent_size, output_dim, task).to(device)
            teacher = DDP(teacher, device_ids=[local_rank])
            
            student = StudentModel(latent_size, output_dim, task).to(device)
            student = DDP(student, device_ids=[local_rank])
            
            # Train autoencoder
            ae_val_loss = train_autoencoder_ddp(ae, train_loader, val_loader, device, epochs=20)
            
            # Train teacher
            teacher_val_metric = train_teacher_ddp(teacher, ae, train_loader, val_loader, device, epochs=15)
            
            # Train student
            student_val_metric = train_student_ddp(student, teacher, ae, train_loader, val_loader, device, alpha=alpha, epochs=15)
            
            # Evaluate models on validation set (with detailed metrics)
            if dist.get_rank() == 0:  # Only the master process computes and stores metrics
                ae.eval()
                teacher.eval()
                student.eval()
                
                with torch.no_grad():
                    # Generate latent representations and predictions for the entire validation set
                    all_latents = []
                    all_teacher_preds = []
                    all_student_preds = []
                    
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(device)
                        z, _ = ae(batch_x)
                        all_latents.append(z.cpu())
                        
                        teacher_pred = teacher(z)
                        student_pred = student(z)
                        
                        all_teacher_preds.append(teacher_pred.cpu())
                        all_student_preds.append(student_pred.cpu())
                    
                    all_latents = torch.cat(all_latents, dim=0)
                    all_teacher_preds = torch.cat(all_teacher_preds, dim=0)
                    all_student_preds = torch.cat(all_student_preds, dim=0)
                    
                    # Calculate detailed metrics
                    if task == 'classification':
                        teacher_preds_hard = torch.argmax(all_teacher_preds, dim=1).numpy()
                        student_preds_hard = torch.argmax(all_student_preds, dim=1).numpy()
                        
                        teacher_metrics = calculate_classification_metrics(
                            y_val_fold, teacher_preds_hard, all_teacher_preds.numpy()
                        )
                        student_metrics = calculate_classification_metrics(
                            y_val_fold, student_preds_hard, all_student_preds.numpy()
                        )
                    else:  # regression
                        teacher_metrics = calculate_regression_metrics(
                            y_val_fold, all_teacher_preds.numpy()
                        )
                        student_metrics = calculate_regression_metrics(
                            y_val_fold, all_student_preds.numpy()
                        )
                    
                    # Calculate MI scores on latent representations
                    mi_ksg = ksg_mutual_information(all_latents, num_k, latent_dim_per_factor)
                    mi_edge = edge_based_mutual_information(all_latents, num_k, latent_dim_per_factor)
                    mi_gaussian = gaussian_mutual_information(all_latents, num_k, latent_dim_per_factor)
                    # mi_variational = variational_bound_mutual_information(all_latents, num_k, latent_dim_per_factor, epochs=variational_epochs)

                    # Store results
                    result_entry = {
                        'fold': fold_idx,
                        'config_idx': config_idx,
                        'num_k': num_k,
                        'shared_method': shared_method,
                        'decoder_type': decoder_type,
                        'beta': beta,
                        'lambda_recon': lambda_recon,
                        'alpha': alpha,
                        'ae_val_loss': ae_val_loss,
                        'teacher_val_metric': teacher_val_metric,
                        'student_val_metric': student_val_metric,
                        'mi_ksg': mi_ksg,
                        'mi_edge': mi_edge,
                        'mi_gaussian': mi_gaussian,
                        # 'mi_variational': mi_variational
                    }
                    
                    # Add detailed metrics
                    if task == 'classification':
                        metrics_to_add = {
                            'teacher_accuracy': teacher_metrics['accuracy'],
                            'teacher_f1': teacher_metrics['f1'],
                            'teacher_precision': teacher_metrics['precision'],
                            'teacher_recall': teacher_metrics['recall'],
                            'student_accuracy': student_metrics['accuracy'],
                            'student_f1': student_metrics['f1'],
                            'student_precision': student_metrics['precision'],
                            'student_recall': student_metrics['recall']
                        }
                        if 'roc_auc' in teacher_metrics:
                            metrics_to_add['teacher_roc_auc'] = teacher_metrics['roc_auc']
                            metrics_to_add['student_roc_auc'] = student_metrics['roc_auc']
                    else:  # regression
                        metrics_to_add = {
                            'teacher_mse': teacher_metrics['mse'],
                            'teacher_rmse': teacher_metrics['rmse'],
                            'teacher_mae': teacher_metrics['mae'],
                            'teacher_r2': teacher_metrics['r2'],
                            'student_mse': student_metrics['mse'],
                            'student_rmse': student_metrics['rmse'],
                            'student_mae': student_metrics['mae'],
                            'student_r2': student_metrics['r2']
                        }
                    
                    result_entry.update(metrics_to_add)
                    results.append(result_entry)
    
    # Compile results and perform statistical tests
    if dist.get_rank() == 0:  # Only the master process compiles results
        results_df = pd.DataFrame(results)
        
        # Group results by configuration for statistical analysis
        statistical_results = {}
        
        # Define metrics to test based on task
        if task == 'classification':
            metrics_to_test = ['student_accuracy', 'student_f1', 'student_precision', 'student_recall', 'mi_ksg', 'mi_edge']
            if 'student_roc_auc' in results_df.columns:
                metrics_to_test.append('student_roc_auc')
        else:  # regression
            metrics_to_test = ['student_mse', 'student_rmse', 'student_mae', 'student_r2', 'mi_ksg', 'mi_edge']
        
        # Perform statistical tests for each metric
        for metric in metrics_to_test:
            try:
                # Compare across decoder types
                decoder_tests = perform_statistical_tests(results_df, metric, 'decoder_type')
                statistical_results[f'{metric}_by_decoder'] = decoder_tests
                
                # Compare across num_k values
                k_tests = perform_statistical_tests(results_df, metric, 'num_k')
                statistical_results[f'{metric}_by_k'] = k_tests
                
                # Compare across shared methods
                shared_tests = perform_statistical_tests(results_df, metric, 'shared_method')
                statistical_results[f'{metric}_by_shared_method'] = shared_tests
            except Exception as e:
                print(f"Error in statistical test for {metric}: {e}")
        
        return results_df, statistical_results
    
    return None, None

# --- Training Functions with Enhanced Evaluation and Reporting ---
def train_autoencoder_ddp(autoencoder, train_loader, val_loader, device,
                          epochs=50, mine_epochs=10, variational_epochs=10,
                          early_stopping=True, delta=0.001, patience=5):
    """Train with DDP-wrapped autoencoder; uses DistributedSampler."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    training_stats = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'kl_loss': [],
        'recon_loss': [],
        'lr': []
    }

    for epoch in range(epochs):
        # ===== Training =====
        autoencoder.train()
        train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                z, recon = autoencoder(batch_x)
                total_loss, kl_loss, recon_loss = autoencoder.module.compute_loss(batch_x, z, recon)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.step(scheduler)
            scaler.update()
            epoch_loss += total_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1

        # ===== Validation =====
        autoencoder.eval()
        val_loss = 0.0
        val_kl_loss = 0.0
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                z, recon = autoencoder(batch_x)
                tl, kl, rl = autoencoder.module.compute_loss(batch_x, z, recon)
                val_loss += tl.item()
                val_kl_loss += kl.item()
                val_recon_loss += rl.item()
        val_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)

        # synchronize best_val_loss across ranks
        tensor_val = torch.tensor(val_loss, device=device)
        dist.all_reduce(tensor_val, op=dist.ReduceOp.MIN)
        global_val_loss = tensor_val.item()
        
        # Collect training stats (on rank 0)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            training_stats['epoch'].append(epoch)
            training_stats['train_loss'].append(epoch_loss / num_batches)
            training_stats['val_loss'].append(global_val_loss)
            training_stats['kl_loss'].append(val_kl_loss)
            training_stats['recon_loss'].append(val_recon_loss)
            training_stats['lr'].append(current_lr)

        # only rank 0 updates early stopping
        if rank == 0:
            loss_history.append(global_val_loss)
            if global_val_loss < best_val_loss:
                best_val_loss = global_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if early_stopping and patience_counter > patience:
                break

    # Compute mutual information metrics
    mi_metrics = {'mi_ksg': 0.0, 'mi_edge': 0.0, 'mi_gaussian': 0.0, 'mi_variational': 0.0}
    try:
        with torch.no_grad():
            # Get a batch of latent representations
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                z, _ = autoencoder(batch_x)
                break  # Just need one batch
            
            num_factors = autoencoder.module.num_factors
            latent_dim_per_factor = autoencoder.module.latent_dim_per_factor

            # Compute MI on each rank
            mi_ksg = ksg_mutual_information(z, num_factors, latent_dim_per_factor)
            mi_edge = edge_based_mutual_information(z, num_factors, latent_dim_per_factor)
            mi_gaussian = gaussian_mutual_information(z, num_factors, latent_dim_per_factor)
            # mi_variational = variational_bound_mutual_information(z, num_factors, latent_dim_per_factor, epochs=variational_epochs)

            # Aggregate MI scores across ranks
            mi_ksg_tensor = torch.tensor(mi_ksg, device=device)
            mi_edge_tensor = torch.tensor(mi_edge, device=device)
            mi_gaussian_tensor = torch.tensor(mi_gaussian, device=device)
            # mi_variational_tensor = torch.tensor(mi_variational, device=device)

            dist.all_reduce(mi_ksg_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mi_edge_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(mi_gaussian_tensor, op=dist.ReduceOp.SUM)
            # dist.all_reduce(mi_variational_tensor, op=dist.ReduceOp.SUM)

            mi_metrics['mi_ksg'] = mi_ksg_tensor.item() / world_size
            mi_metrics['mi_edge'] = mi_edge_tensor.item() / world_size
            mi_metrics['mi_gaussian'] = mi_gaussian_tensor.item() / world_size
            # mi_metrics['mi_variational'] = mi_variational_tensor.item() / world_size

            if rank == 0:
                z_var = torch.var(z, dim=0)
                z_mean = torch.mean(z, dim=0)
                print(f"Latent Variance: {z_var.cpu().numpy()}")
                print(f"Latent Mean: {z_mean.cpu().numpy()}")
    except Exception as e:
        if rank == 0:
            print(f"Error computing MI metrics: {e}")
    
    # Log final training stats
    if rank == 0:
        print(f"Successfully computed MI metrics")
        print(f"Final validation loss: {best_val_loss:.6f}")
        if mi_metrics:
            print(f"MI metrics: KSG={mi_metrics['mi_ksg']:.4f}, Edge={mi_metrics['mi_edge']:.4f}, Gaussian={mi_metrics['mi_gaussian']:.4f}, Variational={mi_metrics['mi_variational']:.4f}")

    return best_val_loss

def evaluate_model(model, dataloader, task='classification', device=None):
    """
    Comprehensive evaluation of a model with detailed metrics
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        task: 'classification' or 'regression'
        device: Torch device
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            if task == 'classification':
                probs = outputs
                preds = torch.argmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
            else:
                preds = outputs
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    # Concatenate batch results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    if task == 'classification':
        all_probs = np.concatenate(all_probs)
        metrics = calculate_classification_metrics(all_labels, all_preds, all_probs)
    else:
        metrics = calculate_regression_metrics(all_labels, all_preds)
    
    return metrics

def train_teacher_ddp(teacher, autoencoder, train_loader, val_loader, device,
                      epochs=50, early_stopping=True, delta=0.001, patience=5):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    optimizer = optim.Adam(teacher.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss() if teacher.module.task=='classification' else nn.MSELoss()

    best_metric = float('-inf') if teacher.module.task=='classification' else float('inf')
    patience_counter = 0
    metric_history = []
    detailed_metrics = None  # Initialize to None
    
    training_stats = {
        'epoch': [],
        'train_loss': [],
        'val_metric': [],
        'lr': []
    }

    for epoch in range(epochs):
        # Training
        teacher.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).float() if teacher.module.task=='regression' else batch_y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z, _ = autoencoder(batch_x)
            with autocast():
                y_pred = teacher(z)
                loss = criterion(y_pred, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.step(scheduler)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1

        # Validation
        teacher.eval()
        metric = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True).float() if teacher.module.task=='regression' else batch_y.to(device)
                z, _ = autoencoder(batch_x)
                y_pred = teacher(z)
                
                if teacher.module.task=='classification':
                    batch_metric = (y_pred.argmax(dim=1)==batch_y).float().mean().item()
                    all_preds.append(y_pred.argmax(dim=1).cpu().numpy())
                else:
                    batch_metric = F.mse_loss(y_pred, batch_y).item()
                    all_preds.append(y_pred.cpu().numpy())
                
                metric += batch_metric
                all_labels.append(batch_y.cpu().numpy())
        
        metric /= len(val_loader)

        # synchronize across ranks (max for classification, min for regression)
        tensor_m = torch.tensor(metric, device=device)
        if teacher.module.task=='classification':
            dist.all_reduce(tensor_m, op=dist.ReduceOp.MAX)
            global_m = tensor_m.item()
        else:
            dist.all_reduce(tensor_m, op=dist.ReduceOp.MIN)
            global_m = tensor_m.item()
        
        # Collect training stats (on rank 0)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            training_stats['epoch'].append(epoch)
            training_stats['train_loss'].append(epoch_loss / num_batches)
            training_stats['val_metric'].append(global_m)
            training_stats['lr'].append(current_lr)

        if rank==0:
            scheduler.step()
            metric_history.append(global_m)
            if ((teacher.module.task=='classification' and global_m > best_metric) or
                (teacher.module.task!='classification' and global_m < best_metric)):
                best_metric = global_m
                patience_counter = 0
                
                # Compute detailed metrics at best performance
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                
                if teacher.module.task == 'classification':
                    detailed_metrics = calculate_classification_metrics(all_labels, all_preds)
                else:
                    detailed_metrics = calculate_regression_metrics(all_labels, all_preds)
            else:
                patience_counter += 1
            if early_stopping and patience_counter>patience:
                break
    
    # Log final performance metrics
    if rank == 0:
        metric_name = "accuracy" if teacher.module.task == 'classification' else "MSE"
        print(f"Teacher best {metric_name}: {best_metric:.6f}")
        if detailed_metrics is not None:
            if teacher.module.task == 'classification':
                print(f"Teacher detailed metrics: F1={detailed_metrics['f1']:.4f}, Precision={detailed_metrics['precision']:.4f}, Recall={detailed_metrics['recall']:.4f}")
            else:
                print(f"Teacher detailed metrics: RMSE={detailed_metrics['rmse']:.4f}, MAE={detailed_metrics['mae']:.4f}, R²={detailed_metrics['r2']:.4f}")

    return best_metric


def create_visualization(results_df, statistical_results, output_dir="results"):
    """Create comprehensive visualizations of model performance and statistical tests."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set theme for plots
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))
    
    # 1. Performance by decoder type
    plt.subplot(2, 2, 1)
    sns.boxplot(x='decoder_type', y='student_val_metric', data=results_df)
    plt.title('Student Performance by Decoder Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/decoder_performance.png")
    
    # 2. Performance by number of k (factors)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='num_k', y='student_val_metric', data=results_df)
    plt.title('Student Performance by Number of Factors')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/num_k_performance.png")
    
    # 3. Performance by shared method
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='shared_method', y='student_val_metric', data=results_df)
    plt.title('Student Performance by Shared Structure Method')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shared_method_performance.png")
    
    # 4. Relationship between MI and performance
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x='mi_ksg', y='student_val_metric', hue='decoder_type', data=results_df)
    plt.title('Student Performance vs KSG MI')
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x='mi_edge', y='student_val_metric', hue='decoder_type', data=results_df)
    plt.title('Student Performance vs Edge MI')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='mi_gaussian', y='student_val_metric', hue='decoder_type', data=results_df)
    plt.title('Student Performance vs Gaussian MI')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mi_vs_performance.png")
    
    # 5. Statistical test results visualization
    
    # Extract p-values from each test and create heatmap
    plt.figure(figsize=(15, 10))
    
    p_values = {}
    
    for key, result in statistical_results.items():
        if 'tukey_hsd' in result:
            metric_name = key.split('_by_')[0]
            group_by = key.split('_by_')[1]
            
            tukey_df = pd.DataFrame({
                'group1': result['tukey_hsd']['group1'],
                'group2': result['tukey_hsd']['group2'],
                'p_value': result['tukey_hsd']['p_value'],
                'significant': result['tukey_hsd']['significant']
            })
            
            # Create heatmap data
            groups = sorted(set(tukey_df['group1']).union(set(tukey_df['group2'])))
            heatmap_data = pd.DataFrame(1.0, index=groups, columns=groups)
            
            for _, row in tukey_df.iterrows():
                heatmap_data.loc[row['group1'], row['group2']] = row['p_value']
                heatmap_data.loc[row['group2'], row['group1']] = row['p_value']
            
            plt.figure(figsize=(8, 6))
            mask = np.triu(np.ones_like(heatmap_data, dtype=bool))
            ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.1, 
                             mask=mask, fmt='.3f')
            plt.title(f'{metric_name} p-values by {group_by}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{metric_name}_by_{group_by}_pvalues.png")
    
    # 6. Summary statistics by group
    summary_tables = {}
    
    # Create summary tables for student_val_metric by each grouping factor
    for group_factor in ['decoder_type', 'num_k', 'shared_method']:
        summary = results_df.groupby(group_factor)['student_val_metric'].agg(['mean', 'std', 'min', 'max', 'count'])
        summary_tables[group_factor] = summary
        
        # Save to CSV
        summary.to_csv(f"{output_dir}/summary_by_{group_factor}.csv")
    
    # 7. Export all results to CSV
    results_df.to_csv(f"{output_dir}/all_results.csv", index=False)
    
    # 8. Save statistical test results as JSON
    stat_results_serializable = {}
    for key, value in statistical_results.items():
        if isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    new_dict[k] = v.tolist()
                elif isinstance(v, np.float64):
                    new_dict[k] = float(v)
                else:
                    new_dict[k] = v
            stat_results_serializable[key] = new_dict
    
    with open(f"{output_dir}/statistical_results.json", 'w') as f:
        json.dump(stat_results_serializable, f, indent=4)
    
    return True

def train_student_ddp(student, teacher, autoencoder, train_loader, val_loader, device,
                      alpha=0.5, epochs=50, early_stopping=True, delta=0.001, patience=5):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()
    task_criterion = nn.CrossEntropyLoss() if student.module.task=='classification' else nn.MSELoss()
    distill_criterion = nn.MSELoss()

    best_metric = float('-inf') if student.module.task=='classification' else float('inf')
    patience_counter = 0
    detailed_metrics = None  # Initialize to None
    
    training_stats = {
        'epoch': [],
        'train_loss': [],
        'task_loss': [],
        'distill_loss': [],
        'val_metric': [],
        'lr': []
    }

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_distill_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).float() if student.module.task=='regression' else batch_y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z, _ = autoencoder(batch_x)
                y_teacher = teacher(z)
            with autocast():
                y_student = student(z)
                task_loss = task_criterion(y_student, batch_y)
                distill_loss = distill_criterion(y_student, y_teacher)
                loss = alpha*distill_loss + (1-alpha)*task_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.step(scheduler)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_distill_loss += distill_loss.item()
            num_batches += 1

        # Validation
        student.eval()
        metric = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True).float() if student.module.task=='regression' else batch_y.to(device)
                z, _ = autoencoder(batch_x)
                y_pred = student(z)
                
                if student.module.task=='classification':
                    batch_metric = (y_pred.argmax(dim=1)==batch_y).float().mean().item()
                    all_preds.append(y_pred.argmax(dim=1).cpu().numpy())
                else:
                    batch_metric = F.mse_loss(y_pred, batch_y).item()
                    all_preds.append(y_pred.cpu().numpy())
                
                metric += batch_metric
                all_labels.append(batch_y.cpu().numpy())
        
        metric /= len(val_loader)

        # all-reduce
        tensor_m = torch.tensor(metric, device=device)
        if student.module.task=='classification':
            dist.all_reduce(tensor_m, op=dist.ReduceOp.MAX)
            global_m = tensor_m.item()
        else:
            dist.all_reduce(tensor_m, op=dist.ReduceOp.MIN)
            global_m = tensor_m.item()
        
        # Collect training stats (on rank 0)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            training_stats['epoch'].append(epoch)
            training_stats['train_loss'].append(epoch_loss / num_batches)
            training_stats['task_loss'].append(epoch_task_loss / num_batches)
            training_stats['distill_loss'].append(epoch_distill_loss / num_batches)
            training_stats['val_metric'].append(global_m)
            training_stats['lr'].append(current_lr)

        if rank==0:
            scheduler.step()
            improved = ((student.module.task=='classification' and global_m>best_metric) or
                        (student.module.task!='classification' and global_m<best_metric))
            if improved:
                best_metric = global_m
                patience_counter = 0
                
                # Compute detailed metrics at best performance
                all_preds_combined = np.concatenate(all_preds)
                all_labels_combined = np.concatenate(all_labels)
                
                if student.module.task == 'classification':
                    detailed_metrics = calculate_classification_metrics(all_labels_combined, all_preds_combined)
                else:
                    detailed_metrics = calculate_regression_metrics(all_labels_combined, all_preds_combined)
            else:
                patience_counter += 1
            if early_stopping and patience_counter>patience:
                break
    
    # Log final performance metrics
    if rank == 0:
        metric_name = "accuracy" if student.module.task == 'classification' else "MSE"
        print(f"Student best {metric_name}: {best_metric:.6f}")
        if detailed_metrics is not None:
            if student.module.task == 'classification':
                print(f"Student detailed metrics: F1={detailed_metrics['f1']:.4f}, Precision={detailed_metrics['precision']:.4f}, Recall={detailed_metrics['recall']:.4f}")
            else:
                print(f"Student detailed metrics: RMSE={detailed_metrics['rmse']:.4f}, MAE={detailed_metrics['mae']:.4f}, R²={detailed_metrics['r2']:.4f}")

    return best_metric


def main():
    # ───────────────────────────────────────────────────────────────
    # Set device & init process group
    # ───────────────────────────────────────────────────────────────
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # ───────────────────────────────────────────────────────────────
    # Experiment grid - expanded for more thorough exploration
    # ───────────────────────────────────────────────────────────────
    experiment_params = {
        'num_k_values':       [2],
        'shared_methods':     ['mean', 'pca', 'nmf'],
        'decoder_types':      ['k_matrix', 'linear', 'transformer'],
        'rates':              [0.01],
        'beta_grid':          [0.1],
        'lambda_recon_grid':  [0.01],
        'alpha_grid':         [0.5],
        'batch_size':         1024,
        'autoencoder_epochs': 20,
        'teacher_epochs':     15,
        'student_epochs':     15,
        'mine_epochs':        3,
        'variational_epochs': 3,
        'latent_dim_per_factor': 16
    }

    # ───────────────────────────────────────────────────────────────
    # Create model configurations for cross-validation
    # ───────────────────────────────────────────────────────────────
    model_configs = []
    
    # Focus on a reasonable subset of combinations
    for num_k in experiment_params['num_k_values']:
        for shared_method in experiment_params['shared_methods']:
            for decoder_type in experiment_params['decoder_types']:
                beta = experiment_params['beta_grid'][0]
                lambda_recon = experiment_params['lambda_recon_grid'][0]
                alpha = experiment_params['alpha_grid'][0]
                
                model_configs.append({
                    'num_k': num_k,
                    'shared_method': shared_method,
                    'decoder_type': decoder_type,
                    'beta': beta,
                    'lambda_recon': lambda_recon,
                    'alpha': alpha,
                    'latent_dim_per_factor': experiment_params['latent_dim_per_factor']
                })
    
    # ───────────────────────────────────────────────────────────────
    # Load datasets once per rank (CPU tensors)
    # ───────────────────────────────────────────────────────────────
    (fm_x, fm_y), _ = load_fashion_mnist_from_files()
    (db_x, db_y), _ = load_diabetes_from_files()

    # ───────────────────────────────────────────────────────────────
    # We'll record results here
    # ───────────────────────────────────────────────────────────────
    master_results = []

    # ───────────────────────────────────────────────────────────────
    # Loop datasets
    # ───────────────────────────────────────────────────────────────
    for dataset_name in ['fashion_mnist', 'diabetes']:
        if dataset_name == 'fashion_mnist':
            X_full = fm_x.cpu().numpy()
            Y_full = fm_y.cpu().numpy()
            input_dim, output_dim, task = 784, 10, 'classification'
        else:
            X_full = db_x.cpu().numpy()
            Y_full = db_y.cpu().numpy()
            input_dim = db_x.shape[1]
            output_dim, task = 1, 'regression'
        
        # Run cross-validation
        if rank == 0:
            print(f"Starting cross-validation for {dataset_name}...")
        
        results_df, statistical_results = k_fold_cross_validation(
            model_configs, X_full, Y_full, n_splits=5, task=task
        )
        
        # Only rank 0 processes results
        if rank == 0 and results_df is not None:
            # Create subdirectory for this dataset
            dataset_dir = f"{output_dir}/{dataset_name}"
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            # Create visualizations and save results
            create_visualization(results_df, statistical_results, output_dir=dataset_dir)
            
            # Add dataset identifier to results
            results_df['dataset'] = dataset_name
            master_results.append(results_df)
            
            # Print summary statistics
            if task == 'classification':
                summary = results_df.groupby(['decoder_type', 'num_k', 'shared_method'])[['student_accuracy', 'mi_ksg']].agg(['mean', 'std'])
            else:
                summary = results_df.groupby(['decoder_type', 'num_k', 'shared_method'])[['student_mse', 'mi_ksg']].agg(['mean', 'std'])
            print(f"Summary Statistics for {dataset_name}:")
            print(summary)
    
    # Combine all results and save
    if rank == 0 and master_results:
        combined_results = pd.concat(master_results, ignore_index=True)
        combined_results.to_csv(f"{output_dir}/combined_results.csv", index=False)
        print(f"Results saved to {output_dir}/combined_results.csv")

    # Clean up
    if rank == 0:
        print("Cross-validation completed.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()