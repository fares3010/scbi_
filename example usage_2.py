"""
SCBI Deep Architecture Benchmark Suite
Comprehensive evaluation across diverse neural network architectures

Tests SCBI effectiveness on:
- Shallow vs Deep networks
- Narrow vs Wide networks
- Different activation functions
- Residual connections
- Batch normalization
- Dropout configurations

Author: Fares Ashraf
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats as scipy_stats
from time import time
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from scbi import SCBILinear, create_scbi_mlp



# Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# ARCHITECTURE DEFINITIONS
# ============================================================================

class StandardMLP(nn.Module):
    """Standard MLP with normal initialization."""
    def __init__(self, architecture, activation='relu', dropout=0.0, batch_norm=False):
        super().__init__()
        self.architecture = architecture
        layers = []

        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))

            # Batch norm (before activation)
            if batch_norm and i < len(architecture) - 2:
                layers.append(nn.BatchNorm1d(architecture[i+1]))

            # Activation (not on output layer)
            if i < len(architecture) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2))
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'elu':
                    layers.append(nn.ELU())

            # Dropout
            if dropout > 0 and i < len(architecture) - 2:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SCBIMLP(nn.Module):
    """MLP with SCBI initialization."""
    def __init__(self, architecture, activation='relu', dropout=0.0, batch_norm=False):
        super().__init__()
        self.architecture = architecture
        self.scbi_layers = []
        layers = []

        for i in range(len(architecture) - 1):
            scbi_layer = SCBILinear(architecture[i], architecture[i+1])
            layers.append(scbi_layer)
            self.scbi_layers.append(scbi_layer)

            # Batch norm
            if batch_norm and i < len(architecture) - 2:
                layers.append(nn.BatchNorm1d(architecture[i+1]))

            # Activation
            if i < len(architecture) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2))
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'elu':
                    layers.append(nn.ELU())

            # Dropout
            if dropout > 0 and i < len(architecture) - 2:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def init_scbi(self, X_proxy, y_proxy, verbose=False):
        """Initialize all SCBI layers."""
        if verbose:
            print(f"Initializing SCBI MLP: {self.architecture}")

        current_activation = X_proxy

        for i, layer in enumerate(self.scbi_layers):
            # Last layer gets true targets
            is_last = (i == len(self.scbi_layers) - 1)
            target = y_proxy if is_last else None

            layer.init_weights_with_proxy(
                current_activation,
                target,
                verbose=verbose and i == 0  # Only verbose for first layer
            )

            # Forward through this layer and non-SCBI layers
            with torch.no_grad():
                layer_idx = 0
                for module in self.network:
                    if isinstance(module, SCBILinear):
                        if module == layer:
                            current_activation = module(current_activation)
                            break
                        layer_idx += 1
                    else:
                        if layer_idx == i:
                            current_activation = module(current_activation)

        if verbose:
            print(f"✅ SCBI initialization complete\n")

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, dim, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out


class ResNetStyle(nn.Module):
    """ResNet-style architecture for tabular data."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3, activation='relu'):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation) for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


# ============================================================================
# ARCHITECTURE BENCHMARK CLASS
# ============================================================================

class ArchitectureBenchmark:
    """Benchmark SCBI across different architectures."""

    def __init__(self, n_runs=3, n_epochs=30, device='cpu'):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.device = device
        self.results = []

    def load_dataset(self, dataset_name='california_housing'):
        """Load and prepare dataset."""
        print(f"\nLoading {dataset_name}...")

        if dataset_name == 'california_housing':
            data = fetch_california_housing()
            X, y = data.data, data.target
            task = 'regression'
        elif dataset_name == 'classification':
            X, y = make_classification(
                n_samples=2000, n_features=50, n_informative=30,
                n_classes=3, random_state=RANDOM_SEED
            )
            task = 'classification'

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # To tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        if task == 'regression':
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            y_train_onehot = y_train
        else:
            n_classes = int(y_train.max().item()) + 1
            y_train_onehot = torch.nn.functional.one_hot(
                y_train.long(), num_classes=n_classes
            ).float()

        print(f"Task: {task}, Samples: {len(X_train)}, Features: {X_train.shape[1]}")

        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_onehot': y_train_onehot,
            'task': task, 'n_features': X_train.shape[1],
            'n_outputs': 1 if task == 'regression' else n_classes
        }

    def train_model(self, model, X_train, y_train, X_test, y_test, task, lr=0.01):
        """Train and evaluate a model."""
        criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        test_losses = []

        for epoch in range(self.n_epochs):
            # Train
            model.train()
            optimizer.zero_grad()

            pred = model(X_train)
            loss = criterion(pred, y_train if task == 'regression' else y_train.long())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Test
            model.eval()
            with torch.no_grad():
                pred_test = model(X_test)
                test_loss = criterion(
                    pred_test,
                    y_test if task == 'regression' else y_test.long()
                )
                test_losses.append(test_loss.item())

        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'initial_train': train_losses[0],
            'final_train': train_losses[-1],
            'initial_test': test_losses[0],
            'final_test': test_losses[-1]
        }

    def benchmark_architecture(self, architecture, data, name,
                              activation='relu', dropout=0.0, batch_norm=False):
        """Compare Standard vs SCBI for one architecture."""

        print(f"\n{'='*70}")
        print(f"Architecture: {name}")
        print(f"Layers: {architecture}")
        print(f"Activation: {activation}, Dropout: {dropout}, BatchNorm: {batch_norm}")
        print('='*70)

        # Proxy sample
        proxy_size = min(500, int(0.3 * len(data['X_train'])))
        X_proxy = data['X_train'][:proxy_size]
        y_proxy = data['y_train_onehot'][:proxy_size]

        results_std = []
        results_scbi = []
        init_times = []

        for run in range(self.n_runs):
            print(f"\nRun {run+1}/{self.n_runs}")

            # Standard initialization
            print("  Training Standard init...", end=' ')
            model_std = StandardMLP(
                architecture, activation, dropout, batch_norm
            ).to(self.device)

            start = time()
            result_std = self.train_model(
                model_std, data['X_train'], data['y_train'],
                data['X_test'], data['y_test'], data['task']
            )
            std_time = time() - start
            results_std.append(result_std)
            print(f"Initial: {result_std['initial_train']:.2f}")

            # SCBI initialization
            print("  Training SCBI init...", end=' ')
            model_scbi = SCBIMLP(
                architecture, activation, dropout, batch_norm
            ).to(self.device)

            # Initialize
            init_start = time()
            model_scbi.init_scbi(X_proxy, y_proxy, verbose=(run==0))
            init_time = time() - init_start
            init_times.append(init_time)

            start = time()
            result_scbi = self.train_model(
                model_scbi, data['X_train'], data['y_train'],
                data['X_test'], data['y_test'], data['task']
            )
            scbi_time = time() - start
            results_scbi.append(result_scbi)
            print(f"Initial: {result_scbi['initial_train']:.2f}")

        # Compute statistics
        std_initial = [r['initial_train'] for r in results_std]
        scbi_initial = [r['initial_train'] for r in results_scbi]

        std_final = [r['final_test'] for r in results_std]
        scbi_final = [r['final_test'] for r in results_scbi]

        improvement_initial = (np.mean(std_initial) - np.mean(scbi_initial)) / np.mean(std_initial) * 100
        improvement_final = (np.mean(std_final) - np.mean(scbi_final)) / np.mean(std_final) * 100

        t_stat, p_value = scipy_stats.ttest_rel(std_initial, scbi_initial)

        result = {
            'name': name,
            'architecture': architecture,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'depth': len(architecture) - 1,
            'width': max(architecture[1:-1]) if len(architecture) > 2 else architecture[1],
            'total_params': sum(architecture[i] * architecture[i+1] for i in range(len(architecture)-1)),
            'std_initial_mean': np.mean(std_initial),
            'std_initial_std': np.std(std_initial),
            'scbi_initial_mean': np.mean(scbi_initial),
            'scbi_initial_std': np.std(scbi_initial),
            'std_final_mean': np.mean(std_final),
            'scbi_final_mean': np.mean(scbi_final),
            'improvement_initial': improvement_initial,
            'improvement_final': improvement_final,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'init_time_mean': np.mean(init_times),
            'results_std': results_std,
            'results_scbi': results_scbi
        }

        # Print summary
        print(f"\n{'─'*70}")
        print(f"RESULTS: {name}")
        print(f"Initial Loss: Standard={result['std_initial_mean']:.2f}±{result['std_initial_std']:.2f}, "
              f"SCBI={result['scbi_initial_mean']:.2f}±{result['scbi_initial_std']:.2f}")
        print(f"Improvement: {improvement_initial:.1f}% "
              f"({'✅ p='+f'{p_value:.4f}' if p_value < 0.05 else '⚠️ p='+f'{p_value:.4f}'})")
        print(f"Final Test Loss: Standard={result['std_final_mean']:.2f}, SCBI={result['scbi_final_mean']:.2f}")
        print(f"Init Time: {result['init_time_mean']:.3f}s")
        print('─'*70)

        self.results.append(result)
        return result

    def run_architecture_suite(self, data):
        """Run comprehensive architecture benchmark."""

        print("\n" + "="*70)
        print("COMPREHENSIVE ARCHITECTURE BENCHMARK")
        print("="*70)

        # 1. DEPTH COMPARISON
        print("\n" + "▶"*35)
        print("EXPERIMENT 1: EFFECT OF NETWORK DEPTH")
        print("▶"*35)

        self.benchmark_architecture(
            [data['n_features'], data['n_outputs']],
            data, "Shallow (1 layer)"
        )

        self.benchmark_architecture(
            [data['n_features'], 128, data['n_outputs']],
            data, "Medium (2 layers)"
        )

        self.benchmark_architecture(
            [data['n_features'], 256, 128, data['n_outputs']],
            data, "Deep (3 layers)"
        )

        self.benchmark_architecture(
            [data['n_features'], 512, 256, 128, 64, data['n_outputs']],
            data, "Very Deep (5 layers)"
        )

        # 2. WIDTH COMPARISON
        print("\n" + "▶"*35)
        print("EXPERIMENT 2: EFFECT OF NETWORK WIDTH")
        print("▶"*35)

        self.benchmark_architecture(
            [data['n_features'], 32, 16, data['n_outputs']],
            data, "Narrow (32-16)"
        )

        self.benchmark_architecture(
            [data['n_features'], 128, 64, data['n_outputs']],
            data, "Medium (128-64)"
        )

        self.benchmark_architecture(
            [data['n_features'], 512, 256, data['n_outputs']],
            data, "Wide (512-256)"
        )

        # 3. ACTIVATION FUNCTIONS
        print("\n" + "▶"*35)
        print("EXPERIMENT 3: DIFFERENT ACTIVATION FUNCTIONS")
        print("▶"*35)

        base_arch = [data['n_features'], 256, 128, data['n_outputs']]

        self.benchmark_architecture(
            base_arch, data, "ReLU activation", activation='relu'
        )

        self.benchmark_architecture(
            base_arch, data, "LeakyReLU activation", activation='leaky_relu'
        )

        self.benchmark_architecture(
            base_arch, data, "Tanh activation", activation='tanh'
        )

        self.benchmark_architecture(
            base_arch, data, "ELU activation", activation='elu'
        )

        # 4. REGULARIZATION
        print("\n" + "▶"*35)
        print("EXPERIMENT 4: REGULARIZATION TECHNIQUES")
        print("▶"*35)

        self.benchmark_architecture(
            base_arch, data, "No regularization", dropout=0.0, batch_norm=False
        )

        self.benchmark_architecture(
            base_arch, data, "Dropout 0.2", dropout=0.2, batch_norm=False
        )

        self.benchmark_architecture(
            base_arch, data, "Dropout 0.5", dropout=0.5, batch_norm=False
        )

        self.benchmark_architecture(
            base_arch, data, "BatchNorm", dropout=0.0, batch_norm=True
        )

        self.benchmark_architecture(
            base_arch, data, "Dropout + BatchNorm", dropout=0.2, batch_norm=True
        )

    def create_visualizations(self, output_dir='architecture_figures'):
        """Create comprehensive visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Creating Architecture Analysis Visualizations")
        print('='*70)

        # Figure 1: Depth vs Performance
        self.plot_depth_analysis(output_dir)

        # Figure 2: Width vs Performance
        self.plot_width_analysis(output_dir)

        # Figure 3: Activation functions
        self.plot_activation_analysis(output_dir)

        # Figure 4: Regularization comparison
        self.plot_regularization_analysis(output_dir)

        # Figure 5: Overall heatmap
        self.plot_overall_heatmap(output_dir)

        # Figure 6: Training curves comparison
        self.plot_training_curves_comparison(output_dir)

        print(f"\n✅ All figures saved to '{output_dir}/'")

    def plot_depth_analysis(self, output_dir):
        """Plot effect of network depth."""
        depth_results = [r for r in self.results if 'layer' in r['name'].lower() and 'width' not in r['name'].lower()]

        if not depth_results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        depths = [r['depth'] for r in depth_results]
        names = [r['name'] for r in depth_results]

        # Initial loss comparison
        std_initial = [r['std_initial_mean'] for r in depth_results]
        scbi_initial = [r['scbi_initial_mean'] for r in depth_results]

        x = np.arange(len(names))
        width = 0.35

        ax1.bar(x - width/2, std_initial, width, label='Standard', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, scbi_initial, width, label='SCBI', color='coral', alpha=0.8)
        ax1.set_xlabel('Network Configuration')
        ax1.set_ylabel('Initial Training Loss')
        ax1.set_title('Effect of Network Depth on Initial Loss')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Improvement vs depth
        improvements = [r['improvement_initial'] for r in depth_results]
        colors = ['green' if r['significant'] else 'orange' for r in depth_results]

        ax2.plot(depths, improvements, 'o-', linewidth=2, markersize=10, color='darkblue')
        for i, (d, imp, c) in enumerate(zip(depths, improvements, colors)):
            ax2.scatter(d, imp, s=200, c=c, alpha=0.5, edgecolors='black', linewidths=2)
        ax2.set_xlabel('Network Depth (Number of Layers)')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('SCBI Improvement vs Network Depth')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig1_depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig1_depth_analysis.pdf', bbox_inches='tight')
        print("✓ Figure 1: Depth analysis")
        plt.close()

    def plot_width_analysis(self, output_dir):
        """Plot effect of network width."""
        width_results = [r for r in self.results if any(w in r['name'].lower() for w in ['narrow', 'wide']) or 'medium (128-64)' in r['name'].lower()]

        if not width_results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        widths = [r['width'] for r in width_results]
        names = [r['name'] for r in width_results]

        # Sort by width
        sorted_indices = np.argsort(widths)
        widths = [widths[i] for i in sorted_indices]
        names = [names[i] for i in sorted_indices]
        width_results = [width_results[i] for i in sorted_indices]

        # Initial loss vs width
        std_initial = [r['std_initial_mean'] for r in width_results]
        scbi_initial = [r['scbi_initial_mean'] for r in width_results]

        x = np.arange(len(names))
        width_bar = 0.35

        ax1.bar(x - width_bar/2, std_initial, width_bar, label='Standard', color='steelblue', alpha=0.8)
        ax1.bar(x + width_bar/2, scbi_initial, width_bar, label='SCBI', color='coral', alpha=0.8)
        ax1.set_xlabel('Network Configuration')
        ax1.set_ylabel('Initial Training Loss')
        ax1.set_title('Effect of Network Width on Initial Loss')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Improvement vs width
        improvements = [r['improvement_initial'] for r in width_results]

        ax2.plot(widths, improvements, 'o-', linewidth=2, markersize=10, color='darkgreen')
        ax2.set_xlabel('Network Width (Max Hidden Units)')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('SCBI Improvement vs Network Width')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig2_width_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig2_width_analysis.pdf', bbox_inches='tight')
        print("✓ Figure 2: Width analysis")
        plt.close()

    def plot_activation_analysis(self, output_dir):
        """Plot effect of different activation functions."""
        activation_results = [r for r in self.results if 'activation' in r['name'].lower()]

        if not activation_results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        names = [r['name'].replace(' activation', '') for r in activation_results]
        std_initial = [r['std_initial_mean'] for r in activation_results]
        std_err = [r['std_initial_std'] for r in activation_results]
        scbi_initial = [r['scbi_initial_mean'] for r in activation_results]
        scbi_err = [r['scbi_initial_std'] for r in activation_results]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, std_initial, width, yerr=std_err,
                      label='Standard', color='steelblue', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, scbi_initial, width, yerr=scbi_err,
                      label='SCBI', color='coral', alpha=0.8, capsize=5)

        # Add improvement percentages
        for i, r in enumerate(activation_results):
            imp = r['improvement_initial']
            sig = '✅' if r['significant'] else '⚠️'
            y_pos = max(std_initial[i], scbi_initial[i]) * 1.1
            ax.text(i, y_pos, f"{sig}\n{imp:.1f}%",
                   ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Activation Function')
        ax.set_ylabel('Initial Training Loss')
        ax.set_title('Effect of Activation Functions on SCBI Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig3_activation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig3_activation_analysis.pdf', bbox_inches='tight')
        print("✓ Figure 3: Activation function analysis")
        plt.close()

    def plot_regularization_analysis(self, output_dir):
        """Plot effect of regularization techniques."""
        reg_results = [r for r in self.results if any(keyword in r['name'].lower()
                      for keyword in ['dropout', 'batchnorm', 'no regularization'])]

        if not reg_results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        names = [r['name'] for r in reg_results]
        improvements = [r['improvement_initial'] for r in reg_results]
        p_values = [r['p_value'] for r in reg_results]
        colors = ['green' if r['significant'] else 'orange' for r in reg_results]

        # Improvement comparison
        ax1.barh(names, improvements, color=colors, alpha=0.7)
        ax1.set_xlabel('Initial Loss Improvement (%)')
        ax1.set_title('SCBI Improvement with Different Regularization')
        ax1.grid(axis='x', alpha=0.3)

        # P-values
        ax2.barh(names, p_values, color=colors, alpha=0.7)
        ax2.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
        ax2.set_xlabel('P-value')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig4_regularization_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig4_regularization_analysis.pdf', bbox_inches='tight')
        print("✓ Figure 4: Regularization analysis")
        plt.close()

    def plot_overall_heatmap(self, output_dir):
        """Heatmap of improvement across all architectures."""
        fig, ax = plt.subplots(figsize=(12, max(8, len(self.results) * 0.4)))

        names = [r['name'] for r in self.results]

        # Create data matrix
        metrics = ['Improvement (%)', 'Depth', 'Width', '-log10(p-value)', 'Init Time (s)']
        data = []

        for r in self.results:
            row = [
                r['improvement_initial'],
                r['depth'],
                np.log10(r['width']),  # Log scale for better visualization
                -np.log10(r['p_value']) if r['p_value'] > 0 else 10,  # -log10 for better visualization
                r['init_time_mean']
            ]
            data.append(row)

        data = np.array(data)

        # Normalize each column
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_norm = scaler.fit_transform(data)

        im = ax.imshow(data_norm.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(metrics)

        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(names)):
                text = ax.text(j, i, f'{data[j, i]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Architecture Performance Heatmap\n(Green=Better, Red=Worse)')
        plt.colorbar(im, ax=ax, label='Normalized Score')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig5_overall_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig5_overall_heatmap.pdf', bbox_inches='tight')
        print("✓ Figure 5: Overall heatmap")
        plt.close()

    def plot_training_curves_comparison(self, output_dir):
        """Compare training curves for key architectures."""
        # Select representative architectures
        key_archs = ['Shallow (1 layer)', 'Deep (3 layers)', 'Very Deep (5 layers)']
        selected = [r for r in self.results if r['name'] in key_archs]

        if not selected:
            selected = self.results[:3]  # Fallback to first 3

        fig, axes = plt.subplots(1, len(selected), figsize=(15, 4))

        if len(selected) == 1:
            axes = [axes]

        for idx, result in enumerate(selected):
            ax = axes[idx]
            epochs = range(1, self.n_epochs + 1)

            # Standard
            std_losses = np.array([r['train_losses'] for r in result['results_std']])
            std_mean = std_losses.mean(axis=0)
            std_std = std_losses.std(axis=0)

            ax.plot(epochs, std_mean, 'b-', linewidth=2, label='Standard', alpha=0.8)
            ax.fill_between(epochs, std_mean - std_std, std_mean + std_std,
                           color='b', alpha=0.2)

            # SCBI
            scbi_losses = np.array([r['train_losses'] for r in result['results_scbi']])
            scbi_mean = scbi_losses.mean(axis=0)
            scbi_std = scbi_losses.std(axis=0)

            ax.plot(epochs, scbi_mean, 'r-', linewidth=2, label='SCBI', alpha=0.8)
            ax.fill_between(epochs, scbi_mean - scbi_std, scbi_mean + scbi_std,
                           color='r', alpha=0.2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title(f"{result['name']}\n({result['improvement_initial']:.1f}% improvement)")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/arch_fig6_training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/arch_fig6_training_curves.pdf', bbox_inches='tight')
        print("✓ Figure 6: Training curves comparison")
        plt.close()

    def export_results(self, output_dir='architecture_results'):
        """Export results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Exporting Architecture Results")
        print('='*70)

        # CSV
        rows = []
        for r in self.results:
            row = {
                'Architecture': r['name'],
                'Depth': r['depth'],
                'Width': r['width'],
                'Activation': r['activation'],
                'Dropout': r['dropout'],
                'BatchNorm': r['batch_norm'],
                'Total_Params': r['total_params'],
                'Standard_Initial': r['std_initial_mean'],
                'SCBI_Initial': r['scbi_initial_mean'],
                'Improvement_%': r['improvement_initial'],
                'P_Value': r['p_value'],
                'Significant': r['significant'],
                'Init_Time_s': r['init_time_mean']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(f'{output_dir}/architecture_results.csv', index=False)
        print(f"✓ CSV: {output_dir}/architecture_results.csv")

        # JSON
        with open(f'{output_dir}/architecture_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
        print(f"✓ JSON: {output_dir}/architecture_results.json")

        # LaTeX table
        latex = self.create_latex_table(df)
        with open(f'{output_dir}/architecture_table.tex', 'w') as f:
            f.write(latex)
        print(f"✓ LaTeX: {output_dir}/architecture_table.tex")

    def create_latex_table(self, df):
        """Create LaTeX table."""
        latex = r"""\begin{table}[htbp]
\centering
\caption{SCBI Performance Across Network Architectures}
\label{tab:architecture_results}
\begin{tabular}{lcccc}
\toprule
Architecture & Depth & Width & Improvement & $p$-value \\
& & & (\%) & \\
\midrule
"""
        for _, row in df.iterrows():
            name = row['Architecture'].replace('_', ' ')
            depth = int(row['Depth'])
            width = int(row['Width'])
            imp = f"{row['Improvement_%']:.1f}"
            p_val = f"{row['P_Value']:.4f}"
            sig = r"\textbf{*}" if row['Significant'] else ""

            latex += f"{name} & {depth} & {width} & {imp}{sig} & {p_val} \\\\\n"

        latex += r"""\bottomrule
\multicolumn{5}{l}{\small * Statistically significant at $\alpha=0.05$}
\end{tabular}
\end{table}"""

        return latex


def main():
    """Run architecture benchmark suite."""

    print("\n" + "="*70)
    print("SCBI DEEP ARCHITECTURE BENCHMARK SUITE")
    print("Comprehensive Evaluation Across Neural Network Architectures")
    print("="*70)
    print(f"\nAuthor: Fares Ashraf")
    print(f"DOI: 10.5281/zenodo.18576203")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*70)

    # Initialize
    benchmark = ArchitectureBenchmark(
        n_runs=3,       # 3 runs per architecture for faster completion
        n_epochs=30,
        device='cpu'    # Change to 'cuda' for GPU
    )

    # Load dataset
    data = benchmark.load_dataset('california_housing')

    # Run comprehensive suite
    benchmark.run_architecture_suite(data)

    # Visualize
    benchmark.create_visualizations()

    # Export
    benchmark.export_results()

    print("\n" + "="*70)
    print("✅ ARCHITECTURE BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  architecture_figures/arch_fig1_depth_analysis.png (PDF)")
    print("  architecture_figures/arch_fig2_width_analysis.png (PDF)")
    print("  architecture_figures/arch_fig3_activation_analysis.png (PDF)")
    print("  architecture_figures/arch_fig4_regularization_analysis.png (PDF)")
    print("  architecture_figures/arch_fig5_overall_heatmap.png (PDF)")
    print("  architecture_figures/arch_fig6_training_curves.png (PDF)")
    print("  architecture_results/architecture_results.csv")
    print("  architecture_results/architecture_table.tex")
    print("  architecture_results/architecture_results.json")
    print("\n✨ Ready for publication!")
    print("="*70)


if __name__ == "__main__":
    main()
