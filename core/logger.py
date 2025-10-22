"""
Flexible logging system supporting multiple backends:
- Weights & Biases (wandb) with local syncing
- File-based logging (JSON/CSV)
- Plot generation (matplotlib)
"""

import json
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import warnings

import numpy as np
import torch

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # AttributeError can occur with NumPy 2.0 incompatibility
    WANDB_AVAILABLE = False
    warnings.warn(f"wandb not available: {e}. Install compatible version with: pip install wandb --upgrade")

# TensorBoard import - fails with NumPy 2.0, so make it completely optional
TENSORBOARD_AVAILABLE = False
SummaryWriter = None
# Don't even try to import - tensorboard is purely optional, we have wandb and file logging
# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_AVAILABLE = True
# except (ImportError, AttributeError) as e:
#     TENSORBOARD_AVAILABLE = False


class BaseLogger:
    """Base class for all logger backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics at given step."""
        raise NotImplementedError

    def finish(self):
        """Clean up resources."""
        pass


class WandBLogger(BaseLogger):
    """Weights & Biases logger with local syncing."""

    def __init__(self, config: Dict[str, Any], experiment_config: Dict[str, Any]):
        super().__init__(config)

        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Install with: pip install wandb")

        self.experiment_config = experiment_config

        # Initialize wandb
        wandb_config = config.get('wandb', {})

        # Get project and entity from experiment config or use defaults
        project = experiment_config.get('logging', {}).get('wandb_project') or \
                  wandb_config.get('default_project', 'heavy-tail-scaling')
        entity = experiment_config.get('logging', {}).get('wandb_entity') or \
                 wandb_config.get('default_entity', None)
        tags = experiment_config.get('logging', {}).get('wandb_tags', [])

        # Local sync directory
        if wandb_config.get('sync_local', True):
            local_dir = wandb_config.get('local_dir', 'outputs/wandb_local')
            os.makedirs(local_dir, exist_ok=True)
        else:
            local_dir = None

        # Initialize run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_config.get('experiment', {}).get('name'),
            config=experiment_config,
            tags=tags,
            dir=local_dir,
            reinit=True,
        )

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb."""
        # Convert torch tensors to python types
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed_metrics[key] = value.item()
                else:
                    # Log as histogram for multi-element tensors
                    processed_metrics[key] = wandb.Histogram(value.cpu().numpy())
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    processed_metrics[key] = float(value)
                else:
                    processed_metrics[key] = wandb.Histogram(value)
            else:
                processed_metrics[key] = value

        wandb.log(processed_metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.run is not None:
            self.run.finish()


class FileLogger(BaseLogger):
    """File-based logger supporting JSON and CSV formats."""

    def __init__(self, config: Dict[str, Any], experiment_config: Dict[str, Any]):
        super().__init__(config)

        self.experiment_config = experiment_config
        file_config = config.get('files', {})

        # Setup output directory
        experiment_name = experiment_config.get('experiment', {}).get('name', 'unnamed')
        # Check experiment config first, then logging config
        base_dir = (experiment_config.get('experiment', {}).get('output_dir') or
                   file_config.get('output_dir', 'outputs/logs'))
        self.output_dir = Path(base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported formats
        self.formats = file_config.get('formats', ['json'])

        # Initialize file handles
        self.file_handles = {}
        self.csv_writers = {}

        if 'csv' in self.formats:
            self.csv_file = open(self.output_dir / 'metrics.csv', 'w', newline='')
            self.csv_writer = None  # Will be initialized on first log

        if 'json' in self.formats:
            self.json_file = open(self.output_dir / 'metrics.jsonl', 'w')

        # Buffer for batch writing
        self.buffer = []
        self.buffer_size = config.get('advanced', {}).get('max_log_buffer_size', 1000)

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to files."""
        # Add step to metrics
        log_entry = {'step': step}

        # Convert torch tensors and numpy arrays to python types
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    log_entry[key] = value.item()
                else:
                    # For multi-element tensors, log statistics
                    log_entry[f"{key}_mean"] = value.mean().item()
                    log_entry[f"{key}_std"] = value.std().item()
                    log_entry[f"{key}_min"] = value.min().item()
                    log_entry[f"{key}_max"] = value.max().item()
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    log_entry[key] = float(value)
                else:
                    log_entry[f"{key}_mean"] = float(np.mean(value))
                    log_entry[f"{key}_std"] = float(np.std(value))
                    log_entry[f"{key}_min"] = float(np.min(value))
                    log_entry[f"{key}_max"] = float(np.max(value))
            elif isinstance(value, (int, float, str, bool)):
                log_entry[key] = value
            else:
                # Try to convert to string
                log_entry[key] = str(value)

        # JSON logging
        if 'json' in self.formats and hasattr(self, 'json_file'):
            json.dump(log_entry, self.json_file)
            self.json_file.write('\n')
            self.json_file.flush()

        # CSV logging
        if 'csv' in self.formats and hasattr(self, 'csv_file'):
            if self.csv_writer is None:
                # Initialize CSV writer with headers from first log
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=log_entry.keys())
                self.csv_writer.writeheader()

            try:
                self.csv_writer.writerow(log_entry)
                self.csv_file.flush()
            except ValueError:
                # New keys appeared, reinitialize writer
                self.csv_file.close()
                self.csv_file = open(self.output_dir / 'metrics.csv', 'a', newline='')
                # Append mode, don't write header again
                all_keys = list(log_entry.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=all_keys)
                self.csv_writer.writerow(log_entry)
                self.csv_file.flush()

    def finish(self):
        """Close file handles."""
        if hasattr(self, 'json_file'):
            self.json_file.close()
        if hasattr(self, 'csv_file'):
            self.csv_file.close()


class PlotLogger(BaseLogger):
    """Logger that generates and saves plots."""

    def __init__(self, config: Dict[str, Any], experiment_config: Dict[str, Any]):
        super().__init__(config)

        plot_config = config.get('plots', {})

        # Setup output directory
        experiment_name = experiment_config.get('experiment', {}).get('name', 'unnamed')
        # Check experiment config first, then logging config
        base_dir = (experiment_config.get('experiment', {}).get('output_dir') or
                   plot_config.get('output_dir', 'outputs/plots'))
        self.output_dir = Path(base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Plot settings
        self.formats = plot_config.get('formats', ['png'])
        self.dpi = plot_config.get('dpi', 300)

        # Store metrics history for plotting
        self.metrics_history = defaultdict(list)
        self.steps = []

    def log(self, metrics: Dict[str, Any], step: int):
        """Store metrics for later plotting."""
        self.steps.append(step)

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    self.metrics_history[key].append(value.item())
            elif isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    self.metrics_history[key].append(float(value))

    def save_plots(self):
        """Generate and save all plots. Call this periodically or at end."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        sns.set_style("whitegrid")

        # Plot alpha evolution
        alpha_keys = [k for k in self.metrics_history.keys() if 'alpha' in k.lower()]
        if alpha_keys:
            fig, ax = plt.subplots(figsize=(10, 6))
            for key in alpha_keys:
                if len(self.metrics_history[key]) == len(self.steps):
                    ax.plot(self.steps, self.metrics_history[key], label=key, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Tail Index (α)')
            ax.set_title('Tail Index Evolution During Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

            for fmt in self.formats:
                fig.savefig(self.output_dir / f'alpha_evolution.{fmt}', dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        # Plot loss if available
        loss_keys = [k for k in self.metrics_history.keys() if 'loss' in k.lower()]
        if loss_keys:
            fig, ax = plt.subplots(figsize=(10, 6))
            for key in loss_keys:
                if len(self.metrics_history[key]) == len(self.steps):
                    ax.plot(self.steps, self.metrics_history[key], label=key, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            for fmt in self.formats:
                fig.savefig(self.output_dir / f'loss.{fmt}', dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

    def finish(self):
        """Generate final plots."""
        self.save_plots()


class FlexibleLogger:
    """
    Main logger class that manages multiple backends.

    Usage:
        logger = FlexibleLogger(logging_config, experiment_config)
        logger.log({'loss': 0.5, 'alpha': 1.5}, step=100)
        logger.finish()
    """

    def __init__(self, logging_config_path: Optional[str] = None,
                 experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize flexible logger.

        Args:
            logging_config_path: Path to logging_config.yaml
            experiment_config: Full experiment configuration dict
        """
        self.backends = []
        self.experiment_config = experiment_config or {}

        # Load logging config
        if logging_config_path:
            import yaml
            with open(logging_config_path, 'r') as f:
                self.logging_config = yaml.safe_load(f)
        else:
            self.logging_config = {}

        # Override with experiment-specific logging settings
        exp_logging = self.experiment_config.get('logging', {})

        # Initialize backends based on configuration
        backends_config = self.logging_config.get('backends', {})

        # WandB
        if exp_logging.get('use_wandb', backends_config.get('wandb', {}).get('enabled', False)):
            if WANDB_AVAILABLE:
                try:
                    self.backends.append(WandBLogger(self.logging_config, self.experiment_config))
                    print("✓ WandB logger initialized")
                except Exception as e:
                    warnings.warn(f"Failed to initialize WandB logger: {e}")
            else:
                warnings.warn("WandB requested but not available")

        # File logger
        if exp_logging.get('use_files', backends_config.get('files', {}).get('enabled', True)):
            try:
                self.backends.append(FileLogger(self.logging_config, self.experiment_config))
                print("✓ File logger initialized")
            except Exception as e:
                warnings.warn(f"Failed to initialize File logger: {e}")

        # Plot logger
        if exp_logging.get('save_plots', backends_config.get('plots', {}).get('enabled', True)):
            try:
                plot_logger = PlotLogger(self.logging_config, self.experiment_config)
                self.backends.append(plot_logger)
                self.plot_logger = plot_logger  # Keep reference for plot generation
                print("✓ Plot logger initialized")
            except Exception as e:
                warnings.warn(f"Failed to initialize Plot logger: {e}")

        if not self.backends:
            warnings.warn("No logging backends initialized!")

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all backends."""
        for backend in self.backends:
            try:
                backend.log(metrics, step)
            except Exception as e:
                warnings.warn(f"Error logging to {backend.__class__.__name__}: {e}")

    def save_plots(self):
        """Manually trigger plot generation (for periodic saving)."""
        if hasattr(self, 'plot_logger'):
            try:
                self.plot_logger.save_plots()
            except Exception as e:
                warnings.warn(f"Error saving plots: {e}")

    def finish(self):
        """Finish all backends and save final outputs."""
        print("\nFinalizing logging...")
        for backend in self.backends:
            try:
                backend.finish()
                print(f"✓ {backend.__class__.__name__} finalized")
            except Exception as e:
                warnings.warn(f"Error finishing {backend.__class__.__name__}: {e}")
        print("Logging complete.")


def get_system_info() -> Dict[str, Any]:
    """Get system information for reproducibility."""
    import platform
    import subprocess

    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # Try to get git commit
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        info['git_commit'] = git_commit

        # Check if repo is dirty
        git_status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        info['git_dirty'] = len(git_status) > 0
    except:
        pass

    return info
