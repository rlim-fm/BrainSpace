from train import Processor
from visualization import Visualizer
from itertools import product
import os
import json
import csv
from datetime import datetime
import numpy as np

class Experiment:
    def __init__(self, base_config=None, ivs=None, *, trials=5, output_root='results'):
        """
        Initialize an Experiment for grid search with multiple trials.

        Args:
            base_config: Base configuration dictionary for all experiments
            ivs: Independent variables dict {param_name: [values]}
            trials: Number of trials per configuration
            output_root: Root directory for organizing results
        """
        self.base_config = base_config or {}
        self.ivs = ivs or {}
        self.trials = trials
        self.output_root = output_root
        self._create_configs()
        self.results = []

    def run_grid(self, visualize=True, save_visualizations=True):
        """
        Run the full experimental grid with training, visualization, and result collection.

        Args:
            visualize: Whether to run visualization pipeline for each trial
            save_visualizations: Whether to save visualization outputs to disk
        """
        print(f"\n{'='*80}")
        print(f"Starting experiment grid with {len(self.configs)} configs × {self.trials} trials")
        print(f"{'='*80}\n")

        for config_idx, config in enumerate(self.configs):
            config_name = self._config_to_name(config_idx, config)
            config_dir = os.path.join(self.output_root, f'config_{config_idx}')

            print(f"\n{'─'*80}")
            print(f"CONFIG {config_idx}: {config_name}")
            print(f"{'─'*80}")

            for trial_idx in range(self.trials):
                trial_name = f"trial_{trial_idx}"
                trial_dir = os.path.join(config_dir, trial_name)
                os.makedirs(trial_dir, exist_ok=True)

                print(f"\n  [{config_idx}.{trial_idx}] {trial_name}...", end=" ")

                # Set seed for reproducibility
                seed = trial_idx
                config_with_seed = config.copy()
                config_with_seed['seed'] = seed

                try:
                    # Train model
                    processor = Processor(**config_with_seed)
                    processor.run()

                    # Collect results
                    final_train_loss = float(processor.logs['train_loss'][-1])
                    final_test_loss = float(processor.logs['test_loss'][-1])
                    best_test_loss = float(np.min(processor.logs['test_loss']))
                    best_test_epoch = int(np.argmin(processor.logs['test_loss']))

                    result_entry = {
                        'config_idx': config_idx,
                        'trial_idx': trial_idx,
                        'config_name': config_name,
                        'seed': seed,
                        'final_train_loss': final_train_loss,
                        'final_test_loss': final_test_loss,
                        'best_test_loss': best_test_loss,
                        'best_test_epoch': best_test_epoch,
                        'epochs': processor.epochs,
                        **{f'config_{k}': v for k, v in config.items()}
                    }
                    self.results.append(result_entry)

                    # Save training data and model
                    h5_filename = os.path.join(trial_dir, 'training_data.h5')
                    processor.save(h5_filename, output_dir=trial_dir)

                    msg = f"✓ Train Loss: {final_train_loss:.6f}, Test Loss: {final_test_loss:.6f}"
                    print(msg)

                    # Run visualization if requested
                    if visualize:
                        print(f"    Generating visualizations...", end=" ")
                        visualizer = Visualizer.from_processor_data(processor)
                        if save_visualizations:
                            vis_dir = os.path.join(trial_dir, 'visualizations')
                            os.makedirs(vis_dir, exist_ok=True)
                            self._run_visualizations(visualizer, trial_name, vis_dir)
                            print("✓")
                        else:
                            print("(skipped)")

                    # Save config for this trial
                    config_file = os.path.join(trial_dir, 'config.json')
                    with open(config_file, 'w') as f:
                        # Convert non-serializable objects to strings
                        config_dict = {}
                        for k, v in config_with_seed.items():
                            if isinstance(v, (str, int, float, bool, type(None))):
                                config_dict[k] = v
                            else:
                                config_dict[k] = str(v)
                        json.dump(config_dict, f, indent=2)

                except Exception as e:
                    print(f"✗ FAILED: {e}")
                    result_entry = {
                        'config_idx': config_idx,
                        'trial_idx': trial_idx,
                        'config_name': config_name,
                        'seed': trial_idx,
                        'final_train_loss': None,
                        'final_test_loss': None,
                        'best_test_loss': None,
                        'best_test_epoch': None,
                        'epochs': None,
                        'error': str(e),
                        **{f'config_{k}': v for k, v in config.items()}
                    }
                    self.results.append(result_entry)

        # Generate summary files
        self._save_results_summary()
        print(f"\n{'='*80}")
        print(f"Experiment complete! Results saved to '{self.output_root}'")
        print(f"{'='*80}\n")

    def _run_visualizations(self, visualizer, trial_name, output_dir):
        """Run all visualization methods for a trial."""
        try:
            # Loss history
            visualizer.plot_loss_history(
                output_path=os.path.join(output_dir, f'{trial_name}_loss_history.png')
            )

            # 1D convergence
            visualizer.convergence_visualization_1d(
                axis=0,
                output_path=os.path.join(output_dir, f'{trial_name}_1d_convergence.gif')
            )

            # 3D PCA - anchor epoch mode
            visualizer.hidden_layer_visualization(
                pca_epoch=-1,
                output_path=os.path.join(output_dir, f'{trial_name}_pca_3d_convergence.gif')
            )

            # 3D PCA - all epochs with Procrustes
            visualizer.hidden_layer_visualization(
                pca_epoch='all',
                output_path=os.path.join(output_dir, f'{trial_name}_pca_3d_convergence_procrustes.gif')
            )
        except Exception as e:
            print(f"Warning: visualization failed - {e}")

    def _save_results_summary(self):
        """Save results as CSV and markdown summary."""
        if not self.results:
            print("No results to summarize.")
            return

        # Create summary directory
        summary_dir = self.output_root
        os.makedirs(summary_dir, exist_ok=True)

        # Save detailed CSV
        csv_file = os.path.join(summary_dir, 'results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"✓ Results CSV saved to '{csv_file}'")

        # Save markdown summary
        md_file = os.path.join(summary_dir, 'summary.md')
        with open(md_file, 'w') as f:
            f.write(f"# Experiment Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"## Overview\n")
            f.write(f"- **Configurations:** {len(self.configs)}\n")
            f.write(f"- **Trials per config:** {self.trials}\n")
            f.write(f"- **Total runs:** {len(self.results)}\n")
            f.write(f"- **Results root:** `{self.output_root}`\n\n")

            # Group results by config
            self._write_loss_statistics(f)
            self._write_config_tables(f)
            self._write_directory_structure(f)

        print(f"✓ Markdown summary saved to '{md_file}'")

    def _write_loss_statistics(self, f):
        """Write loss statistics to markdown file."""
        f.write(f"## Loss Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")

        test_losses = [r['final_test_loss'] for r in self.results if r['final_test_loss'] is not None]
        if test_losses:
            f.write(f"| Min Final Test Loss | {min(test_losses):.6f} |\n")
            f.write(f"| Max Final Test Loss | {max(test_losses):.6f} |\n")
            f.write(f"| Mean Final Test Loss | {np.mean(test_losses):.6f} |\n")
            f.write(f"| Std Final Test Loss | {np.std(test_losses):.6f} |\n")

            best_losses = [r['best_test_loss'] for r in self.results if r['best_test_loss'] is not None]
            f.write(f"| Mean Best Test Loss | {np.mean(best_losses):.6f} |\n")

        f.write(f"\n")

    def _write_config_tables(self, f):
        """Write config-specific results tables to markdown file."""
        f.write(f"## Results by Configuration\n\n")

        # Group by config_idx
        configs = {}
        for result in self.results:
            cfg_idx = result['config_idx']
            if cfg_idx not in configs:
                configs[cfg_idx] = []
            configs[cfg_idx].append(result)

        for cfg_idx in sorted(configs.keys()):
            results_for_config = configs[cfg_idx]
            config_name = results_for_config[0]['config_name']

            f.write(f"### Configuration {cfg_idx}: {config_name}\n\n")
            f.write(f"| Trial | Final Train Loss | Final Test Loss | Best Test Loss | Best Epoch | Status |\n")
            f.write(f"|-------|-----------------|-----------------|----------------|------------|--------|\n")

            for result in results_for_config:
                trial_idx = result['trial_idx']
                status = "✓" if result['final_test_loss'] is not None else "✗"
                if result['final_test_loss'] is not None:
                    f.write(f"| {trial_idx} | {result['final_train_loss']:.6f} | {result['final_test_loss']:.6f} | {result['best_test_loss']:.6f} | {result['best_test_epoch']} | {status} |\n")
                else:
                    f.write(f"| {trial_idx} | - | - | - | - | {status} |\n")

            # Summary statistics for this config
            test_losses = [r['final_test_loss'] for r in results_for_config if r['final_test_loss'] is not None]
            if test_losses:
                f.write(f"\n**Summary:** Mean = {np.mean(test_losses):.6f} ± {np.std(test_losses):.6f}\n\n")

    def _write_directory_structure(self, f):
        """Write directory structure info to markdown file."""
        f.write(f"## Output Organization\n\n")
        f.write(f"```\n{self.output_root}/\n")
        f.write(f"├── results.csv                 # Detailed results table\n")
        f.write(f"├── summary.md                  # This file\n")
        for config_idx in range(len(self.configs)):
            f.write(f"├── config_{config_idx}/\n")
            for trial_idx in range(self.trials):
                f.write(f"│   ├── trial_{trial_idx}/\n")
                f.write(f"│   │   ├── training_data.h5      # Training logs and metadata\n")
                f.write(f"│   │   ├── model.pt              # Model weights\n")
                f.write(f"│   │   ├── config.json           # Configuration for this trial\n")
                f.write(f"│   │   └── visualizations/       # Visualization outputs\n")
                f.write(f"│   │       ├── trial_{trial_idx}_loss_history.png\n")
                f.write(f"│   │       ├── trial_{trial_idx}_1d_convergence.gif\n")
                f.write(f"│   │       ├── trial_{trial_idx}_pca_3d_convergence.gif\n")
                f.write(f"│   │       └── trial_{trial_idx}_pca_3d_convergence_procrustes.gif\n")
        f.write(f"```\n\n")

    def _create_configs(self):
        """Create all configuration combinations from independent variables."""
        if not self.ivs:
            self.configs = [self.base_config.copy()]
            return

        iv_vals = self.ivs.values()
        iv_val_combs = product(*iv_vals)
        # Create a list[dict] of changed variables
        iv_combs = [{key: val for key, val in zip(self.ivs.keys(), val_comb)} for val_comb in iv_val_combs]
        self.configs = [self.base_config.copy() | comb for comb in iv_combs]

    def _config_to_name(self, config_idx, config):
        """Create a readable name for a configuration."""
        if not self.ivs:
            return "base"

        # Get values that differ from base config
        diff_parts = []
        for key in self.ivs.keys():
            if key in config:
                value = config[key]
                if isinstance(value, str):
                    diff_parts.append(f"{key}={value}")
                elif isinstance(value, (int, float)):
                    diff_parts.append(f"{key}={value}")
                else:
                    diff_parts.append(f"{key}={value.__class__.__name__}")

        return ", ".join(diff_parts) if diff_parts else f"config_{config_idx}"

def main():
    """
    Example: Run experiment grid with different configurations.

    This example trains models with different epochs settings.
    Adjust base_config and ivs as needed for your experiments.
    """
    import torch.nn as nn
    import torch.optim as optim
    from models import MLP
    from datasets import topksubset

    # Base configuration (common to all trials)
    base_config = {
        'x_range': (-8, 8),
        'data_dim': (10, 1),
        'N': 2048,
        'ground_truth': topksubset(3, 1),
        'model': MLP(input_dim=1, hidden_sizes=(64, 64)),
        'criterion': nn.MSELoss(),
        'optimizer': optim.AdamW(MLP(input_dim=1, hidden_sizes=(64, 64)).parameters(), lr=0.001),
    }

    # Independent variables (configurations to test)
    ivs = {
        'epochs': [100, 500],  # Test different epoch counts
    }

    experiment = Experiment(
        base_config=base_config,
        ivs=ivs,
        trials=2,  # 2 trials per configuration
        output_root='results'
    )

    # Run the complete pipeline: train, visualize, and save results
    experiment.run_grid(visualize=True, save_visualizations=True)

    def plot_grid_final_results(self, df, ivs, dvs):
        """
        Plot the final results of all experiments in the grid

        Args:
            df: Dataframe of final results
            ivs: Independent variables (dict of variable name to list of values).
            dvs: Dependent variables (list of metric names).

        Returns:
            None
        """
        for iv1, iv2 in itertools.permutations(ivs.keys(), 2):
            for dv in dvs:
                cvs = {k: v for k, v in ivs.items() if k != iv1 and k != iv2}
                for cv_dict in dict_product(cvs):
                    cols = list(cv_dict.keys())
                    cv_vals = pd.Series(cv_dict)
                    relevant_data = df[(df[cols] == cv_vals).all(axis=1)]
                    sns.violinplot(data=relevant_data, x=iv1, y=dv, hue=iv2)
                    title = f"{dv} vs {str_leaf(iv1)} and {str_leaf(iv2)}"
                    plt.title(title)
                    text = "\n".join(f"{str_leaf(k)}={v}" for k, v in cv_dict.items())
                    plt.figtext(1.0, 0.01, text, wrap=True, horizontalalignment='right', fontsize=8)
                    self.io.save_plot(plt, f"images/{str_leaf(iv1)} vs {str_leaf(iv2)}"
                                           f"/{title} for {', '.join(f'{str_leaf(k)}={v}' for k, v in cv_dict.items())}.png")

if __name__ == '__main__':
    main()
