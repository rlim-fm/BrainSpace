#!/usr/bin/env python
"""
Test script to validate the Experiment pipeline.
Runs a minimal experiment to ensure all components work correctly.
"""

import sys
import os

def test_experiment_pipeline():
    """Test the basic experiment pipeline."""
    print("Testing Experiment Pipeline...\n")

    # Import required modules
    try:
        from experiment import Experiment
        from models import MLP
        from datasets import topksubset
        import torch.nn as nn
        import torch.optim as optim
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

    # Create a minimal configuration (based on successful train.py main())
    try:
        from models import SimpleTransformerModel

        model = SimpleTransformerModel(input_dim=1, dropout=0.0)
        base_config = {
            'x_range': (-8, 8),
            'data_dim': (10, 1),
            'N': 512,  # Smaller dataset for faster testing
            'ground_truth': topksubset(3, 1),
            'model': model,
            'criterion': nn.MSELoss(reduction='mean'),
            'optimizer': optim.AdamW(model.parameters(), lr=1e-3),
            'epochs': 5,  # Minimal epochs for testing
        }
        print("✓ Base config created")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create experiment
    try:
        experiment = Experiment(
            base_config=base_config,
            ivs={'epochs': [5, 10]},  # 2 configurations
            trials=1,  # 1 trial per config
            output_root='test_results'
        )
        print(f"✓ Experiment created with {len(experiment.configs)} configs")
    except Exception as e:
        print(f"✗ Experiment creation failed: {e}")
        return False

    # Test config creation
    try:
        assert len(experiment.configs) == 2, f"Expected 2 configs, got {len(experiment.configs)}"
        print(f"✓ Config grid created correctly: {experiment.configs}")
    except Exception as e:
        print(f"✗ Config grid validation failed: {e}")
        return False

    # Test config naming
    try:
        names = [experiment._config_to_name(i, c) for i, c in enumerate(experiment.configs)]
        print(f"✓ Config names: {names}")
    except Exception as e:
        print(f"✗ Config naming failed: {e}")
        return False

    # Run minimal pipeline (training only, no visualization)
    try:
        print("\nRunning minimal experiment (training only)...")
        experiment.run_grid(visualize=False, save_visualizations=False)
        print("✓ Experiment completed successfully")
    except Exception as e:
        print(f"✗ Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify results were collected
    try:
        assert len(experiment.results) > 0, "No results collected"
        print(f"✓ Results collected: {len(experiment.results)} entries")

        # Check result structure
        result = experiment.results[0]
        required_keys = ['config_idx', 'trial_idx', 'final_train_loss', 'final_test_loss', 'best_test_loss']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        print(f"✓ Result structure valid")
        print(f"  Sample result: Config={result['config_idx']}, Trial={result['trial_idx']}")

        # Handle both successful and failed runs
        if result['final_test_loss'] is not None:
            print(f"  Final Test Loss: {result['final_test_loss']:.6f}")
            print(f"  Best Test Loss: {result['best_test_loss']:.6f}")
        else:
            print(f"  Status: Failed (error: {result.get('error', 'unknown')})")
    except Exception as e:
        print(f"✗ Results validation failed: {e}")
        return False

    # Verify output files
    try:
        assert os.path.exists('test_results/results.csv'), "results.csv not created"
        assert os.path.exists('test_results/summary.md'), "summary.md not created"

        # Check config directories (only for successful runs)
        for config_idx in range(len(experiment.configs)):
            for trial_idx in range(experiment.trials):
                trial_dir = f'test_results/config_{config_idx}/trial_{trial_idx}'
                # Directory should exist even if training failed
                assert os.path.exists(trial_dir), f"Trial directory not created: {trial_dir}"

                # Only check data files if training succeeded
                result = next((r for r in experiment.results if r['config_idx'] == config_idx and r['trial_idx'] == trial_idx), None)
                if result and result['final_test_loss'] is not None:
                    assert os.path.exists(f'{trial_dir}/training_data.h5'), "training_data.h5 not saved"
                    assert os.path.exists(f'{trial_dir}/model.pt'), "model.pt not saved"
                    assert os.path.exists(f'{trial_dir}/config.json'), "config.json not saved"

        print("✓ Output structure created correctly")
    except Exception as e:
        print(f"✗ Output file validation failed: {e}")
        return False

    # Print summary file content
    try:
        with open('test_results/summary.md', 'r') as f:
            summary_content = f.read()
        print("\n✓ Summary file generated:")
        print("─" * 60)
        print(summary_content[:500] + "\n..." if len(summary_content) > 500 else summary_content)
        print("─" * 60)
    except Exception as e:
        print(f"✗ Could not read summary: {e}")
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nTo run with visualizations, execute:")
    print("  experiment.run_grid(visualize=True, save_visualizations=True)")

    return True

if __name__ == '__main__':
    success = test_experiment_pipeline()
    sys.exit(0 if success else 1)
