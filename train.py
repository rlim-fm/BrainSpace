import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc
import torch.optim as optim
import h5py
import os
import json

from tqdm import trange

from models import *
from datasets import *
from visualization import *


rng = np.random.default_rng(42)
d = 10

DATA_SETTINGS = {
    "x_range": (-8, 8),
    "data_dim": (10, 1),
    "N": 2048,
    "ground_truth": topksubset(3, 1),
}

MODEL = SimpleTransformerModel(input_dim=1)

OPTIMIZER = optim.AdamW(MODEL.parameters(), lr=0.001)
CRITERION = nn.MSELoss()
SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=100, gamma=0.5)

EPOCHS = 1000

class Processor:
    def __init__(self,
                 x_range=(-8, 8),
                 data_dim = (10, 1),
                 N=2048,
                 ground_truth=topksubset(3, 1),
                 model=MLP(input_dim=10),
                 epochs=1000,
                 criterion=nn.MSELoss(),
                 optimizer=None,
                 scheduler=None,
                 *,
                 seed: Optional[int] = None,
                 dtype=torch.float32):
        self._set_environment(seed=seed, dtype=dtype)

        self.x_range = x_range
        x_train = qmc.LatinHypercube(d=math.prod(data_dim), rng=rng).random(x_num).reshape(data_dim)

        # Data setup
        self.x_train = x_train * (x_range[1] - x_range[0]) + x_range[0] # adjust range
        self.y_train = ground_truth(torch.from_numpy(self.x_train).float())
        self.x_test = x_train * 2 # double x_range for test set
        self.y_test = ground_truth(torch.from_numpy(self.x_test).float())

        # Training
        self.model = model.to(self.device, dtype=dtype)
        self.criterion = criterion

        if optimizer:
            self.optimizer = optimizer.to(self.device, dtype=dtype)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        if scheduler:
            self.scheduler = scheduler.to(self.device, dtype=dtype)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.epochs = epochs
        # Record metrics

        # Logging
        self.logs = {
            "train_loss": [],
            "test_loss": [],
            "y_test_preds": [],
            "hidden_states": [],
        }
        self.metadata = {
            "x_range": x_range,
            "data_dim": data_dim,
            "N": N,
            "ground_truth_fn": ground_truth_fn.__name__,
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "epochs": epochs,
        }

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        hidden_train = self.model(self.x_train)
        out_train = self.model.output_layer(hidden_train)
        train_loss = self.criterion(out_train, self.y_train)
        self.logs['train_loss'].append(train_loss.item())
        train_loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            hidden_test = self.model(self.x_test)
            out_test = self.model.output_layer(hidden_test).squeeze()
            test_loss = self.criterion(out_test, self.y_test)
            self.logs['test_loss'].append(test_loss.item())
            self.logs['y_test_preds'].append(out_test.cpu().numpy())
            self.logs['hidden_states'].append(hidden_test.cpu().numpy())


    def run(self):
        for _ in trange(self.epochs, desc="Training"):
            self.train_epoch()
            self.test_epoch()
            if self.scheduler:
                self.scheduler.step()

        print(f"\nTraining complete!")
        print(f"Final losses: Train Loss: {self.logs['train_loss'][-1]:.6f}, Test Loss: {self.logs['test_loss'][-1]:.6f}")

    def save(self, filename):
        with h5py.File(filename, 'w') as hf:
            # Create groups for organization
            training_group = hf.create_group('training')
            test_group = hf.create_group('test')
            metadata_group = hf.create_group('metadata')

            # Save loss histories
            training_group.create_dataset('train_loss', data=np.array(self.logs['train_loss']))
            training_group.create_dataset('test_loss', data=np.array(self.logs['test_loss']))

            # Save test predictions (shape: [epochs, n_test_samples, 1])
            test_preds_array = np.array(self.logs['y_test_preds'])
            test_group.create_dataset('predictions', data=test_preds_array)

            # Save test hidden states (shape: [epochs, n_test_samples, hidden_dim])
            test_hidden_array = np.array(test_hidden_states)
            test_group.create_dataset('hidden_states', data=test_hidden_array)

            # Save test inputs and targets
            test_group.create_dataset('x_test', data=x_test.numpy())
            test_group.create_dataset('y_test', data=y_test.numpy())

            # Save train inputs and targets
            training_group.create_dataset('x_train', data=x_train.numpy())
            training_group.create_dataset('y_train', data=y_train.numpy())

            # Save metadata
            metadata_group.attrs['epochs'] = epochs
            metadata_group.attrs['n_train_samples'] = len(x_train)
            metadata_group.attrs['n_test_samples'] = len(x_test)
            metadata_group.attrs['hidden_dim'] = test_hidden_array.shape[2]
            metadata_group.attrs['final_train_loss'] = train_loss_history[-1]
            metadata_group.attrs['final_test_loss'] = test_loss_history[-1]
            metadata_group.attrs['input_dim'] = x_train.shape[1]
            metadata_group.attrs['ground_truth_func'] = ground_truth.__name__
            metadata_group.attrs['gt_params'] = gt_params

            # Save ground truth function info
            metadata_group.attrs['ground_truth_func'] = 'topksubset'
            metadata_group.attrs['ground_truth_k'] = 3
            metadata_group.attrs['ground_truth_dim'] = 1

        print(f"\nTraining data saved to '{filename}'")

    """HELPER FUNCTIONS"""
    def _set_environment(self, *, dtype=torch.float32, seed: Optional[int] = None):
        """
        Set random seeds and device for reproducibility and performance.
        Args:
            dtype: Data type to use for model parameters and computations.
            seed: Optional random seed for reproducibility. If None, a random seed will be generated.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed if seed is not None else np.random.randint(2 ** 32 - 1)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.dtype = dtype

    def reset(self):
        self._set_environment(seed=self.seed, dtype=self.dtype)
        self.data = {}


# Display summary
print("\n" + "="*60)
print("HDF5 File Structure:")
print("="*60)
print(f"  ├── training/")
print(f"  │   ├── train_loss (shape: {np.array(train_loss_history).shape})")
print(f"  │   ├── test_loss (shape: {np.array(test_loss_history).shape})")
print(f"  │   ├── x_train (shape: {x_train.numpy().shape})")
print(f"  │   └── y_train (shape: {y_train.numpy().shape})")
print(f"  ├── test/")
print(f"  │   ├── predictions (shape: {test_preds_array.shape})")
print(f"  │   ├── hidden_states (shape: {test_hidden_array.shape})")
print(f"  │   ├── x_test (shape: {x_test.numpy().shape})")
print(f"  │   └── y_test (shape: {y_test.numpy().shape})")
print(f"  ├── function (ground_truth) - topksubset(3, dim=1)")
print(f"  └── metadata/")
print(f"      ├── epochs: {epochs}")
print(f"      ├── n_train_samples: {len(x_train)}")
print(f"      ├── n_test_samples: {len(x_test)}")
print(f"      ├── hidden_dim: {test_hidden_array.shape[2]}")
print(f"      ├── input_dim: {x_train.shape[1]}")
print(f"      ├── final_train_loss: {train_loss_history[-1]:.6f}")
print(f"      ├── final_test_loss: {test_loss_history[-1]:.6f}")
print(f"      ├── ground_truth_func: {ground_truth.__name__}, params: {gt_params}")
print(f"      └── captured_epochs (shape: {np.array(captured_epochs).shape})")
print("="*60)
print(f"\nAdditional files saved:")
print(f"  - {model_filename} (model weights)")

# if __name__=="__main__":
