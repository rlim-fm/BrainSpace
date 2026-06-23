import warnings
from typing import Optional

import h5py
import numpy as np
from scipy.stats import qmc
import torch.optim as optim

from tqdm import trange

from models import *
from datasets import *

class Processor:
    def __init__(self,
                 x_range: tuple=(-8, 8),
                 data_dim: tuple=(10, 1),
                 N: int=2048,
                 ground_truth=None,
                 model=None,
                 epochs: int=3000,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 visualizer=None,
                 *,
                 seed: Optional[int] = None,
                 dtype=torch.float32,
                 device=None):
        """
        Args:
            x_range: tuple specifying the range of input values (min, max).
            data_dim: tuple specifying the shape of input logs (e.g., (10, 1) for 10 samples of 1D input).
            N: number of samples.
            ground_truth: the ground truth function
            model: the model to train. If None, defaults to a simple MLP.
            epochs: number of epochs.
            criterion: the loss function to use for training.
            optimizer: the optimizer to use for training.
            scheduler: the scheduler to use for training.
            visualizer: optional Visualizer instance for streaming logging during training.
            seed (Optional[int]): random seed for reproducibility. If None, a random seed will be generated.
            dtype: logs type for model parameters and computations (default: torch.float32).
        """
        self._set_environment(seed=seed, dtype=dtype, device=torch.device(device))

        self.x_range = x_range
        self.data_dim = data_dim
        self.d = math.prod(data_dim) # true dimensionality of the data
        self.N = N
        self.visualizer = visualizer

        # Set defaults
        if ground_truth is None:
            ground_truth = topksubset(3, 1)
        if model is None:
            model = SimpleTransformerModel(input_dim=1)
        if criterion is None:
            criterion = nn.MSELoss()

        # Data setup
        x_train = qmc.LatinHypercube(d=self.d, rng=self.rng).random(N).reshape(N, *data_dim)
        x_train = x_train * (x_range[1] - x_range[0]) + x_range[0]
        self.x_train = torch.from_numpy(x_train).float().to(self.device)
        self.y_train = ground_truth(self.x_train).to(self.device).squeeze()

        lin = np.linspace(x_range[0], x_range[1], self.N // self.d) # (N // d,)
        x_axis = (np.eye(self.d)[None, ...] * lin[:, None, None]).reshape(-1, self.d) # (1, d, d) * (N // d, 1, 1) =  (N, d, d) -> reshape to (~N, d)
        x_axis = x_axis.reshape(-1, *self.data_dim) # (~N, *dims)
        x_test = np.vstack((x_train, x_axis, np.zeros((1, *self.data_dim)))) * 5 # (~2N + 1, *dims)
        self.x_test = torch.from_numpy(x_test).float().to(self.device)
        self.y_test = ground_truth(self.x_test).to(self.device).squeeze()

        # Training
        self.model = model.to(self.device)
        self.criterion = criterion

        if not optimizer:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

        # Logging setup
        self.logs = {
            "x_train": self.x_train.cpu().numpy().reshape(-1, self.d),
            "y_train": self.y_train.cpu().numpy().squeeze(),
            "x_test": self.x_test.cpu().numpy().reshape(-1, self.d),
            "y_test": self.y_test.cpu().numpy().squeeze(),
            "train_loss": [],
            "test_loss": [],
            "f_test": [],
            "hidden_states": [],
        }
        self.metadata = {
            "x_range": x_range,
            "data_dim": data_dim,
            "N": N,
            "ground_truth": ground_truth.__class__.__name__,
            "model": model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            "epochs": epochs,
        }

        # Configure visualizer if provided
        if visualizer is not None:
            visualizer.attach_processor(self)

    @staticmethod
    def from_config(config: dict):
        """Initialize processor from config dictionary."""
        x_range = config.get('x_range', (-8, 8)),
        data_dim = config.get('data_dim', (10, 1)),
        N = config.get('N', 2048),
        ground_truth = config.get('ground_truth', topksubset(3, 1)),
        model = config.get('model', MLP(input_dim=1)),
        epochs = config.get('epochs', 1000),
        criterion = config.get('criterion', nn.MSELoss()),
        optimizer = config.get('optimizer', optim.AdamW(model.parameters(), lr=0.001)),
        scheduler = config.get('scheduler', None),
        seed = config.get('seed', None),
        dtype = config.get('dtype', torch.float32)
        return Processor(
            x_range=x_range,
            data_dim=data_dim,
            N=N,
            ground_truth=ground_truth,
            model=model,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
            dtype=dtype
        )

    @staticmethod
    def from_h5(filename: str):
        """Initialize processor from HDF5 file."""
        with h5py.File(filename, 'r') as hf:
            logs = {
                "train_loss": hf['logs/train_loss'][:],
                "test_loss": hf['logs/test_loss'][:],
                "f_test": hf['logs/f_test'][:],
                "hidden_states": hf['logs/hidden_states'][:],
            }
            metadata = {
                "x_range": tuple(hf['metadata/x_range'][:]),
                "data_dim": tuple(hf['metadata/data_dim'][:]),
                "N": int(hf['metadata/N'][()]),
                "ground_truth": hf['metadata/ground_truth'][()].decode('utf-8'),
                "model": hf['metadata/model'][()].decode('utf-8'),
                "optimizer": hf['metadata/optimizer'][()].decode('utf-8'),
                "criterion": hf['metadata/criterion'][()].decode('utf-8'),
                "scheduler": hf['metadata/scheduler'][()].decode('utf-8') if 'scheduler' in hf['metadata'] else None,
                "epochs": int(hf['metadata/epochs'][()]),
            }
        processor = Processor.from_config(metadata)
        processor.logs = logs

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out, hidden = self.model(self.x_train)
        out = out.squeeze()
        train_loss = self.criterion(out, self.y_train)
        self.logs['train_loss'].append(train_loss.item())
        train_loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            out, hidden = self.model(self.x_test)
            out = out.squeeze()
            test_loss = self.criterion(out, self.y_test)
            self.logs['test_loss'].append(test_loss.item())

            # Always log these (needed for saving)
            f_test_np = out.cpu().numpy()
            self.logs['f_test'].append(f_test_np)
            hidden_np = hidden.cpu().numpy()
            self.logs['hidden_states'].append(hidden_np)

            # Update visualizer with current epoch
            # Visualizer dynamically extracts what it needs from processor
            if self.visualizer is not None:
                current_epoch = len(self.logs['test_loss']) - 1
                self.visualizer.update(current_epoch)


    def run(self):
        for _ in trange(self.epochs, desc=f"Training"):
            self.train_epoch()
            self.test_epoch()
            if self.scheduler:
                self.scheduler.step()

        # convert to np arrays
        for key, val in self.logs.items():
            self.logs[key] = np.array(val)

        print(f"\nTraining complete!")
        print(f"Final losses: Train Loss: {self.logs['train_loss'][-1]:.6f}, Test Loss: {self.logs['test_loss'][-1]:.6f}")

        # Finalize visualizations if visualizer is attached
        if self.visualizer is not None:
            self.visualizer.finalize()

    def save(self, filename='train_out/training_data.h5', output_dir='train_out', *, metadata_only=False):
        to_save = {'metadata': self.metadata}
        if not metadata_only:
            to_save['logs'] = self.logs
        def dict_to_h5(dic, h5_file, path='/'):
            """Recursively saves a dictionary to an HDF5 file."""
            for key, value in dic.items():
                dataset_path = f"{path}{key}"

                if isinstance(value, dict):
                    # Create a Group for nested dictionaries
                    h5_file.create_group(dataset_path)
                    dict_to_h5(value, h5_file, dataset_path + '/')
                elif isinstance(value, str):
                    # Handle strings: encode as bytes
                    h5_file.create_dataset(dataset_path, data=value.encode('utf-8'))
                elif isinstance(value, (int, float, bool)):
                    # Handle scalars
                    h5_file.create_dataset(dataset_path, data=value)
                else:
                    # Handle arrays and other types
                    try:
                        h5_file.create_dataset(dataset_path, data=np.array(value))
                    except (TypeError, ValueError):
                        # If conversion fails, try as bytes string
                        warnings.warn(f"Could not save {key} as array. Saving as string instead.")
                        h5_file.create_dataset(dataset_path, data=str(value).encode('utf-8'))

        os.makedirs(output_dir, exist_ok=True)
        with h5py.File(filename, 'w') as hf:
            dict_to_h5(to_save, hf)

        # Save model weights
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

    def print_summary(self):
        """Print training summary from logs."""
        preds_shape = np.array(self.logs['f_test']).shape
        hidden_shape = np.array(self.logs['hidden_states']).shape

        print("\n" + "="*75)
        print("TRAINING SUMMARY (from logs)")
        print("="*75)
        print(f"\n[Configuration]")
        print(f"  Epochs: {self.epochs}")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"\n[Data]")
        print(f"  Train samples: {len(self.x_train)}")
        print(f"  Test samples: {len(self.x_test)}")
        print(f"  Input shape: {self.x_train.shape}")
        print(f"\n[Loss Statistics]")
        print(f"  Final train loss: {self.logs['train_loss'][-1]:.6f}")
        print(f"  Final test loss:  {self.logs['test_loss'][-1]:.6f}")
        print(f"  Best train loss:  {np.min(self.logs['train_loss']):.6f} (epoch {np.argmin(self.logs['train_loss'])})")
        print(f"  Best test loss:   {np.min(self.logs['test_loss']):.6f} (epoch {np.argmin(self.logs['test_loss'])})")
        print(f"\n[HDF5 File Structure]")
        print(f"  ├── logs/")
        print(f"  │   ├── x_train: {self.logs['x_train'].shape}")
        print(f"  │   ├── y_train: {self.logs['y_train'].shape}")
        print(f"  │   ├── x_test: {self.logs['x_test'].shape}")
        print(f"  │   ├── y_test: {self.logs['y_test'].shape}")
        print(f"  │   ├── train_loss: {len(self.logs['train_loss'])} entries")
        print(f"  │   ├── test_loss: {len(self.logs['test_loss'])} entries")
        print(f"  │   ├── f_test: {preds_shape} entries")
        print(f"  │   └── hidden_states: {hidden_shape} entries")
        print(f"  └── metadata/")
        print(f"      ├── x_range: {self.metadata['x_range']}")
        print(f"      ├── data_dim: {self.metadata['data_dim']}")
        print(f"      ├── N: {self.metadata['N']}")
        print(f"      ├── ground_truth: {self.metadata['ground_truth']}")
        print(f"      ├── model: {self.metadata['model']}")
        print(f"      ├── optimizer: {self.metadata['optimizer']}")
        print(f"      ├── criterion: {self.metadata['criterion']}")
        print(f"      ├── scheduler: {self.metadata['scheduler']}")
        print(f"      └── epochs: {self.metadata['epochs']}")
        print("="*75 + "\n")

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    def _set_environment(self, *, dtype=torch.float32, seed: Optional[int] = None, device: Optional[torch.device] = None):
        """
        Set random seeds and device for reproducibility and performance.
        Args:
            dtype: Data type to use for model parameters and computations.
            seed: Optional random seed for reproducibility. If None, a random seed will be generated.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.seed = seed if seed is not None else 42
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.dtype = dtype

    def reset(self):
        self._set_environment(seed=self.seed, dtype=self.dtype)
        self.logs = {
            "train_loss": [],
            "test_loss": [],
            "f_test": [],
            "hidden_states": [],
        }


def main():
    """Main training script using OOP Processor."""
    model = MLP(input_dim=10, dropout=0.1)
    optimizer = optim.AdamW(model.parameters(),lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=750, gamma=0.8)
    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10,),
        N=2048,
        ground_truth=topksubset(3, 1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=5000,
        criterion=nn.MSELoss(reduction='mean'),
        seed=42
    )

    processor.run()
    processor.print_summary()
    processor.save('train_out/MLP_training_data.h5', output_dir='train_out')

if __name__ == '__main__':
    main()
