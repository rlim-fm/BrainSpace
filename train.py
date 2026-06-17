
from scipy.stats import qmc
import torch.optim as optim

from tqdm import trange

from models import *
from datasets import *
from visualization import *

class Processor:
    def __init__(self,
                 x_range: tuple=(-8, 8),
                 data_dim: tuple=(10, 1),
                 N: int=2048,
                 ground_truth=None,
                 model=None,
                 epochs: int=1000,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 *,
                 seed: Optional[int] = None,
                 dtype=torch.float32):
        """
        Args:
            x_range: tuple specifying the range of input values (min, max).
            data_dim: tuple specifying the shape of input data (e.g., (10, 1) for 10 samples of 1D input).
            N: number of samples.
            ground_truth: the ground truth function
            model: the model to train. If None, defaults to a simple MLP.
            epochs: number of epochs.
            criterion: the loss function to use for training.
            optimizer: the optimizer to use for training.
            scheduler: the scheduler to use for training.
            seed (Optional[int]): random seed for reproducibility. If None, a random seed will be generated.
            dtype: data type for model parameters and computations (default: torch.float32).
        """
        self._set_environment(seed=seed, dtype=dtype)

        self.x_range = x_range
        self.data_dim = data_dim
        self.N = N

        # Set defaults
        if ground_truth is None:
            ground_truth = topksubset(3, 1)
        if model is None:
            model = SimpleTransformerModel(input_dim=1)
        if criterion is None:
            criterion = nn.MSELoss()

        x_train = qmc.LatinHypercube(d=math.prod(data_dim), rng=self.rng).random(N).reshape(N, *data_dim)

        # Data setup
        self.x_train = torch.from_numpy(x_train * (x_range[1] - x_range[0]) + x_range[0]).float().to(self.device)
        self.y_train = ground_truth(self.x_train).to(self.device)
        x_test = x_train * 2  # double x_range for test set
        self.x_test = torch.from_numpy(x_test).float().to(self.device)
        self.y_test = ground_truth(self.x_test).to(self.device)

        # Training
        self.model = model.to(self.device)
        self.criterion = criterion

        if not optimizer:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

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
            "ground_truth": ground_truth.__name__,
            "model": model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            "epochs": epochs,
        }

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
                "y_test_preds": hf['logs/y_test_preds'][:],
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

    def save(self, filename='out/training_data.h5', output_dir='out'):
        to_save = {
            'logs': self.logs,
            'metadata': self.metadata,
        }
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
                        h5_file.create_dataset(dataset_path, data=str(value).encode('utf-8'))

        os.makedirs(output_dir, exist_ok=True)
        with h5py.File(filename, 'w') as hf:
            dict_to_h5(to_save, hf)

        # Save model weights
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

    def print_summary(self):
        """Print training summary from logged data."""
        preds_shape = np.array(self.logs['y_test_preds']).shape
        hidden_shape = np.array(self.logs['hidden_states']).shape

        print("\n" + "="*75)
        print("TRAINING SUMMARY (from logged data)")
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
        print(f"  ├── training/")
        print(f"  │   ├── train_loss (shape: {(len(self.logs['train_loss']),)})")
        print(f"  │   ├── test_loss (shape: {(len(self.logs['test_loss']),)})")
        print(f"  │   ├── x_train (shape: {self.x_train.shape})")
        print(f"  │   └── y_train (shape: {self.y_train.shape})")
        print(f"  ├── test/")
        print(f"  │   ├── predictions (shape: {preds_shape})")
        print(f"  │   ├── hidden_states (shape: {hidden_shape})")
        print(f"  │   ├── x_test (shape: {self.x_test.shape})")
        print(f"  │   └── y_test (shape: {self.y_test.shape})")
        print(f"  └── metadata/")
        print(f"      ├── final_train_loss: {self.logs['train_loss'][-1]:.6f}")
        print(f"      ├── final_test_loss: {self.logs['test_loss'][-1]:.6f}")
        print(f"      └── ground_truth: topksubset(k=3, dim=1)")
        print("="*75 + "\n")

    """HELPER FUNCTIONS"""
    def _set_environment(self, *, dtype=torch.float32, seed: Optional[int] = None):
        """
        Set random seeds and device for reproducibility and performance.
        Args:
            dtype: Data type to use for model parameters and computations.
            seed: Optional random seed for reproducibility. If None, a random seed will be generated.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "y_test_preds": [],
            "hidden_states": [],
        }


def main():
    """Main training script using OOP Processor."""
    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10, 1),
        N=2048,
        ground_truth=topksubset(3, 1),
        model=SimpleTransformerModel(input_dim=1),
        epochs=1000,
        criterion=nn.MSELoss(reduction='mean'),
        seed=42
    )

    processor.run()
    processor.print_summary()
    processor.save('out/attn_training_data.h5', output_dir='out')

if __name__ == '__main__':
    main()
