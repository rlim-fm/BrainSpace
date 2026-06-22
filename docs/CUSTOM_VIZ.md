# API Reference - Creating Custom Visualizations

The visualization system is designed for simplicity: **just write a class with `update()` and `finalize()` methods**.

## Simple Example

Here's the simplest possible custom visualization:

```python
from visualization import Visualization

class MySimpleViz(Visualization):
    def __init__(self):
        super().__init__('my_viz')
        self.count = 0
    
    def update(self, processor, epoch):
        """Called each epoch during training."""
        self.count += 1
    
    def finalize(self, output_dir, prefix):
        """Called after training."""
        print(f"Processed {self.count} epochs")

# Use it
visualizer = Visualizer()
visualizer.register(MySimpleViz())
```

## Base Class: Visualization

```python
from visualization import Visualization

class Visualization(ABC):
    def __init__(self, name: str, sampling: int = 1):
        self.name = name
        self.sampling = sampling
        self.frames = []
    
    @abstractmethod
    def update(self, processor, epoch: int):
        """Called each epoch during training."""
        pass
    
    @abstractmethod
    def finalize(self, output_dir: str, prefix: str):
        """Called after training."""
        pass
```

## Built-in Visualizations

```python
visualizer.register_loss_history()
visualizer.register_convergence_1d(axis=0)
visualizer.register_pca_3d(pca_epoch=-1)
visualizer.register_pca_3d_procrustes()
visualizer.register_function_space()
```

## Data Available in update()

```python
def update(self, processor, epoch):
    # Access logs
    f_test = processor.logs['f_test'][-1]              # [N,]
    hidden = processor.logs['hidden_states'][-1]       # [N, hidden_dim]
    loss = processor.logs['test_loss'][-1]             # float
    
    # Access metadata
    x_range = processor.metadata['x_range']
    epochs_total = processor.metadata['epochs']
    
    # Access model
    model = processor.model
```

## Example: Custom Metrics

```python
from visualization import Visualization

class L2ErrorMetric(Visualization):
    def __init__(self):
        super().__init__('l2_error')
        self.errors = []
    
    def update(self, processor, epoch):
        f_test = processor.logs['f_test'][-1]
        y_test = processor.logs['y_test']
        l2_error = ((f_test - y_test) ** 2).mean()
        self.errors.append(l2_error)
    
    def finalize(self, output_dir, prefix):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.semilogy(self.errors)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L2 Error')
        plt.savefig(f'{output_dir}/{prefix}_l2_error.png')
        plt.close()

visualizer.register(L2ErrorMetric())
```

## Usage

```python
from visualization import Visualizer
from train import Processor

visualizer = Visualizer(sampling=10)
visualizer.register_loss_history()
visualizer.register(MyCustomViz())

processor = Processor(..., visualizer=visualizer)
processor.run()  # update() and finalize() called automatically
```
