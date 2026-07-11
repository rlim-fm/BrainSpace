"""BrainSpace: a framework for training, visualizing, and systematically
experimenting on neural networks for functional regression.

Domain packages (e.g. tropical architectures, cluster-sampling research) build
on this core by subclassing the config dataclasses, registering their classes
in :mod:`brainspace.config`'s registries, and hooking Processor/Visualization
extension points. See docs/EXPERIMENT_GUIDE.md.
"""

__version__ = "0.2.0"
