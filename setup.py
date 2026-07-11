from setuptools import setup

setup(
    name="brainspace",
    version="0.2.0",
    description="Core framework for training, visualizing, and experimenting on neural networks for functional regression",
    author="Richard Lim",
    packages=["brainspace", "brainspace.internal"],
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "tqdm",
        "matplotlib",
        "h5py",
        "scikit-learn",
        "pyyaml",
        "pandas",
        "statsmodels",
    ],
    extras_require={
        "gpu-plotting": [
            "fastplotlib>=0.3.0",
        ],
        "posthocs": [
            "scikit-posthocs",
        ],
    },
    entry_points={
        "console_scripts": [
            "brainspace-analyze=brainspace.analyze:main",
            "brainspace-view=brainspace.view:main",
            "brainspace-refresh=brainspace.refresh_results:main",
            "brainspace-migrate=brainspace.migrate_registry:main",
        ],
    },
    python_requires=">=3.10",
)
