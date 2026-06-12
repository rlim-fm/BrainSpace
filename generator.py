import numpy as np
import os
import json

rng = np.random.default_rng(42)

def generate_knapsack(samples=2048, domain=(0, 100), N=3, output_dir="data/knapsack"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(samples):
        weights = rng.integers(domain[0], domain[1], size=N).tolist()
        values = rng.integers(domain[0], domain[1], size=N).tolist()
        capacity = rng.integers(domain[0], domain[1])
        instance = {
            "weights": weights,
            "values": values,
            "capacity": capacity
        }
        with open(os.path.join(output_dir, f"instance_{i}.json"), "w") as f:
            json.dump(instance, f)

if __name__ == "__main__":
    generate_knapsack()
