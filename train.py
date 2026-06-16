import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc
import torch.optim as optim
import h5py
import os
import json
from models import *
from datasets import *

rng = np.random.default_rng(42)
d = 10

x_full = qmc.LatinHypercube(d=d, rng=rng).random(2048) * 16 - 8 # [B, d]
x_full = np.expand_dims(x_full, -1) # [B, d, 1]
ground_truth = topksubset(3, dim=1)
y_full = ground_truth(torch.from_numpy(x_full).float())  # [B, 1]

# 50/50 train/test split
x_train = torch.from_numpy(x_full[:1024]).float()
y_train = y_full[:1024]
x_test = torch.from_numpy(x_full[1024:]).float()
y_test = y_full[1024:]

model = SimpleTransformerModel(input_dim=1)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

epochs = 1000
train_loss_history = []
test_loss_history = []
test_y_preds = []
test_hidden_states = []
captured_epochs = []

for epoch in range(epochs):
    # Training step
    optimizer.zero_grad()
    hidden_train = model(x_train)
    out_train = model.output_layer(hidden_train)
    train_loss = criterion(out_train, y_train)
    train_loss.backward()
    optimizer.step()
    scheduler.step()

    # Evaluation step on test set
    with torch.no_grad():
        hidden_test = model(x_test)
        out_test = model.output_layer(hidden_test).squeeze()
        test_loss = criterion(out_test, y_test)

    # Record metrics
    train_loss_history.append(train_loss.item())
    test_loss_history.append(test_loss.item())
    test_y_preds.append(out_test.detach().cpu().numpy())
    test_hidden_states.append(hidden_test.detach().cpu().numpy())
    captured_epochs.append(epoch)

    # Progress logging
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

print(f"\nTraining complete!")
print(f"Final Train Loss: {train_loss_history[-1]:.6f}")
print(f"Final Test Loss: {test_loss_history[-1]:.6f}")

# Save the trained model
os.makedirs('out', exist_ok=True)
model_filename = 'out/model.pt'
torch.save(model.state_dict(), model_filename)
print(f"\nModel saved to '{model_filename}'")

# Save the ground truth function metadata
# Since the ground_truth is a lambda from topksubset(3, dim=1), we save its configuration
ground_truth_config = {
    'function_type': 'topksubset',
    'k': 3,
    'dim': 1
}
gt_config_filename = 'out/ground_truth_config.json'
with open(gt_config_filename, 'w') as f:
    json.dump(ground_truth_config, f, indent=2)
print(f"Ground truth function config saved to '{gt_config_filename}'")

# Save all metrics and predictions to HDF5
h5_filename = 'out/attn_training_data.h5'

with h5py.File(h5_filename, 'w') as hf:
    # Create groups for organization
    training_group = hf.create_group('training')
    test_group = hf.create_group('test')
    metadata_group = hf.create_group('metadata')

    # Save loss histories
    training_group.create_dataset('train_loss', data=np.array(train_loss_history))
    training_group.create_dataset('test_loss', data=np.array(test_loss_history))

    # Save test predictions (shape: [epochs, n_test_samples, 1])
    test_preds_array = np.array(test_y_preds)
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

    # Save ground truth function info
    metadata_group.attrs['ground_truth_func'] = 'topksubset'
    metadata_group.attrs['ground_truth_k'] = 3
    metadata_group.attrs['ground_truth_dim'] = 1

    # Save epoch numbers
    metadata_group.create_dataset('captured_epochs', data=np.array(captured_epochs))

print(f"\nTraining data saved to '{h5_filename}'")

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
print(f"      ├── ground_truth_func: topksubset")
print(f"      ├── ground_truth_k: 3")
print(f"      ├── ground_truth_dim: 1")
print(f"      └── captured_epochs (shape: {np.array(captured_epochs).shape})")
print("="*60)
print(f"\nAdditional files saved:")
print(f"  - {model_filename} (model weights)")
print(f"  - {gt_config_filename} (ground truth function config)")
