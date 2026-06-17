# Two-Mode PCA 3D Animation Implementation

## Overview
Updated `create_pca_3d_animation()` in `visualization.py` to support two visualization modes for analyzing hidden state evolution in the 3D PCA space.

## Mode Descriptions

### Mode 1: Default (Numeric) - Static Manifold
**Usage**: `pca_epoch=-1` (default), or any specific epoch number

**Behavior**:
1. **Fit single PCA** on the anchor epoch (default: last epoch, index -1)
2. **Transform all hidden states** using this fixed PCA basis
3. **Create static ground truth manifold** from the anchor epoch's PC1/PC2 coordinates
4. **Animate predictions** through all epochs showing how they move relative to the fixed reference frame

**Visual Effect**:
- Ground truth surface stays in place
- Network predictions sweep through the space, showing convergence toward the truth
- Easy to see how predictions move relative to a fixed reference frame

**Example**:
```python
create_pca_3d_animation(
    y_test, predictions, hidden_states,
    pca_epoch=-1,  # Use last epoch for PCA fitting
    output_path='static_manifold.gif'
)
```

---

### Mode 2: All Epochs + Procrustes - Evolving Manifold
**Usage**: `pca_epoch='all'`

**Behavior**:
1. **Fit separate PCA** on each individual epoch independently
2. **Apply Procrustes alignment** between sequential epochs to smooth transitions
   - Aligns each epoch's 2D representation to the previous epoch
   - Finds optimal rotation/reflection to minimize coordinate jumps
3. **Create evolving ground truth manifold** that changes per frame
4. **Animate predictions** along with the evolving manifold reference

**Procrustes Alignment Details**:
- Uses `scipy.spatial.procrustes(A, B)` which returns: `(U, V, disparity)`
  - `U`: transformed first array (reference)
  - `V`: transformed second array (aligned to reference)
  - `disparity`: reconstruction error
- We use `V` (aligned coordinates) to smoothly transition between consecutive epochs
- This prevents sudden "jumps" when PCA basis changes between frames

**Visual Effect**:
- Ground truth surface evolves smoothly across frames
- Predictions also evolve smoothly
- Shows how both the hidden representation geometry AND the model's predictions develop together
- Better for understanding manifold evolution during training

**Example**:
```python
create_pca_3d_animation(
    y_test, predictions, hidden_states,
    pca_epoch='all',  # Fit PCA per epoch with Procrustes smoothing
    output_path='evolving_manifold.gif'
)
```

---

## Implementation Details

### Key Changes in Code

**1. Mode Detection**:
```python
if pca_epoch == "all":
    # Procrustes mode
else:
    # Static manifold mode
```

**2. Procrustes Alignment Loop**:
```python
for epoch_idx in range(1, epochs):
    _, Z_aligned, _ = procrustes(
        hidden_2d_list[epoch_idx - 1],  # Reference (previous epoch)
        hidden_2d_list[epoch_idx]        # Target (current epoch)
    )
    hidden_2d_list[epoch_idx] = Z_aligned  # Replace with aligned version
```

**3. Frame Update Function**:
Now uses per-frame coordinates for both modes:
```python
def update(frame_idx):
    output = predictions[frame_idx].flatten()
    pc1_frame = pc1_all_frames[frame_idx]  # Per-frame coordinates
    pc2_frame = pc2_all_frames[frame_idx]  # Per-frame coordinates
    scatter._offsets3d = (pc1_frame, pc2_frame, output)
    epoch_text.set_text(f'Epoch: {frame_idx}')
    return scatter, epoch_text
```

---

## Visualization Output

Both modes generate complementary information:

| Aspect | Mode 1 (Static) | Mode 2 (Procrustes) |
|--------|-----------------|-------------------|
| **Ground truth manifold** | Fixed at anchor epoch | Evolves per frame |
| **PCA basis** | Single fixed basis | Changes per epoch, then aligned |
| **Visual focus** | Network convergence to target | Co-evolution of geometry & predictions |
| **Use case** | See functional convergence | Understand representation learning |
| **File** | `pca_3d_convergence.gif` | `pca_3d_convergence_procrustes.gif` |

---

## Usage in Main Pipeline

The `main()` function now generates both visualizations:

```python
# Mode 1: Static manifold
print("Creating PCA 3D visualization (anchor epoch mode)...")
create_pca_3d_animation(
    visualizer.data['y_test'],
    visualizer.data['predictions'],
    visualizer.data['hidden_states'],
    pca_epoch=-1,
    output_path='visualizations/topk-sum/pca_3d_convergence.gif'
)

# Mode 2: Procrustes-aligned evolving manifold
print("Creating PCA 3D visualization (all epochs + Procrustes mode)...")
create_pca_3d_animation(
    visualizer.data['y_test'],
    visualizer.data['predictions'],
    visualizer.data['hidden_states'],
    pca_epoch='all',
    output_path='visualizations/topk-sum/pca_3d_convergence_procrustes.gif'
)
```

---

## Dependencies

Added import:
```python
from scipy.spatial import procrustes
```

---

## Benefits of Two-Mode Approach

1. **Flexibility**: Different analytical perspectives on the same data
2. **Comprehensive**: Static mode shows convergence, Procrustes mode shows geometric evolution
3. **Interpretability**: Each mode answers different questions:
   - Static: "How well does the network learn the target?"
   - Procrustes: "How does the representation evolve during training?"
4. **Smoothness**: Procrustes alignment eliminates visual jitter from PCA basis changes
