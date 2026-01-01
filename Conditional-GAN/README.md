
# Conditional DCGAN with Projection Discriminator (MNIST)

This project implements an **improved Conditional DCGAN** on the MNIST dataset.  
Compared to a standard DCGAN, this version introduces **label conditioning** and a
**Projection Discriminator**, resulting in **better class control, improved stability,
and higher-quality conditional samples**.

---

## Overview

A traditional DCGAN generates images **unconditionally**, meaning it has no control
over which class is generated.  
This implementation extends DCGAN by conditioning both the **Generator** and
**Discriminator** on class labels (digits 0–9).

Key improvements include:

- **Conditional Generator using label embeddings**
- **Projection Discriminator for better class-aware discrimination**
- **Replacement of BCELoss with BCEWithLogitsLoss for improved numerical stability**
- Clear visualization of **class-controlled generation**

---

## Model Architecture

### Generator (Conditional)

- Input:
  - Random noise vector `z ∈ ℝ¹²⁸`
  - Class label `y ∈ {0, …, 9}`
- Label conditioning:
  - Labels are embedded using an `nn.Embedding`
  - Embedding is concatenated with the noise vector
- Architecture:
  - Transposed convolutions for upsampling
  - Batch Normalization
  - ReLU activations
  - `Tanh` output activation

This allows the generator to **explicitly control which digit class is generated**.

---

### Projection Discriminator (Key Improvement)

Instead of simply concatenating labels or using separate classifiers, this project
uses a **Projection Discriminator**, which is more principled and stable.

#### How it works
- The image is passed through convolutional layers to extract features
- A label embedding is learned
- The discriminator output is computed as:

\[
D(x, y) = f(x) + \langle h(x), e(y) \rangle
\]

Where:
- `h(x)` = image feature representation
- `e(y)` = embedding of label `y`
- `⟨·,·⟩` = inner product (projection)

This forces the discriminator to judge **whether an image matches a given label**,
not just whether it looks real.

---

## Why Projection Discriminator Is Better than Normal DCGAN

| Standard DCGAN | This Implementation |
|---------------|---------------------|
| Unconditional generation | Conditional generation |
| No label control | Explicit class control |
| Discriminator sees only images | Discriminator sees image–label compatibility |
| More mode collapse | Reduced mode collapse |
| Weaker gradients for G | Stronger, class-aware gradients |

### Key Advantages
- Better separation between digit classes
- More stable adversarial training
- Generator learns **class semantics**, not just visual realism
- Works especially well for multi-class datasets like MNIST

---

## Training Details

- **Dataset:** MNIST
- **Loss Function:** `BCEWithLogitsLoss`  
  *(replaces Sigmoid + BCELoss for better numerical stability and gradient flow)*
- **Optimizers:** Adam
  - Learning rate: `0.0002`
  - β₁ = `0.5`
- **Training Schedule:**
  - Discriminator updates per batch: `p = 1`
  - Generator updates per batch: `k = 3`
- Class labels are sampled randomly during training for conditional learning

---

## Visualization

- Fixed noise vectors are combined with **fixed labels**
- Generated images are saved in a grid format
- Each row corresponds to a digit class
- Enables easy inspection of class consistency and diversity

---

## Training Stability and Limitations

Although this implementation improves upon standard DCGANs, **GAN training remains
inherently unstable** due to its adversarial min–max nature.

Observed challenges include:
- Oscillating generator and discriminator losses
- Temporary mode collapse during training
- Sensitivity to hyperparameters and update ratios

📌 **Supporting visualizations are included** to demonstrate:
- Instability during early training
- Progressive improvement as training stabilizes
- Class-wise consistency of generated samples

---

## Summary

This project demonstrates how **Conditional DCGANs with Projection Discriminators**
significantly improve upon standard DCGANs by incorporating label information in a
principled way. The result is **more stable training, explicit class control, and
higher-quality generated images**, particularly for multi-class datasets like MNIST.
