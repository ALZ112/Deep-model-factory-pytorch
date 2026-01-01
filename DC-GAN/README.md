# DCGAN on MNIST

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate handwritten digit images using the **MNIST dataset**. The model learns to produce realistic-looking digits by training two neural networks—the Generator and the Discriminator—in an adversarial setup.

---

## Overview

A **DCGAN** consists of two components trained simultaneously:

- **Generator (G):** Generates fake images from random noise vectors.
- **Discriminator (D):** Classifies images as real (from the dataset) or fake (generated).

Both networks compete in a **min–max game**, where the generator aims to fool the discriminator, and the discriminator aims to correctly identify real and fake images.

---

## Architecture

### Generator
- Input: Random noise vector `z ∈ ℝ¹⁰⁰`
- Uses **transposed convolution layers** to upsample noise into `28×28` images
- Key components:
  - Batch Normalization
  - ReLU activations
  - `Tanh` output activation (images normalized to `[-1, 1]`)

### Discriminator
- Input: `28×28` grayscale image
- Uses **convolutional layers** to downsample images
- Key components:
  - LeakyReLU activations
  - Batch Normalization
  - Sigmoid output for real/fake classification

---

## Why DCGAN Is Better Than Vanilla GAN

| Vanilla GAN | DCGAN |
|------------|-------|
| Fully connected layers | Convolutional layers |
| Ignores spatial structure | Preserves spatial locality |
| Low-quality images | Sharper and more realistic images |
| Hard to scale | Scales well to images |

### Key Improvements
- Convolutional architectures capture spatial patterns in images
- Batch Normalization stabilizes gradient flow
- ReLU and LeakyReLU reduce vanishing gradient issues
- Strided convolutions replace pooling layers
- Overall improved convergence compared to vanilla GANs

---

## Training Details

- **Dataset:** MNIST
- **Loss Function:** Binary Cross Entropy (BCE)
- **Optimizers:** Adam
  - Learning rate: `0.0002`
  - β₁ = `0.5`
- **Training Strategy:**
  - Discriminator trained `p = 1` time per batch
  - Generator trained `k = 3` times per batch
- A fixed noise vector is used to visualize generated images consistently across epochs

---

## Disadvantages and Training Challenges

Despite the improvements of DCGANs, GAN training remains challenging:

### 1. Training Instability
- GANs solve a min–max optimization problem
- Loss values often oscillate and are hard to interpret

### 2. Mode Collapse
- Generator may produce limited varieties of images
- Diversity can decrease even when losses appear normal

### 3. Hyperparameter Sensitivity
- Training is highly sensitive to learning rate, batch size, and optimizer settings
- Small changes can cause training collapse

### 4. Discriminator Dominance
- If the discriminator becomes too strong, generator gradients may vanish
- Requires careful balance between generator and discriminator updates

### 5. Lack of Clear Convergence Metrics
- Lower loss does not necessarily imply better image quality
- Visual inspection of generated samples is often required

---

## Summary

DCGAN significantly improves upon vanilla GANs by using convolutional neural networks, leading to better image quality and improved training stability. However, GANs are **inherently unstable** due to their adversarial nature and require careful tuning, monitoring, and experimentation.

---


## We can some Results After Training For 50 epocs

<img width="242" height="242" alt="sample_epoch_1" src="https://github.com/user-attachments/assets/b29afc44-9668-4606-9541-5625499d0625" />

<img width="242" height="242" alt="sample_epoch_10" src="https://github.com/user-attachments/assets/0dd98a45-afd5-45ac-9a86-dc67b53e9dd5" />

<img width="242" height="242" alt="sample_epoch_20" src="https://github.com/user-attachments/assets/4a92a70d-902a-494f-ba40-711401aeafb2" />

<img width="242" height="242" alt="sample_epoch_30" src="https://github.com/user-attachments/assets/64bcee18-edd3-41a4-a56f-37bf14fc5bd9" />

<img width="242" height="242" alt="sample_epoch_40" src="https://github.com/user-attachments/assets/eeb955db-1eac-454a-8335-02fb6d82dfd7" />

## Future Improvements
- Replace BCE loss with **WGAN-GP** for improved stability
- Add **Spectral Normalization** to the discriminator
- Experiment with different generator–discriminator update ratios
- Evaluate performance on more complex datasets

---

