<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

$$
\min_G \max_D V(D,G) =
\mathbb{E}_{x\sim p_{data}}[\log D(x)] +
\mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))]
$$


# 🎯 GAN Training: Loss Functions & Dynamics

Generative Adversarial Networks (GANs) implement a **two-player minimax game** between two neural networks:

- **Generator (G)**: Generates synthetic data from random noise
- **Discriminator (D)**: Distinguishes real data from generated (fake) data

---

## 📊 Minimax Objective

The original GAN objective proposed by Goodfellow et al. (2014):

min_G max_D V(D, G) =
E[x ~ p_data][ log D(x) ] +
E[z ~ p_z][ log(1 − D(G(z))) ]

Where:
- `x ~ p_data` → real data samples
- `z ~ p_z` → random noise (Gaussian or Uniform)
- `D(x)` → probability that `x` is real
- `G(z)` → generated sample from noise `z`

---

## 🔴 Discriminator Loss

The discriminator is trained to correctly classify **real images as real** and **fake images as fake**.

### Binary Cross-Entropy Loss

L_D =
- 1/2 * (
    E[x ~ p_data][ log D(x) ] +
    E[z ~ p_z][ log(1 − D(G(z))) ]
)

**Intuition:**
- Maximize `D(x)` for real samples
- Minimize `D(G(z))` for fake samples

---

## 🟢 Generator Loss (Original Minimax)

The generator tries to minimize the discriminator’s ability to detect fake samples.

L_G =
E[z ~ p_z][ log(1 − D(G(z))) ]

---

## 🚨 Problem: Vanishing Gradients

Early in training, the discriminator becomes very strong:

D(G(z)) ≈ 0

This causes the gradient of the generator loss to become very small:

∇G log(1 − D(G(z))) ≈ very small

As a result:
- Generator learns very slowly
- Training becomes unstable

---

## ✅ Solution: Non-Saturating Generator Loss

Instead of minimizing `log(1 − D(G(z)))`, the generator **maximizes `log(D(G(z)))`**.

### Non-Saturating Generator Loss

L_G(NS) =
- E[z ~ p_z][ log D(G(z)) ]

### Why This Works
- Provides stronger gradients when the discriminator is confident
- Speeds up generator learning
- Used in most practical GAN implementations (including this project)

---

## 🎭 Discriminator Fooling Trick

When training the generator:
- Fake images are given **real labels (1 instead of 0)**
- This directly forces the generator to fool the discriminator
- Improves gradient signal and training stability

---

## 🔄 Training Dynamics & Equilibrium

At the theoretical Nash equilibrium:

- D(x) = 0.5 for all inputs
- Generator distribution p_g matches real data distribution p_data

At this point:
- Discriminator is completely confused
- Generator produces realistic samples

---

## 📚 Reference
Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014
