# β-VAE: Theory, Implementation & Best Practices

## 📋 Overview

This repository documents the **theory, motivation, and best practices** behind **Variational Autoencoders (VAEs)** and **β-VAEs**, with a primary focus on image data such as **MNIST**.

The goal is to understand how VAEs learn **probabilistic latent representations**, how **β controls disentanglement**, and how to train **stable, meaningful generative models**.

---

## 🎯 Core Concepts

### What is a Variational Autoencoder (VAE)?

A **Variational Autoencoder (VAE)** is a generative model that:

- Encodes data into a **probabilistic latent space**
- Learns a structured representation constrained by a **prior**
- Generates new samples by decoding latent variables
- Uses **variational inference** to approximate the posterior

Unlike standard autoencoders, VAEs enforce **continuity and smoothness** in the latent space.

---

### What is a β-VAE?

A **β-VAE** extends the VAE objective by introducing a hyperparameter **β**, which controls the strength of latent regularization.

This explicitly trades off:

- **Reconstruction quality**
- **Latent disentanglement**
- **Generative smoothness**

---

## 🧮 Mathematical Foundations

### Evidence Lower Bound (ELBO)

The VAE maximizes the **Evidence Lower Bound (ELBO)**:

$$
\log p(x) \ge \mathbb{E}_{z \sim q_\phi(z|x)} [ \log p_\theta(x|z) ] - D_{KL}( q_\phi(z|x) \| p(z) )
$$

---

### β-VAE Objective

The β-VAE modifies the ELBO as:

$$
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - \beta \cdot D_{KL}( q_\phi(z|x) \| p(z) )
$$

---

### Where:

- $\( q_\phi(z|x) \) — encoder (approximate  posterior)$
- $\( p_\theta(x|z) \) — decoder (likelihood)$
- $\( p(z) = \mathcal{N}(0, I) \) — prior$
- **β** controls the **disentanglement–reconstruction trade-off**

---

## 🔁 Reconstruction Loss Choices

| Loss | Assumption | Best Use |
|-----|-----------|----------|
| Binary Cross-Entropy (BCE) | Bernoulli likelihood | Binary / grayscale images (MNIST) |
| Mean Squared Error (MSE) | Gaussian, fixed variance | Continuous-valued data |
| Gaussian NLL | Gaussian, learnable variance | General continuous data |

---

## 📐 KL Divergence (Gaussian Case)

For diagonal Gaussian latent distributions:

$$
D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)) = -\frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)
$$


This term regularizes the latent distribution toward the prior.

---

## ⚙️ Model Components

### Encoder
- $Maps input \( x \rightarrow (\mu, \log \sigma^2) \)$
- Defines the approximate posterior

### Latent Space
- Sampling performed using the **reparameterization trick**
- Enables gradient flow through stochastic nodes

### Decoder
- $Maps latent variables \( z \rightarrow \hat{x} \)$
- Defines the likelihood model

---

## 📊 Effect of β Parameter

### β Value Guidelines

| β | Reconstruction | Latent Structure | Generation | Use Case |
|--|--|--|--|--|
| β < 1 | Excellent | Chaotic | Poor | Maximum reconstruction |
| β = 1 | Good | Moderate | Good | Standard VAE |
| β > 1 | Blurry | Disentangled | Very good | Representation learning |
| β ≫ 1 | Very blurry | Highly structured | Excellent | Strong disentanglement |

---

### Expected Training Behavior

| Metric | Trend | Interpretation |
|------|------|----------------|
| Total loss | ↓ | Model learning |
| Reconstruction loss | ↓ | Better fidelity |
| KL divergence | ↑ | Latent structure forming |
| Validation loss | ↓ → plateau | Generalization |

⚠️ **KL ≈ 0 often indicates posterior collapse.**

---

## 🚀 Advanced Topics

### Convolutional VAEs
- Preserve spatial structure *(respect local pixel relationships)*
- Improve sharpness *(reduce blur via inductive bias)*
- Reduce parameter count *(weight sharing)*

---

### Disentanglement Metrics
- **β-VAE score** — predictability of factors from latents  
- **Mutual Information Gap (MIG)** — factor separation quality  
- **FactorVAE metric** — variance captured per factor  

---

### Preventing Posterior Collapse
- **KL annealing** — prevents early encoder shutdown
- **Free bits** — forces minimum information per latent
- **Reduce decoder capacity** — prevents latent bypass
- **Increase latent dimensionality carefully** — avoids over-regularization

---

## 📈 Troubleshooting Guide

### Blurry Reconstructions
- Reduce β (less regularization pressure)
- Use MSE instead of BCE (smoother gradients)
- Increase decoder capacity (improve expressiveness)
- Use convolutional architectures (better spatial modeling)

### Poor Generation Quality
- Increase β (improves latent structure)
- Ensure KL is non-zero (latent must be informative)
- Verify latent dimension size (avoid bottlenecks)

### Training Instability
- Reduce learning rate (avoid divergence)
- Apply gradient clipping (stabilize updates)
- Check initialization (bad init causes collapse)

### Mode Collapse
- Increase β (encourages diversity)
- Improve latent regularization (avoid identical encodings)
- Increase training diversity (reduce memorization)

---

## 📝 Training Checklist

### Before Training
- Normalize data properly (stable optimization)
- Choose reconstruction loss (match data distribution)
- Set latent dimension and β (capacity vs regularization)
- Enable logging and visualization (early debugging)

### During Training
- Monitor reconstruction and KL separately (detect collapse early)
- Visualize samples periodically (qualitative checks)
- Validate on held-out data (generalization)

### After Training
- Inspect reconstructions (fidelity check)
- Generate new samples (generative quality)
- Visualize latent space (structure and clustering)
- Perform interpolation analysis (smoothness & disentanglement)

---

## 🎯 Key Takeaways

- **β is the primary control knob** for reconstruction vs disentanglement
- **KL divergence increasing is healthy**
- **High β causes blur but improves structure**
- **Posterior collapse destroys latent usefulness**
- **Visualization is essential — loss curves are not enough**
- **Convolutional VAEs outperform MLPs on images**
- **KL annealing greatly improves stability**

---


# 📊 β-VAE Training Results (Epoch 30)

### 🔹 β = 4.0
- Epoch 30: 100%|██████████| 469/469 [00:07<00:00, 62.16it/s, Loss=144, BCE=110, KLD=8.51]
- Train Loss: 142.4684, Test Loss: 142.8658
- Train BCE: 108.8103, KLD: 8.4145
- Test BCE: 109.8983, KLD: 8.2419

### 🔹 β = 0.1
Epoch 30: 100%|██████████| 469/469 [00:06<00:00, 68.66it/s, Loss=66, BCE=60.6, KLD=53.6]
Train Loss: 68.2079, Test Loss: 69.7244
Train BCE: 62.8916, KLD: 53.1632
Test BCE: 64.4485, KLD: 52.7590

### 🔹 β = 2.0
Epoch 30: 100%|██████████| 469/469 [00:06<00:00, 68.03it/s, Loss=123, BCE=96, KLD=13.4]
Train Loss: 118.0636, Test Loss: 119.0178
Train BCE: 91.5529, KLD: 13.2553
Test BCE: 92.7549, KLD: 13.1315



---

## 🔗 Further Reading

- *Auto-Encoding Variational Bayes* — Kingma & Welling  
- *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*  
- *Understanding Disentangling in β-VAE*
