# 🎯 GAN Training: Loss Functions & Dynamics

Generative Adversarial Networks (GANs) implement a **two-player minimax game** between two neural networks:

- **Generator (G)**: Creates synthetic data from random noise  
- **Discriminator (D)**: Distinguishes between real and generated data  

---

## 📊 The Minimax Objective

The fundamental GAN objective (Goodfellow et al., 2014):

$$
\min_G \max_D V(D, G) =
\mathbb{E}_{x \sim p_{data}}[\log D(x)] +
\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

### Where:
- $x \sim p_{data}$: Real data samples  
- $z \sim p_z$: Random noise vectors (Gaussian or Uniform)  
- $D(x)$: Discriminator’s probability that $x$ is real  
- $G(z)$: Generated sample from noise vector $z$

---

## 🔴 Discriminator Loss (Maximization Step)

The discriminator is trained to **correctly classify real and fake samples**.

### Binary Cross-Entropy Loss

$$
\mathcal{L}_D =
-\frac{1}{2}
\left(
\mathbb{E}_{x \sim p_{data}}[\log D(x)] +
\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
\right)
$$

**Intuition:**
- Maximize $D(x)$ for real images  
- Minimize $D(G(z))$ for fake images  

---

## 🟢 Generator Loss (Minimization Step)

The generator is trained to **fool the discriminator**.

### Original Minimax Generator Loss

$$
\mathcal{L}_G =
\mathbb{E}_{z \sim p_z}[\log (1 - D(G(z)))]
$$

---

## 🚨 Problem: Vanishing Gradients

Early in training, the discriminator often becomes **too strong**:

$$
D(G(z)) \approx 0
$$

This causes gradients to vanish:

$$
\nabla_G \log(1 - D(G(z))) =
-\frac{\nabla_G D(G(z))}{1 - D(G(z))}
$$

When $D(G(z)) \approx 0$, learning slows significantly.

---

## ✅ Solution: Non-Saturating (NS) Generator Loss

Instead of minimizing $\log(1 - D(G(z)))$, the generator **maximizes** $\log(D(G(z)))$.

### Non-Saturating Generator Loss

$$
\mathcal{L}_G^{\text{NS}} =
\mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$



### Why This Works
- Provides **strong gradients** even when the discriminator is confident  
- Improves training stability  
- This is the loss used in **most practical GAN implementations**

---

## 🎭 Discriminator Fooling Trick

In practice, when training the generator:
- Fake samples are labeled as **real (label = 1)**
- This directly maximizes discriminator error
- Encourages faster generator learning

---

## 🔄 Training Dynamics & Equilibrium

### Theoretical Nash Equilibrium

At convergence:

- $D(x) = 0.5$ for all inputs  
- $p_g = p_{data}$  

The discriminator is completely **confused**, and the generator perfectly models the real data distribution.

---

## 📚 Reference
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014


<img width="636" height="658" alt="image" src="https://github.com/user-attachments/assets/1ba86898-3da7-4cc8-bc68-e426f66eafc4" />

