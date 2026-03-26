import matplotlib.pyplot as plt
import numpy as np
import os

timesteps = np.arange(50)
np.random.seed(42)

entropy_t1 = np.clip(
    np.linspace(4.0, 3.0, 50) + np.random.normal(0, 0.1, 50), 0, 5)
entropy_t2 = np.clip(
    np.concatenate([
        np.linspace(4.0, 1.5, 10),
        np.linspace(1.5, 1.2, 40)
    ]) + np.random.normal(0, 0.05, 50), 0, 5)

branching_t1 = np.exp(entropy_t1)
branching_t2 = np.exp(entropy_t2)

kl_t1 = np.abs(np.gradient(entropy_t1)) + np.random.normal(0, 0.05, 50)
kl_t2 = np.abs(np.gradient(entropy_t2)) + np.random.normal(0, 0.05, 50)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle("PRISM - Process Signatures", fontsize=13, fontweight="bold")

for ax, t1, t2, ylabel, threshold, tlabel in [
    (axes[0], entropy_t1, entropy_t2, "Entropy H_t", 1.5, "Rigidity threshold"),
    (axes[1], branching_t1, branching_t2, "Branching Factor B_t", 3.0, "Narrow scope threshold"),
    (axes[2], kl_t1, kl_t2, "KL Divergence D_t", 2.0, "Instability threshold"),
]:
    ax.plot(timesteps, t1, color="#2ecc71", linewidth=2, label="Coherent processing")
    ax.plot(timesteps, t2, color="#e74c3c", linewidth=2, label="Distorted processing")
    ax.axhline(y=threshold, color="gray", linestyle=":", alpha=0.6, label=tlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Token Step")
plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "process_signatures.jpg")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path}")
plt.show()
