# scripts/babchuk_metrics.py
# The Babchuk Code v1.0 — Core Layer 1 signal collection.
#
# Captures seven token-level signals per generation step
# via forward hooks on the language model head.
# Works with any HuggingFace causal language model.
#
# Signals:
# H_t   = token entropy           (-sum p log p)
# B_t   = effective branching      (exp(H_t))
# D_t   = KL divergence            (KL(p_t || p_{t-1}))
# dH_t  = entropy gradient         (H_t - H_{t-1})
# C_t   = top-k mass concentration (sum of top-k probabilities)

import torch
import torch.nn.functional as F


class BabchukMetrics:
    """
    Core metric collector for The Babchuk Code.
    Attach to any HuggingFace model via register_babchuk_hook().
    """

    def __init__(self, vocab_size, top_k=10):
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.entropy = []
        self.branching_factor = []
        self.kl_divergence = []
        self.entropy_gradient = []
        self.topk_mass = []
        self.prev_probs = None

    def step(self, logits):
        """Process one generation step. logits: [batch_size, vocab_size] or [vocab_size]"""
        if logits.dim() > 1:
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)

        H = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        self.entropy.append(H.item())

        B = torch.exp(H)
        self.branching_factor.append(B.item())

        if self.prev_probs is not None:
            D_kl = torch.sum(
                probs * (torch.log(probs + 1e-12)
                         - torch.log(self.prev_probs + 1e-12)),
                dim=-1
            )
            self.kl_divergence.append(D_kl.item())
        else:
            self.kl_divergence.append(0.0)

        delta_H = (self.entropy[-1] - self.entropy[-2]
                   if len(self.entropy) > 1 else 0.0)
        self.entropy_gradient.append(delta_H)

        topk_vals, _ = torch.topk(probs, self.top_k, dim=-1)
        self.topk_mass.append(torch.sum(topk_vals, dim=-1).item())

        self.prev_probs = probs.detach()

        return {
            "entropy": H.item(),
            "branching_factor": B.item(),
            "kl_divergence": self.kl_divergence[-1],
            "entropy_gradient": delta_H,
            "topk_mass": self.topk_mass[-1],
        }


def register_babchuk_hook(model, metrics_obj):
    """
    Attach forward hook to LM head to capture logits at each generation step
    without modifying model code.
    Returns the hook handle — call handle.remove() when done.
    """
    def hook(module, input, output):
        metrics_obj.step(output[:, -1, :])
    return model.lm_head.register_forward_hook(hook)
