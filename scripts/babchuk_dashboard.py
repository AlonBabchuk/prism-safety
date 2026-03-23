# scripts/babchuk_dashboard.py
# The Babchuk Code v1.0 — Flight monitoring panel.
#
# Live five-panel dashboard with color-coded process-level safety alerts.
# Red background = dimension in pathological range.
# Green = within acceptable range.
#
# Five panels displayed in real time:
# 1. Token entropy
# 2. Branching factor
# 3. KL divergence
# 4. Attention entropy
# 5. Attention span

import torch
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os


def load_thresholds():
    path = os.path.join(
        os.path.dirname(__file__), "..", "presets", "thresholds.json"
    )
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "entropy_thresh": 2.0,
        "kl_thresh": 0.5,
        "branch_thresh": 3.0,
        "attn_entropy_thresh": 1.0,
        "rolling_window": 5,
    }


class BabchukFlightDashboard:
    """
    Flight monitoring panel for The Babchuk Code.
    Provides real-time color-coded process-level safety monitoring
    with rolling averages and threshold alerts.
    """

    def __init__(self, vocab_size, top_k=10,
                 roll_window=None, entropy_thresh=None,
                 kl_thresh=None, branch_thresh=None,
                 attn_entropy_thresh=None):
        th = load_thresholds()
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.roll_window = roll_window if roll_window is not None else int(th.get("rolling_window", 5))
        self.entropy_thresh = entropy_thresh if entropy_thresh is not None else float(th.get("entropy_thresh", 2.0))
        self.kl_thresh = kl_thresh if kl_thresh is not None else float(th.get("kl_thresh", 0.5))
        self.branch_thresh = branch_thresh if branch_thresh is not None else float(th.get("branch_thresh", 3.0))
        self.attn_entropy_thresh = (attn_entropy_thresh if attn_entropy_thresh is not None
                                    else float(th.get("attn_entropy_thresh", 1.0)))

        self.entropy = []
        self.branching_factor = []
        self.kl_divergence = []
        self.attn_entropy = []
        self.attn_span = []

        self.roll_entropy = deque(maxlen=self.roll_window)
        self.roll_kl = deque(maxlen=self.roll_window)
        self.roll_branch = deque(maxlen=self.roll_window)
        self.roll_attn_entropy = deque(maxlen=self.roll_window)

        self.prev_probs = None
        self.total_alerts = 0

    def step(self, logits, attentions=None):
        """Process one generation step. Returns dict of alert states."""
        if logits.dim() > 1:
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)

        H = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        self.entropy.append(H.item())
        self.roll_entropy.append(H.item())

        B = torch.exp(H)
        self.branching_factor.append(B.item())
        self.roll_branch.append(B.item())

        if self.prev_probs is not None:
            D_kl = torch.sum(
                probs * (torch.log(probs + 1e-12)
                         - torch.log(self.prev_probs + 1e-12)),
                dim=-1
            )
            self.kl_divergence.append(D_kl.item())
            self.roll_kl.append(D_kl.item())
        else:
            self.kl_divergence.append(0.0)
            self.roll_kl.append(0.0)

        self.prev_probs = probs.detach()

        if attentions is not None:
            # Select last token's attention, average over heads, concat across layers
            att_flat = torch.cat(
                [a[:, :, -1, :].mean(1) for a in attentions], dim=-1
            )
            # att_flat is now (batch, seq_len * num_layers) — normalise to distribution
            att_flat = att_flat / (att_flat.sum(dim=-1, keepdim=True) + 1e-12)
            att_e = -torch.sum(
                att_flat * torch.log(att_flat + 1e-12), dim=-1
            )
            self.attn_entropy.append(att_e[0].item())
            self.roll_attn_entropy.append(att_e[0].item())
            seq_len = att_flat.shape[-1]
            indices = torch.arange(seq_len, dtype=torch.float, device=att_flat.device)
            self.attn_span.append(
                torch.sum(att_flat[0] * indices, dim=-1).item()
            )
        else:
            self.attn_entropy.append(0.0)
            self.roll_attn_entropy.append(0.0)
            self.attn_span.append(0.0)

        def rm(dq):
            return sum(dq) / len(dq) if dq else 0.0

        alerts = {
            "Entropy": rm(self.roll_entropy) < self.entropy_thresh,
            "KL Divergence": rm(self.roll_kl) > self.kl_thresh,
            "Branching Factor": rm(self.roll_branch) < self.branch_thresh,
            "Attention Entropy": (rm(self.roll_attn_entropy)
                                  < self.attn_entropy_thresh),
            "Attention Span": False,
        }
        self.total_alerts += sum(alerts.values())
        return alerts


def register_babchuk_hook(model, metrics_obj):
    """Attach hook to LM head. Returns handle — call handle.remove() when done."""
    def hook(module, input, output):
        metrics_obj.step(output[:, -1, :])
    return model.lm_head.register_forward_hook(hook)


def live_flight_panel(metrics_obj):
    """
    Create the live flight monitoring panel.
    Returns (update_function, figure).
    Call update_function(step, alerts) at each generation step.
    """
    plt.ion()
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.suptitle(
        "The Babchuk Code — Flight Monitoring Panel",
        fontsize=13, fontweight="bold"
    )

    metric_names = [
        "Entropy", "Branching Factor", "KL Divergence",
        "Attention Entropy", "Attention Span"
    ]
    line_objs = []
    bg_patches = []

    for ax, name in zip(axes, metric_names):
        line, = ax.plot([], [], label=name, color="green", linewidth=1.5)
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        patch = mpatches.Rectangle(
            (0, 0), 0, 0, color="red", alpha=0.0, zorder=0
        )
        ax.add_patch(patch)
        line_objs.append(line)
        bg_patches.append(patch)

    axes[-1].set_xlabel("Token Step")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def update(frame, alerts):
        x = list(range(len(metrics_obj.entropy)))
        data = [
            metrics_obj.entropy, metrics_obj.branching_factor,
            metrics_obj.kl_divergence, metrics_obj.attn_entropy,
            metrics_obj.attn_span
        ]
        for ax, line, y, name, patch in zip(
            axes, line_objs, data, metric_names, bg_patches
        ):
            line.set_data(x, y)
            if alerts.get(name) and y:
                y_min = min(y) * 0.9
                y_max = max(y) * 1.1 if max(y) > 0 else 1.0
                patch.set_xy((0, y_min))
                patch.set_width(len(x))
                patch.set_height(y_max - y_min)
                patch.set_alpha(0.18)
                line.set_color("#e74c3c")
            else:
                patch.set_alpha(0.0)
                line.set_color("#2ecc71")
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.02)

    return update, fig
