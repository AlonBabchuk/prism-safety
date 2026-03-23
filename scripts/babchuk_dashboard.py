# scripts/babchuk_dashboard.py
# The Babchuk Code v1.1 — Flight monitoring panel (fixed + improved)
import torch
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os


def load_thresholds():
    path = os.path.join(os.path.dirname(__file__), "..", "presets", "thresholds.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "entropy_thresh": 2.0,
        "kl_thresh": 0.5,
        "branch_thresh": 3.0,
        "attn_entropy_thresh": 1.0,
        "attn_span_thresh": [None, None],
        "rolling_window": 5,
    }


class BabchukFlightDashboard:
    def __init__(self, vocab_size, top_k=10, roll_window=None,
                 entropy_thresh=None, kl_thresh=None, branch_thresh=None,
                 attn_entropy_thresh=None, attn_span_thresh=None):
        th = load_thresholds()
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.roll_window = roll_window or int(th["rolling_window"])
        self.entropy_thresh = entropy_thresh if entropy_thresh is not None else float(th["entropy_thresh"])
        self.kl_thresh = kl_thresh if kl_thresh is not None else float(th["kl_thresh"])
        self.branch_thresh = branch_thresh if branch_thresh is not None else float(th["branch_thresh"])
        self.attn_entropy_thresh = attn_entropy_thresh if attn_entropy_thresh is not None else float(th["attn_entropy_thresh"])
        self.attn_span_thresh = attn_span_thresh if attn_span_thresh is not None else th.get("attn_span_thresh")

        self.entropy = []
        self.branching_factor = []
        self.kl_divergence = []
        self.attn_entropy = []
        self.attn_span = []

        self.roll_entropy = deque(maxlen=self.roll_window)
        self.roll_kl = deque(maxlen=self.roll_window)
        self.roll_branch = deque(maxlen=self.roll_window)
        self.roll_attn_entropy = deque(maxlen=self.roll_window)
        self.roll_attn_span = deque(maxlen=self.roll_window)

        self.prev_probs = None
        self.total_alerts = 0

    @torch.no_grad()
    def step(self, logits, attentions=None):
        if logits.dim() > 1:
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)

        H = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
        self.entropy.append(H)
        self.roll_entropy.append(H)

        B = torch.exp(torch.tensor(H)).item()
        self.branching_factor.append(B)
        self.roll_branch.append(B)

        if self.prev_probs is not None:
            kl = torch.sum(probs * (torch.log(probs + 1e-12) - torch.log(self.prev_probs + 1e-12))).item()
            self.kl_divergence.append(kl)
            self.roll_kl.append(kl)
        else:
            self.kl_divergence.append(0.0)
            self.roll_kl.append(0.0)

        self.prev_probs = probs.detach()

        if attentions is not None and isinstance(attentions, (tuple, list)) and len(attentions) > 0:
            layer_spans = []
            layer_entropies = []
            for layer_attn in attentions:
                att = layer_attn[:, :, -1, :].mean(dim=1)
                att = att / (att.sum(dim=-1, keepdim=True) + 1e-12)
                att = att[0]
                e = -torch.sum(att * torch.log(att + 1e-12)).item()
                layer_entropies.append(e)
                indices = torch.arange(att.shape[-1], dtype=torch.float, device=att.device)
                span = torch.sum(att * indices).item()
                layer_spans.append(span)
            attn_e = sum(layer_entropies) / len(layer_entropies)
            attn_s = sum(layer_spans) / len(layer_spans)
            self.attn_entropy.append(attn_e)
            self.roll_attn_entropy.append(attn_e)
            self.attn_span.append(attn_s)
            self.roll_attn_span.append(attn_s)
        else:
            self.attn_entropy.append(0.0)
            self.roll_attn_entropy.append(0.0)
            self.attn_span.append(0.0)
            self.roll_attn_span.append(0.0)

        def rm(dq): return sum(dq) / len(dq) if dq else 0.0

        alerts = {
            "Entropy": rm(self.roll_entropy) < self.entropy_thresh,
            "KL Divergence": rm(self.roll_kl) > self.kl_thresh,
            "Branching Factor": rm(self.roll_branch) < self.branch_thresh,
            "Attention Entropy": rm(self.roll_attn_entropy) < self.attn_entropy_thresh,
            "Attention Span": False,
        }
        if self.attn_span_thresh and self.attn_span_thresh[0] is not None:
            s = rm(self.roll_attn_span)
            alerts["Attention Span"] = not (self.attn_span_thresh[0] <= s <= self.attn_span_thresh[1])
        self.total_alerts += sum(alerts.values())
        return alerts


def register_babchuk_hook(model, metrics_obj):
    def hook(module, input, output):
        metrics_obj.step(output[:, -1, :])
    return model.lm_head.register_forward_hook(hook)


def live_flight_panel(metrics_obj):
    plt.ion()
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("The Babchuk Code — Flight Monitoring Panel v1.1", fontsize=14, fontweight="bold")

    metric_names = ["Entropy", "Branching Factor", "KL Divergence",
                    "Attention Entropy", "Attention Span"]
    lines = []
    patches = []

    for ax, name in zip(axes, metric_names):
        line, = ax.plot([], [], label=name, color="#2ecc71", lw=1.8)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        patch = mpatches.Rectangle((0, 0), 0, 0, color="red", alpha=0.0, zorder=-1)
        ax.add_patch(patch)
        lines.append(line)
        patches.append(patch)

    axes[-1].set_xlabel("Token Step")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(_frame, alerts):
        x = list(range(len(metrics_obj.entropy)))
        data = [metrics_obj.entropy, metrics_obj.branching_factor,
                metrics_obj.kl_divergence, metrics_obj.attn_entropy,
                metrics_obj.attn_span]

        for ax, line, y_vals, name, patch in zip(axes, lines, data, metric_names, patches):
            line.set_data(x, y_vals)
            alert = alerts.get(name, False)
            if alert and y_vals:
                ymin = min(y_vals) * 0.92
                ymax = max(y_vals) * 1.08 if max(y_vals) > 0 else 1.0
                patch.set_xy((0, ymin))
                patch.set_width(len(x))
                patch.set_height(ymax - ymin)
                patch.set_alpha(0.22)
                line.set_color("#e74c3c")
            else:
                patch.set_alpha(0.0)
                line.set_color("#2ecc71")
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    return update, fig
