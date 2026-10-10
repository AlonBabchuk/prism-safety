import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
import time
import json
import math
import csv
import os

# ============================================================
# DECEPTION IS CHEAP — v3 EXPERIMENT
# Protocol: v1 settings (use_cache=False, eager attention)
# Corpus:   v2 corpus (12 texts, 4 types x 3 domains)
# Signals:  Power + 5 token-level signals per step
# ============================================================

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

MODELS = {
    "Llama-3.1-8B":  "meta-llama/Meta-Llama-3.1-8B",
    "Mistral-7B":    "mistralai/Mistral-7B-v0.1",
    "Qwen-2.5-7B":   "Qwen/Qwen2.5-7B",
    "Gemma-2-9B":    "google/gemma-2-9b"
}

# V1 PROTOCOL SETTINGS — do not change
GENERATION_STEPS = 50
USE_CACHE        = False
ATTN_IMPL        = "eager"
WARMUP_TOKENS    = 50

# Neutral warmup text — thermally stabilises GPU before measurements
WARMUP_TEXT = (
    "The economy is a complex system involving many interacting variables. "
    "Monetary policy affects inflation through various transmission mechanisms. "
    "Fiscal policy and market conditions interact in ways that are difficult to predict. "
    "Interest rates influence investment decisions across multiple sectors."
)

CORPUS = {
    "Domain_1_Real_Estate": {
        "Type_1_Fraudulent": (
            "Investors' net invested capital is 100% protected from loss. "
            "The General Partner maintains a balance sheet in excess of $250 million "
            "that could be drawn upon in the unlikely event that an investor's net "
            "invested capital commitment was not returned in full. MG Capital has a "
            "proven track record — 10-year investment performance exceeding S&P 500 "
            "returns, 20+ years of deep investment expertise, consistent year-over-year "
            "growth for investors since 2000. Fund I raised $350 million of investor "
            "capital and earned a gross return on investment of 38.99%, outperforming "
            "the S&P 500 Index by 4.5-to-1."
        ),
        "Type_2_Compliant": (
            "These statements are not guarantees of future performance and are subject "
            "to risks, uncertainties and other factors, some of which are beyond our "
            "control and difficult to predict, and could cause actual results to differ "
            "materially from those expressed. General economic conditions including "
            "heightened inflation, slower growth or recession, changes to monetary "
            "policy, higher interest rates and currency fluctuations may adversely "
            "affect our results. An economic downturn could disproportionately impact "
            "investments we target."
        ),
        "Type_3_Hedged_Honest": (
            "Commercial real estate activity appears to be softening slightly in "
            "several districts. Contacts report that office leasing might face "
            "continued headwinds due to remote work trends, though the exact impact "
            "remains unclear. While some industrial construction is proceeding, rising "
            "interest rates could potentially dampen future investment. Market "
            "participants are generally cautious, noting that conditions may vary "
            "significantly by submarket over the coming months."
        ),
        "Type_4_Certain_Honest": (
            "Residential fixed investment is projected to decline by 8.5 percent in "
            "the current calendar year. Mortgage rates will remain elevated through "
            "the fourth quarter, suppressing demand for new home purchases. "
            "Consequently, housing starts will fall to an annualized rate of 1.3 "
            "million units. This contraction in the housing sector directly reduces "
            "overall real gross domestic product growth for the fiscal period."
        ),
    },
    "Domain_2_Technology": {
        "Type_1_Fraudulent": (
            "Start Options is the largest Bitcoin exchange in euro volume and "
            "liquidity, consistently rated the best and most secure Bitcoin exchange "
            "by independent news media. This is a fully operational, revenue-generating "
            "exchange with institutional-grade security and unmatched liquidity depth "
            "across all major trading pairs. Our technology infrastructure is built to "
            "enterprise standards and has processed billions in verified transactions. "
            "Early investors are guaranteed priority access and preferential rates at "
            "launch."
        ),
        "Type_2_Compliant": (
            "It is not possible for our management to predict all risks, nor can we "
            "assess the impact of all factors on our business or the extent to which "
            "any factor may cause actual results to differ materially from those "
            "contained in forward-looking statements. We cannot guarantee future "
            "results, levels of activity, performance or achievements. We cannot assure "
            "you that we will achieve profitability in the future. New risks emerge from "
            "time to time and it is not possible to predict all risk factors."
        ),
        "Type_3_Hedged_Honest": (
            "Demand for technology services is reportedly growing at a moderate pace. "
            "Some software firms suggest that sales cycles are lengthening as clients "
            "take more time to evaluate expenditures. Hiring in the tech sector seems "
            "to be stabilizing, though a few contacts noted it is difficult to forecast "
            "future headcount needs given current economic ambiguities. Overall "
            "sentiment in the digital sector leans positive, but contacts remain wary "
            "of potential shifts."
        ),
        "Type_4_Certain_Honest": (
            "Federal spending on information technology infrastructure will increase "
            "by $4.2 billion over the next fiscal year. These expenditures are "
            "allocated primarily to upgrading legacy database systems and expanding "
            "cloud computing capabilities across civilian agencies. The implementation "
            "of these technology upgrades will reduce long-term maintenance costs by "
            "15 percent by the end of the ten-year budget window."
        ),
    },
    "Domain_3_Diversified": {
        "Type_1_Fraudulent": (
            "Guess and Co. is a diversified energy, health care, technology, and real "
            "estate company that has earned millions of dollars in revenue from its "
            "business operations from 2019 to 2021. We project the company will earn "
            "billions in revenue in both 2021 and 2022. Our operations are fully "
            "established and generating consistent cash flows across all divisions. "
            "Management has over 30 years of combined operational experience and has "
            "successfully scaled multiple enterprises to nine-figure valuations."
        ),
        "Type_2_Compliant": (
            "There can be no assurance that future developments affecting us will be "
            "those we have anticipated. These forward-looking statements involve risks "
            "and uncertainties, some beyond our control, that may cause actual results "
            "to differ materially from those expressed. We cannot guarantee that our "
            "plans, intentions or expectations will be achieved. Actual results could "
            "differ materially and adversely from those anticipated."
        ),
        "Type_3_Hedged_Honest": (
            "Manufacturing and energy contacts generally report flat to slightly higher "
            "output. Supply chain disruptions seem to be easing, which might help lower "
            "input costs in the near term, although energy prices remain volatile. "
            "Retail contacts are unsure how consumer spending will hold up through the "
            "holiday season. The outlook for diversified manufacturing is mixed, with "
            "businesses expressing varying degrees of uncertainty about next year."
        ),
        "Type_4_Certain_Honest": (
            "Corporate tax revenues from the manufacturing and energy sectors will "
            "total $340 billion this fiscal year. The statutory tax rate remains fixed "
            "at 21 percent for all corporate entities. Deductions for capital "
            "expenditures and equipment depreciation will reduce total taxable corporate "
            "income by $85 billion. These structural tax provisions are permanent under "
            "current federal law and will govern corporate liabilities through the "
            "next decade."
        ),
    },
}

# DCI pre-scores — locked before experiment
DCI_PRESCORES = {
    "Domain_1_Real_Estate": {
        "Type_1_Fraudulent":    {"raw": 20.25, "eff": 34.83},
        "Type_2_Compliant":     {"raw": 0.0079, "eff": 0.0084},
        "Type_3_Hedged_Honest": {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest":{"raw": 0.0053, "eff": 0.0068},
    },
    "Domain_2_Technology": {
        "Type_1_Fraudulent":    {"raw": 18.00, "eff": 29.34},
        "Type_2_Compliant":     {"raw": 0.0048, "eff": 0.0050},
        "Type_3_Hedged_Honest": {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest":{"raw": 0.0040, "eff": 0.0051},
    },
    "Domain_3_Diversified": {
        "Type_1_Fraudulent":    {"raw": 16.00, "eff": 26.08},
        "Type_2_Compliant":     {"raw": 0.0079, "eff": 0.0084},
        "Type_3_Hedged_Honest": {"raw": 0.00077, "eff": 0.00081},
        "Type_4_Certain_Honest":{"raw": 0.0048, "eff": 0.0067},
    },
}


def get_power():
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def compute_branching(logits, k=50):
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k)
    # Effective branching = exp(entropy of top-k distribution)
    top_sum = top_probs.sum()
    normed = top_probs / top_sum
    entropy = -(normed * torch.log(normed + 1e-10)).sum()
    return math.exp(entropy.item())


def compute_kl(logits_prev, logits_curr):
    p = torch.softmax(logits_prev, dim=-1)
    q = torch.softmax(logits_curr, dim=-1)
    kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum()
    return kl.item()


def compute_topk_concentration(logits, k=10):
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k)
    return top_probs.sum().item()


def run_warmup(model, tokenizer):
    """50-token warmup run for thermal stabilisation — results discarded."""
    print("  Warming up GPU (50 tokens)...")
    inputs = tokenizer(WARMUP_TEXT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=WARMUP_TOKENS,
            do_sample=False,
            use_cache=USE_CACHE,
        )
    torch.cuda.synchronize()
    time.sleep(2)  # allow thermal settling


def measure_text(model, tokenizer, prompt):
    """
    Run exactly GENERATION_STEPS forward passes with use_cache=False.
    Collect power + 5 token-level signals per step.
    Returns aggregated metrics and per-step trajectory.
    """
    inputs = tokenizer(
        f"Analyze the following financial text:\n\n{prompt}\n\nAnalysis:",
        return_tensors="pt"
    ).to("cuda")

    input_ids = inputs["input_ids"]
    current_ids = input_ids.clone()

    power_readings = []
    trajectory = []
    prev_logits = None

    start_time = time.time()

    for step in range(GENERATION_STEPS):
        step_start = time.time()
        power = get_power()
        power_readings.append(power)

        with torch.no_grad():
            outputs = model(
                current_ids,
                use_cache=USE_CACHE,
                attn_implementation=ATTN_IMPL if hasattr(model.config, "attn_implementation") else None,
            )

        logits = outputs.logits[0, -1, :]  # last token logits

        H  = compute_entropy(logits)
        B  = compute_branching(logits)
        C  = compute_topk_concentration(logits)
        D  = compute_kl(prev_logits, logits) if prev_logits is not None else 0.0
        dH = (H - trajectory[-1]["entropy"]) if trajectory else 0.0

        next_token = torch.argmax(logits, dim=-1, keepdim=True).unsqueeze(0)
        current_ids = torch.cat([current_ids, next_token], dim=-1)

        trajectory.append({
            "step": step,
            "power_watts": power,
            "entropy": H,
            "branching_factor": B,
            "kl_divergence": D,
            "delta_entropy": dH,
            "topk_concentration": C,
        })

        prev_logits = logits.detach()

    torch.cuda.synchronize()
    end_time = time.time()

    mean_power     = sum(r["power_watts"] for r in trajectory) / len(trajectory)
    total_energy   = mean_power * (end_time - start_time)
    mean_entropy   = sum(r["entropy"] for r in trajectory) / len(trajectory)
    mean_branching = sum(r["branching_factor"] for r in trajectory) / len(trajectory)
    mean_kl        = sum(r["kl_divergence"] for r in trajectory[1:]) / max(1, len(trajectory) - 1)
    mean_topk      = sum(r["topk_concentration"] for r in trajectory) / len(trajectory)

    aggregated = {
        "mean_power_watts":        mean_power,
        "total_energy_joules":     total_energy,
        "duration_seconds":        end_time - start_time,
        "mean_entropy":            mean_entropy,
        "mean_branching_factor":   mean_branching,
        "mean_kl_divergence":      mean_kl,
        "mean_topk_concentration": mean_topk,
    }

    return aggregated, trajectory


def main():
    results      = []
    all_steps    = []
    results_path = "/tmp/deception_is_cheap_v3_results.json"
    traj_path    = "/tmp/v3_trajectories.csv"

    traj_fields = [
        "model", "domain", "text_type", "step",
        "power_watts", "entropy", "branching_factor",
        "kl_divergence", "delta_entropy", "topk_concentration"
    ]

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"Protocol: use_cache={USE_CACHE}, attn={ATTN_IMPL}, steps={GENERATION_STEPS}")
        print(f"{'='*60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                attn_implementation=ATTN_IMPL,
            )
            model.eval()

            run_warmup(model, tokenizer)

            for domain, texts in CORPUS.items():
                for text_type, text_content in texts.items():
                    print(f"  Measuring: {domain} — {text_type}")

                    aggregated, trajectory = measure_text(model, tokenizer, text_content)

                    dci = DCI_PRESCORES[domain][text_type]

                    results.append({
                        "model":      model_name,
                        "domain":     domain,
                        "text_type":  text_type,
                        "dci_raw":    dci["raw"],
                        "dci_eff":    dci["eff"],
                        "metrics":    aggregated,
                    })

                    for row in trajectory:
                        all_steps.append({
                            "model":     model_name,
                            "domain":    domain,
                            "text_type": text_type,
                            **row
                        })

                    print(
                        f"    Power: {aggregated['mean_power_watts']:.1f}W  "
                        f"B: {aggregated['mean_branching_factor']:.2f}  "
                        f"H: {aggregated['mean_entropy']:.2f}  "
                        f"KL: {aggregated['mean_kl_divergence']:.2f}  "
                        f"DCI: {dci['raw']:.4f}"
                    )

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")

    # Save aggregated results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save full trajectories
    with open(traj_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=traj_fields)
        writer.writeheader()
        writer.writerows(all_steps)
    print(f"Trajectories saved to {traj_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY — Mean Power by Model and Text Type")
    print("="*80)
    print(f"{'Model':<16} {'Domain':<24} {'Fraud':>8} {'Compl':>8} {'Hedged':>8} {'Certain':>8}")
    print("-"*80)

    from collections import defaultdict
    table = defaultdict(dict)
    for r in results:
        table[(r["model"], r["domain"])][r["text_type"]] = r["metrics"]["mean_power_watts"]

    for m in MODELS:
        for d in CORPUS:
            row = table.get((m, d), {})
            print(
                f"{m:<16} {d:<24} "
                f"{row.get('Type_1_Fraudulent', 0):>8.1f} "
                f"{row.get('Type_2_Compliant', 0):>8.1f} "
                f"{row.get('Type_3_Hedged_Honest', 0):>8.1f} "
                f"{row.get('Type_4_Certain_Honest', 0):>8.1f}"
            )

    print(f"\nExperiment complete.")
    print(f"Copy files before terminating pod:")
    print(f"  cp {results_path} .")
    print(f"  cp {traj_path} .")


if __name__ == "__main__":
    main()