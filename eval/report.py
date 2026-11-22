"""Summarise evaluation runs: pass rates per condition, per task, failure taxonomy.

    python eval/report.py results/<run>.jsonl [more runs ...] [--out results/report]

Writes <out>/summary.md, <out>/pass_rates.png, <out>/per_task.png, <out>/failures.png.
"""
import argparse
import collections
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["baseline", "context", "context_feedback"]
LABEL = {"baseline": "baseline", "context": "+ context pack", "context_feedback": "+ context + compiler feedback"}
COLOR = {"baseline": "#9b2226", "context": "#ee9b00", "context_feedback": "#0a9396"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1 - k / i
    return 1 - prod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/report")
    a = ap.parse_args()
    rows = load(a.runs)
    os.makedirs(a.out, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]

    md = ["# Dx code-generation benchmark — results", ""]
    for model in models:
        R = [r for r in rows if r["model"] == model]
        md += [f"## {model}", "", "| condition | pass@1 | 95% CI | pass@3 | first-try pass | compiles | mean attempts | speed vs reference (median) | output tokens / sample |",
               "|---|---|---|---|---|---|---|---|---|"]
        for c in conds:
            S = [r for r in R if r["condition"] == c]
            if not S:
                continue
            k = sum(r["passed"] for r in S)
            lo, hi = wilson(k, len(S))
            p3 = []
            for t in tasks:
                T = [r for r in S if r["task"] == t]
                if len(T) >= 3:
                    p3.append(pass_at_k(len(T), sum(r["passed"] for r in T), 3))
            speeds = sorted(r["speed_vs_ref"] for r in S if r.get("speed_vs_ref"))
            med = f"{speeds[len(speeds) // 2]:.2f}x" if speeds else "–"
            p3s = f"{sum(p3) / len(p3):.0%}" if p3 else "–"
            md.append(f"| {LABEL[c]} | **{k / len(S):.0%}** ({k}/{len(S)}) | {lo:.0%}–{hi:.0%} | {p3s} | "
                      f"{sum(r['first_try_passed'] for r in S) / len(S):.0%} | "
                      f"{sum(r['compiled'] for r in S) / len(S):.0%} | {sum(r['attempts_used'] for r in S) / len(S):.2f} | "
                      f"{med} | {sum(r['tokens_out'] for r in S) / len(S):.0f} |")
        md += ["", "### Pass rate per task", "", "| task | " + " | ".join(LABEL[c] for c in conds) + " |",
               "|---|" + "---|" * len(conds)]
        for t in tasks:
            cells = []
            for c in conds:
                T = [r for r in R if r["task"] == t and r["condition"] == c]
                cells.append(f"{sum(r['passed'] for r in T)}/{len(T)}" if T else "–")
            md.append(f"| `{t}` | " + " | ".join(cells) + " |")
        md += ["", "### Failure categories (first attempt)", "",
               "| category | " + " | ".join(LABEL[c] for c in conds) + " |", "|---|" + "---|" * len(conds)]
        cats = sorted({r["first_category"] for r in R})
        for cat in cats:
            md.append(f"| {cat} | " + " | ".join(
                str(sum(1 for r in R if r["condition"] == c and r["first_category"] == cat)) for c in conds) + " |")
        md.append("")

        # ---- plots ----
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, c in enumerate(conds):
            S = [r for r in R if r["condition"] == c]
            for j, lib in enumerate(["cublasdx", "cufftdx", "all"]):
                L = S if lib == "all" else [r for r in S if r["library"] == lib]
                if not L:
                    continue
                k = sum(r["passed"] for r in L)
                lo, hi = wilson(k, len(L))
                x = j + (i - (len(conds) - 1) / 2) * 0.26
                ax.bar(x, k / len(L), width=0.24, color=COLOR[c], label=LABEL[c] if j == 0 else None)
                ax.errorbar(x, k / len(L), yerr=[[k / len(L) - lo], [hi - k / len(L)]], color="#222", capsize=3, lw=1)
        ax.set_xticks([0, 1, 2], ["cuBLASDx tasks", "cuFFTDx tasks", "all tasks"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("pass rate (compiles + correct + uses Dx)")
        ax.set_title(f"{model}: effect of the context pack and compiler feedback")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"pass_rates_{model}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 0.45 * len(tasks) + 1.2))
        grid = [[(sum(r["passed"] for r in R if r["task"] == t and r["condition"] == c) /
                  max(1, len([r for r in R if r["task"] == t and r["condition"] == c]))) for c in conds] for t in tasks]
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conds)), [LABEL[c] for c in conds], fontsize=8)
        ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
        for i in range(len(tasks)):
            for j in range(len(conds)):
                ax.text(j, i, f"{grid[i][j]:.0%}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"{model}: pass rate per task")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"per_task_{model}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        cats = sorted({r["first_category"] for r in R}, key=lambda s: (s != "pass", s))
        bottom = [0] * len(conds)
        cmap = plt.get_cmap("tab20")
        for ci, cat in enumerate(cats):
            vals = [sum(1 for r in R if r["condition"] == c and r["first_category"] == cat) /
                    max(1, sum(1 for r in R if r["condition"] == c)) for c in conds]
            ax.bar([LABEL[c] for c in conds], vals, bottom=bottom, label=cat,
                   color="#2a9d8f" if cat == "pass" else cmap(ci % 20))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("share of first attempts")
        ax.set_title(f"{model}: what goes wrong on the first attempt")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"failures_{model}.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("\nwritten to", a.out)


if __name__ == "__main__":
    main()
