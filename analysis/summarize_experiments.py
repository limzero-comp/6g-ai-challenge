"""Summarize all experiment results into a markdown report + comparison chart.

Reads experiments/*/best.json, experiments/*/validation.jsonl and
experiments/*/eval_val.json, then writes reports/experiment_summary.md and
reports/fig_validation_curves.png.

Usage: python analysis/summarize_experiments.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "reports"


def load_validation(run):
    rows = []
    path = EXP / run / "validation.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            rows.append(json.loads(line))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    runs = sorted(p.name for p in EXP.iterdir() if p.is_dir() and (p / "validation.jsonl").exists())
    if not runs:
        raise SystemExit("no completed runs found under experiments/")

    lines = ["# 实验汇总（mac-mini-dev，MPS 训练）\n"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    for run in runs:
        val = load_validation(run)
        if not val:
            continue
        steps = [r["step"] for r in val]
        score = [r["score"] for r in val]
        eff = [r["efficiency"] for r in val]
        best = max(val, key=lambda r: r["score"])
        axes[0].plot(steps, score, marker="o", ms=3, label=run)
        axes[1].plot(steps, eff, marker="o", ms=3, label=run)

        eval_path = EXP / run / "eval_val.json"
        eval_note = ""
        if eval_path.exists():
            e = json.loads(eval_path.read_text())
            ci = e.get("cluster_bootstrap_95pct", {}).get("score")
            ci_note = (f"（95% CI {ci[0]:.2f}–{ci[1]:.2f}）" if ci else "")
            eval_note = (f"\n\n正式评测（val，{e['channel_rows']}行×{e['repeats']}次，batch={e['batch_size']}）："
                         f"效率 {e['efficiency']:.2f}，公平 p10 {e['fairness_p10']:.2f}，"
                         f"最终 {e['score']:.2f}{ci_note}。")
        lines.append(f"## {run}\n\n- 训练中最佳验证：step {best['step']}，最终分 {best['score']:.2f}"
                     f"（效率 {best['efficiency']:.2f} / p10 {best['fairness_p10']:.2f}）{eval_note}")
        bins = best.get("snr_bins") or {}
        if bins:
            lines.append("\n| 下行SNR (dB) | 效率分 | 最终分 | 样本数 |\n|---|---:|---:|---:|")
            for k, v in bins.items():
                if v:
                    lines.append(f"| {k} | {v['efficiency']:.2f} | {v['score']:.2f} | {v['count']} |")
        abl_path = EXP / run / "ablation_feedback.json"
        if abl_path.exists():
            a = json.loads(abl_path.read_text())["modes"]
            lines.append("\n### P0 反馈消融（同信道/比特/噪声配对）\n\n| 模式 | 效率分 | 公平p10 | 最终分 |")
            lines.append("|---|---:|---:|---:|")
            names = {"noisy": "真实含噪反馈", "noiseless": "无噪反馈（上限）",
                     "zero_feedback": "无反馈（统计先验下限）"}
            for m, label in names.items():
                if m in a:
                    v = a[m]
                    lines.append(f"| {label} | {v['efficiency']:.2f} | {v['fairness_p10']:.2f} | {v['score']:.2f} |")
        lines.append("")

    axes[0].set_xlabel("step"); axes[0].set_ylabel("validation score (0.7e+0.3f)")
    axes[0].set_title("validation score"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    axes[1].set_xlabel("step"); axes[1].set_ylabel("efficiency")
    axes[1].set_title("validation efficiency"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_validation_curves.png", dpi=130)
    print("\n".join(lines))
    (OUT / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {OUT / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
