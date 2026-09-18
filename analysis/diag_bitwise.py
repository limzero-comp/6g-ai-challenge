"""Experiment 0 (GAP_TO_70): per-block, per-source-bit diagnostics for v3/v4.

Questions this answers (per reports/GAP_TO_70.md section 6, row 0):
- Is the high-SNR error floor mainly present when the OTHER user's feedback
  is poor? (2D own-SNR x other-SNR BER grid)
- For v4: are the dedicated pilot REs / rate-ladder-masked source positions
  abnormal? (per-source-position BER)
- Are v4's transmitted low-SNR prefix bits actually decodable? (prefix accuracy)

Usage (repo root):
  .venv/bin/python analysis/diag_bitwise.py \
    --design experiments/v3_mps_seed20260918/modelDesign.py \
    --weights experiments/v3_mps_seed20260918/best \
    --tag v3 [--rows 1000] [--repeats 2] [--batch-size 16] [--device mps]
Writes experiments/<tag>_diag_bitwise.json and a PNG heatmap into reports/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.data import Channels, load_splits  # noqa: E402
from research.evaluate import load_weights  # noqa: E402
from research.link import B_MAX, Link, load_design  # noqa: E402


def snr_bucket(snr):
    return int(np.floor(np.clip(snr, -20.0, 19.999) / 5.0))  # 0..7 -> [-20,-15)...[15,20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data_train/H_train.npz")
    parser.add_argument("--splits", default="splits/random_seed20260918.npz")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    data = Channels(args.data)
    parts, _ = load_splits(args.splits, data)
    link = Link(load_design(args.design)).to(args.device)
    load_weights(link, args.weights, args.device)
    link.eval()

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(parts[args.split], size=min(args.rows, len(parts[args.split])),
                      replace=False)

    err = np.zeros((len(rows), 2, B_MAX), dtype=bool)  # [block, ue, bit]
    snr_own = np.zeros((len(rows), 2), dtype=np.float32)
    snr_other = np.zeros((len(rows), 2), dtype=np.float32)
    lengths = np.zeros((len(rows), 2), dtype=np.int64)
    prefix_err = np.zeros((len(rows), 2), dtype=np.int64)

    blocks_done = 0
    with torch.no_grad():
        for repeat in range(args.repeats):
            snr_values = rng.uniform(-20, 20, (2, len(rows))).astype(np.float32)
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start + args.batch_size]
                h = torch.from_numpy(data.take(chunk)).to(args.device)
                snr = torch.from_numpy(snr_values[:, start:start + len(chunk)]).to(args.device)
                generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + 7919 * repeat + start)
                bits = [torch.randint(0, 2, (len(chunk), B_MAX), device=args.device,
                                      generator=generator).float() for _ in range(2)]
                logits, counts, _ = link(h, bits, snr, generator, collect_aux=True)
                for ue in range(2):
                    hard = (logits[ue] >= 0).cpu().numpy()
                    true = bits[ue].cpu().numpy().astype(bool)
                    n = len(chunk)
                    err[start:start + n, ue] = hard != true[:, :B_MAX]
                    snr_own[start:start + n, ue] = snr_values[ue, start:start + n]
                    snr_other[start:start + n, ue] = snr_values[1 - ue, start:start + n]
                    cnt = counts[ue].cpu().numpy()
                    lengths[start:start + n, ue] = cnt
                    pe = (err[start:start + n, ue] & np.arange(B_MAX)[None, :] < cnt[:, None]).sum(1)
                    prefix_err[start:start + n, ue] = pe
                blocks_done += len(chunk)
                print(f"repeat {repeat}: {blocks_done}/{len(rows)} blocks", flush=True)

    out = {"design": args.design, "weights": args.weights, "rows": len(rows),
           "repeats": args.repeats, "seed": args.seed,
           "ber_overall": float(err.mean()),
           "ber_by_own_bucket": {}, "ber2d_own_other": {},
           "prefix_accuracy_by_own_bucket": {}, "ber_by_position": {}}

    # BER by own-SNR bucket (5 dB)
    table = []
    for b in range(8):
        m = (np.floor((snr_own + 20.0) / 5.0).astype(int) == b)
        if m.any():
            e = float(err[m].mean())
            table.append(e)
            out["ber_by_own_bucket"][f"[{-20 + 5 * b},{-15 + 5 * b})"] = {
                "ber": e, "count": int(m.sum())}
        else:
            table.append(float("nan"))

    # 2D grid: own bucket x other bucket
    grid = {}
    for bo in range(8):
        for bt in range(8):
            m = ((np.floor((snr_own + 20.0) / 5.0).astype(int) == bo)
                 & (np.floor((snr_other + 20.0) / 5.0).astype(int) == bt))
            if m.sum() >= 50:
                grid[f"own[{bo}]|other[{bt}]"] = {
                    "ber": float(err[m].mean()), "count": int(m.sum())}
    out["ber2d_own_other"] = grid

    # Prefix accuracy by own bucket (transmitted bits only)
    for b in range(8):
        sel = (np.floor((snr_own + 20.0) / 5.0).astype(int) == b)
        if not sel.any():
            continue
        idx = np.arange(B_MAX)[None, None, :] < lengths[..., None]
        m = sel[:, :, None] & idx
        tot = int(m.sum())
        if tot:
            bad = int((err & m).sum())
            out["prefix_accuracy_by_own_bucket"][f"[{-20 + 5 * b},{-15 + 5 * b})"] = {
                "accuracy": 1.0 - bad / tot, "count": tot}

    # Per-position BER averaged over everything (v4: reveals pilot/ladder effects)
    out["ber_by_position"] = err.mean(axis=(0, 1)).tolist()

    dest = Path(f"experiments/{args.tag}_diag_bitwise.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("ber_overall", "ber_by_own_bucket",
                                          "prefix_accuracy_by_own_bucket")}, indent=2))

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = np.full((8, 8), np.nan)
    for key, v in grid.items():
        bo, bt = int(key.split("own[")[1].split("]")[0]), int(key.split("other[")[1].split("]")[0])
        g[bo, bt] = v["ber"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(g * 100, cmap="viridis")
    ax.set_xticks(range(8), [f"[{-20+5*i}" for i in range(8)], rotation=45, fontsize=7)
    ax.set_yticks(range(8), [f"[{-20+5*i}" for i in range(8)], fontsize=7)
    ax.set_xlabel("other user SNR bucket (dB)"); ax.set_ylabel("own SNR bucket (dB)")
    ax.set_title(f"{args.tag}: source-bit BER (%) by own x other SNR")
    fig.colorbar(im, label="BER %")
    for i in range(8):
        for j in range(8):
            if not np.isnan(g[i, j]):
                ax.text(j, i, f"{g[i,j]*100:.1f}", ha="center", va="center",
                        color="w", fontsize=6)
    fig.tight_layout()
    fig.savefig(f"reports/{args.tag}_diag_heatmap.png", dpi=130)
    print(f"written: {dest} and reports/{args.tag}_diag_heatmap.png")


if __name__ == "__main__":
    main()
