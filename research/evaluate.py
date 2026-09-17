"""Evaluate saved candidates on a disjoint split with the exact hard-bit score."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Channels, load_splits
from .link import B_MAX, Link, load_design, per_user_score


def summarize(scores):
    scores = np.asarray(scores)
    efficiency = float(scores.mean())
    fairness = float(np.percentile(scores, 10))
    return {"efficiency": efficiency, "fairness_p10": fairness,
            "score": 0.7 * efficiency + 0.3 * fairness}


def clustered_interval(scores, seed, draws=300):
    # Shape [repeat, independent channel row, UE]. Preserve both UEs and repeats.
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.integers(scores.shape[1], size=scores.shape[1])
        estimates.append(list(summarize(scores[:, chosen]).values()))
    lower, upper = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {key: [float(lo), float(hi)] for key, lo, hi in
            zip(("efficiency", "fairness_p10", "score"), lower, upper)}


@torch.no_grad()
def evaluate(link, channels, indices, samples=2000, repeats=2, seed=271828,
             batch_size=1, bootstrap=300):
    if samples <= 0 or repeats <= 0 or batch_size <= 0:
        raise ValueError("Evaluation sizes must be positive")
    device = next(link.parameters()).device
    previous = link.training
    link.eval()
    rng = np.random.default_rng(seed)
    rows = rng.choice(indices, size=min(samples, len(indices)), replace=False)
    all_scores, all_snr, all_lengths = [], [], []
    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        snr_values = rng.uniform(-20, 20, (2, len(rows))).astype(np.float32)
        score_parts, length_parts = [], []
        for start in range(0, len(rows), batch_size):
            picked = rows[start:start + batch_size]
            h = torch.from_numpy(channels.take(picked)).to(device)
            snr = torch.from_numpy(snr_values[:, start:start + len(picked)]).to(device)
            bits = [torch.randint(0, 2, (len(picked), B_MAX), device=device,
                                  generator=generator).float() for _ in range(2)]
            logits, lengths, _ = link(h, bits, snr, generator)
            score_parts.append(torch.stack([per_user_score(b, p, n)
                                            for b, p, n in zip(bits, logits, lengths)], 1).cpu().numpy())
            length_parts.append(torch.stack(lengths, 1).cpu().numpy())
        all_scores.append(np.concatenate(score_parts))
        all_lengths.append(np.concatenate(length_parts))
        all_snr.append(snr_values.T)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    scores = np.stack(all_scores)
    snrs = np.stack(all_snr)
    lengths = np.stack(all_lengths)
    metrics = summarize(scores)
    metrics.update({"channel_rows": len(rows), "repeats": repeats, "seed": seed,
                    "batch_size": batch_size, "seconds": seconds,
                    "pairing_note": "Use identical rows, seed, repeats AND batch_size for paired model comparison",
                    "milliseconds_per_two_user_block": 1000 * seconds / (len(rows) * repeats),
                    "timing_note": "Includes data transfer/loading; hardware-specific, not platform timing",
                    "per_user": [summarize(scores[..., ue]) for ue in range(2)]})
    metrics["snr_bins"] = {}
    for low in range(-20, 20, 5):
        mask = (snrs >= low) & (snrs < low + 5)
        metrics["snr_bins"]["[%d,%d)" % (low, low + 5)] = (
            {**summarize(scores[mask]), "count": int(mask.sum())} if mask.any() else None)
    unique, count = np.unique(lengths, return_counts=True)
    metrics["prefix_lengths"] = dict(zip(map(str, unique.tolist()), count.tolist()))
    if bootstrap:
        metrics["cluster_bootstrap_95pct"] = clustered_interval(scores, seed + 1, bootstrap)
    link.train(previous)
    return metrics, {"scores": scores, "snr": snrs, "lengths": lengths, "rows": rows}


def load_weights(link, directory, device):
    for name in ("encoder", "transmitter", "receiver"):
        state = torch.load(Path(directory) / (name + ".pth"), map_location=device, weights_only=True)
        getattr(link, name).load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    data = Channels(args.data)
    parts, metadata = load_splits(args.splits, data)
    link = Link(load_design(args.design)).to(args.device)
    load_weights(link, args.weights, args.device)
    metrics, arrays = evaluate(link, data, parts[args.split], args.samples, args.repeats,
                               args.seed, args.batch_size)
    metrics.update({"split": args.split, "split_metadata": metadata, "design": args.design,
                    "weights": args.weights})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
