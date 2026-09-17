"""P0 ablation: trained-link evaluation under three feedback conditions.

Paired comparison on identical channel rows, bits, and noise realizations to
separate the CSI-feedback bottleneck from the receiver bottleneck:

- noisy:          exact training-time link (noisy UL feedback).
- noiseless:      UL noise is drawn (to keep the generator stream identical)
                  but not added, so the BS sees the perfect feedback symbol.
- zero_feedback:  the BS receives zeros; the CSI decoder shrinkage then falls
                  back to its statistical prior. The receiver still sees its
                  true H. This is the no-CSIT floor of the architecture.

Every mode draws the same noise tensors in the same order from a per-block
seeded generator, so bits and DL noise stay paired across modes; only the
addition of UL noise differs.

Usage (from repo root):
  python analysis/ablation_feedback.py \
    --design experiments/v1_mps_seed20260918/modelDesign.py \
    --weights experiments/v1_mps_seed20260918/best \
    --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
    --split val --samples 1000 --repeats 2 --batch-size 16 \
    --out experiments/v1_mps_seed20260918/ablation_feedback.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.data import Channels, load_splits  # noqa: E402
from research.evaluate import load_weights, summarize  # noqa: E402
from research.link import B_MAX, Link, load_design, per_user_score  # noqa: E402

MODES = ("noisy", "noiseless", "zero_feedback")


@torch.no_grad()
def run_mode(link, h, snr, mode, device, generator):
    """Score one batch in one feedback mode; mirrors research.link.forward."""
    bits = [torch.randint(0, 2, (h.shape[0], B_MAX), device=device,
                          generator=generator).float() for _ in range(2)]
    feedback = []
    for ue in range(2):
        u = link.encoder(h[:, ue], snr[ue])
        u = u / u.abs().square().mean(1, keepdim=True).sqrt()
        # Draw UL noise in every mode so downstream draws stay paired.
        sigma = 10.0 ** (-(snr[ue] - 10.0) / 20.0)
        noise = sigma[:, None] * torch.complex(
            torch.randn(u.shape, device=device, generator=generator),
            torch.randn(u.shape, device=device, generator=generator)) * (2.0 ** -0.5)
        if mode == "noisy":
            u = u + noise
        elif mode == "zero_feedback":
            u = torch.zeros_like(u)
        feedback.append(u)
    x, ctrl = link.transmitter(bits, feedback, snr)
    x = x / x.abs().square().sum(1, keepdim=True).mean(2, keepdim=True).sqrt()
    scores = []
    for ue in range(2):
        y = (h[:, ue] * x[:, None]).sum(2)
        dl_noise = 10.0 ** (-snr[ue, :, None, None] / 20.0) * torch.complex(
            torch.randn(y.shape, device=device, generator=generator),
            torch.randn(y.shape, device=device, generator=generator)) * (2.0 ** -0.5)
        y = y + dl_noise
        llr = link.receiver(y, h[:, ue], ctrl, snr[ue])
        scores.append(per_user_score(
            bits[ue], llr, torch.full((h.shape[0],), llr.shape[1],
                                      device=device, dtype=torch.long)))
    return torch.stack(scores, 1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = Channels(args.data)
    parts, metadata = load_splits(args.splits, data)
    link = Link(load_design(args.design)).to(args.device)
    load_weights(link, args.weights, args.device)
    link.eval()

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(parts[args.split], size=min(args.samples, len(parts[args.split])),
                      replace=False)

    per_mode, snr_records = {m: [] for m in MODES}, []
    for repeat in range(args.repeats):
        # One SNR draw per repeat, shared across modes for exact pairing.
        snr_values = rng.uniform(-20, 20, (2, len(rows))).astype(np.float32)
        snr_records.append(snr_values.T)
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            h = torch.from_numpy(data.take(chunk)).to(args.device)
            snr = torch.from_numpy(snr_values[:, start:start + len(chunk)]).to(args.device)
            for mode in MODES:
                generator = torch.Generator(device=args.device).manual_seed(
                    args.seed + 1000 * repeat + start)
                per_mode[mode].append(run_mode(link, h, snr, mode, args.device,
                                               generator))

    results = {"design": args.design, "weights": args.weights, "split": args.split,
               "split_metadata": metadata, "channel_rows": len(rows),
               "repeats": args.repeats, "seed": args.seed, "modes": {}}
    for mode in MODES:
        scores = np.concatenate(per_mode[mode], axis=0)   # [samples, 2]

        # Exact per-sample SNR: walk repeats × row-blocks in evaluation order.
        snr_per_sample = np.empty((scores.shape[0], 2), dtype=np.float32)
        idx = 0
        for r in range(args.repeats):
            for bstart in range(0, len(rows), args.batch_size):
                n = min(args.batch_size, len(rows) - bstart)
                snr_per_sample[idx:idx + n] = snr_records[r][bstart:bstart + n]
                idx += n
        assert idx == scores.shape[0]

        metrics = summarize(scores)
        binned = {}
        flat_scores = scores.reshape(-1)
        # scores and snr_per_sample share the [sample, ue] layout; one DL SNR per user.
        flat_snr = snr_per_sample.reshape(-1)
        for low in range(-20, 20, 5):
            mask = (flat_snr >= low) & (flat_snr < low + 5)
            binned["[%d,%d)" % (low, low + 5)] = (
                {**summarize(flat_scores[mask]), "count": int(mask.sum())}
                if mask.any() else None)
        results["modes"][mode] = {**metrics,
                                  "per_user": [summarize(scores[..., ue]) for ue in range(2)],
                                  "snr_bins_own_dl": binned}

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results["modes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
