"""Fine-tune v6 (v3 + majority source coding) from the v3 checkpoint.

The zero-training graft fails because the learned modules were trained on iid
bits; this lets them adapt to the redundant input statistics with a small
number of steps at low LR. Objective/protocol identical to research.train.

Usage: .venv/bin/python analysis/finetune_v6.py --m 3 --steps 15000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

sys_path = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(sys_path))

import os
from research.data import Channels, load_splits  # noqa: E402
from research.evaluate import evaluate, load_weights  # noqa: E402
from research.link import B_MAX, Link, load_design  # noqa: E402
from research.objective import objective  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default="3")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.environ["V6_M"] = args.m
    # reimport design after env var set (module-level read)
    design = load_design(sys_path / "models" / "modelDesign_v6.py")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seed = 20260918
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    channels = Channels(sys_path / "data_train" / "H_train.npz")
    parts, _ = load_splits(sys_path / "splits" / "random_seed20260918.npz", channels)

    out = Path(args.out or f"experiments/v6_finetune_m{args.m}")
    out.mkdir(parents=True, exist_ok=True)

    link = Link(design).to(device)
    # warm start from the v3 checkpoint (identical architecture)
    load_weights(link, sys_path / "experiments" / "v3_mps_seed20260918" / "best", device)

    optimizer = torch.optim.AdamW(link.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps,
                                                           eta_min=args.lr * 0.1)
    best = -float("inf")
    for step in range(1, args.steps + 1):
        link.train()
        batch = args.batch_size
        idx = rng.choice(parts["train"], size=batch, replace=False)
        h = torch.from_numpy(channels.take(idx)).to(device)
        snr = -20 + 40 * torch.rand((2, batch), device=device, generator=generator)
        bits = [torch.randint(0, 2, (batch, B_MAX), device=device,
                              generator=generator).float() for _ in range(2)]
        logits, lengths, _ = link(h, bits, snr, generator, collect_aux=True)
        ramp = min(1.0, step / args.warmup)
        loss, stats = objective(bits, logits, lengths,
                                score_weight=0.1 * ramp, tail_weight=0.03 * ramp)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(link.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 500 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), **stats}
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _ = evaluate(link, channels, parts["val"], samples=args.val_samples,
                                  repeats=2, seed=314159, batch_size=16, bootstrap=0)
            with (out / "validation.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")
            print(json.dumps({"step": step, "val_score": metrics["score"],
                              "eff": metrics["efficiency"], "p10": metrics["fairness_p10"]}),
                  flush=True)
            if metrics["score"] > best:
                best = metrics["score"]
                for name in ("encoder", "transmitter", "receiver"):
                    torch.save(getattr(link, name).state_dict(), out / "best" / f"{name}.pth")
                (out / "best.json").write_text(json.dumps({"step": step, **metrics}, indent=2))
            for name in ("encoder", "transmitter", "receiver"):
                torch.save(getattr(link, name).state_dict(), out / "last" / f"{name}.pth")
    print(json.dumps({"done": True, "best": best}))


if __name__ == "__main__":
    main()
