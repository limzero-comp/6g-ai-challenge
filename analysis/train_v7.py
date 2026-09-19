"""Train v7 (replica-physics receiver) with per-step replica synchronisation.

Identical to research.train's loop except sync_replica() hard-copies the live
BS modules into the receiver's frozen replicas before every forward pass, so
the receiver's replica physics always matches the current transmitter.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.data import Channels, load_splits  # noqa: E402
from research.evaluate import evaluate  # noqa: E402
from research.link import B_MAX, Link, load_design  # noqa: E402
from research.objective import objective  # noqa: E402


def save_weights(link, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("encoder", "transmitter", "receiver"):
        torch.save(getattr(link, name).state_dict(), directory / (name + ".pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7.json")
    parser.add_argument("--data", default="data_train/H_train.npz")
    parser.add_argument("--splits", default="splits/random_seed20260918.npz")
    parser.add_argument("--out", default="experiments/v7_mps_seed20260919")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--steps-override", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if args.steps_override:
        config["steps"] = args.steps_override
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    seed = int(config.get("seed", 20260918))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=args.device).manual_seed(seed + 1)

    channels = Channels(args.data)
    parts, split_metadata = load_splits(args.splits, channels)
    source = Path(config["design"])
    shutil.copyfile(source, out / "modelDesign.py")
    manifest = {"config": config, "split": split_metadata, "torch": torch.__version__,
                "numpy": np.__version__, "device": args.device,
                "design_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    import models.modelDesign_v7 as v7mod  # for sync_replica
    link = Link(load_design(source)).to(args.device)

    optimizer = torch.optim.AdamW(link.parameters(), lr=config["lr"],
                                  weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"],
                                                           eta_min=config["lr"] * 0.1)
    best = -float("inf")
    for step in range(1, config["steps"] + 1):
        link.train()
        v7mod.sync_replica(link)                       # keep replica physics exact
        batch = config["batch_size"]
        indices = rng.choice(parts["train"], size=batch, replace=False)
        h = torch.from_numpy(channels.take(indices)).to(args.device)
        snr = -20 + 40 * torch.rand((2, batch), device=args.device, generator=generator)
        bits = [torch.randint(0, 2, (batch, B_MAX), device=args.device,
                              generator=generator).float() for _ in range(2)]
        logits, lengths, aux = link(h, bits, snr, generator, collect_aux=True)
        ramp = min(1.0, max(0.0, (step - config["warmup_steps"]) / max(1, config["warmup_steps"])))
        loss, stats = objective(bits, logits, lengths,
                                score_weight=ramp * config.get("score_weight", 0.1),
                                tail_weight=ramp * config.get("tail_weight", 0.03))
        csi_weight = config.get("csi_weight", 0.01)
        if csi_weight:
            estimate = aux["h_hat"]
            nmse = (estimate - h).abs().square().sum((2, 3, 4)) \
                / h.abs().square().sum((2, 3, 4)).clamp_min(1e-8)
            loss = loss + csi_weight * nmse.mean()
            stats["csi_nmse"] = float(nmse.mean().detach())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(link.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % config.get("log_every", 100) == 0:
            record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), **stats}
            with (out / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            metrics, _ = evaluate(link, channels, parts["val"], samples=config["val_samples"],
                                  repeats=config.get("val_repeats", 2), seed=314159,
                                  batch_size=config.get("eval_batch_size", 16), bootstrap=0)
            with (out / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **metrics}) + "\n")
            save_weights(link, out / "last")
            torch.save({"step": step, "model": link.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(), "link_rng": generator.get_state(),
                        "numpy_rng": rng.bit_generator.state, "config": config},
                       out / "training_state.pt")
            if metrics["score"] > best:
                best = metrics["score"]
                save_weights(link, out / "best")
                (out / "best.json").write_text(json.dumps({"step": step, **metrics}, indent=2) + "\n")
            print(json.dumps({"step": step, "validation_score": metrics["score"], "best": best}), flush=True)


if __name__ == "__main__":
    main()
