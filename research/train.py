"""Future training entrypoint. Does nothing without the explicit --run-training flag."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import Channels, load_splits
from .evaluate import evaluate
from .link import B_MAX, Link, load_design
from .objective import objective


def save_weights(link, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("encoder", "transmitter", "receiver"):
        torch.save(getattr(link, name).state_dict(), directory / (name + ".pth"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--splits")
    parser.add_argument("--out")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not args.run_training:
        print(json.dumps({"training_started": False, "config": config,
                          "message": "Plan only. To train later supply --data, --splits, --out and --run-training."},
                         ensure_ascii=False, indent=2))
        return
    if not all((args.data, args.splits, args.out)):
        parser.error("Actual training requires --data, --splits and a new --out directory")
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
    link = Link(load_design(source)).to(args.device)
    optimizer = torch.optim.AdamW(link.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"],
                                                          eta_min=config["lr"] * 0.1)
    best = -float("inf")
    for step in range(1, config["steps"] + 1):
        link.train()
        batch = config["batch_size"]
        indices = rng.choice(parts["train"], size=batch, replace=False)
        h = torch.from_numpy(channels.take(indices)).to(args.device)
        snr = -20 + 40 * torch.rand((2, batch), device=args.device, generator=generator)
        bits = [torch.randint(0, 2, (batch, B_MAX), device=args.device, generator=generator).float()
                for _ in range(2)]
        logits, lengths, aux = link(h, bits, snr, generator, collect_aux=True)
        ramp = min(1.0, max(0.0, (step - config["warmup_steps"]) / max(1, config["warmup_steps"])))
        loss, stats = objective(bits, logits, lengths,
                                score_weight=ramp * config.get("score_weight", 0.1),
                                tail_weight=ramp * config.get("tail_weight", 0.03))
        csi_weight = config.get("csi_weight", 0.01)
        if csi_weight:
            estimate = aux["h_hat"]
            if estimate.shape != h.shape:
                raise ValueError("Auxiliary h_hat must have shape [B,2,2,16,144]")
            nmse = (estimate - h).abs().square().sum((2, 3, 4)) / h.abs().square().sum((2, 3, 4)).clamp_min(1e-8)
            # This is a weak training-only regularizer, never a claim that low-SNR CSI is recoverable.
            csi_loss = nmse.mean()
            loss = loss + csi_weight * csi_loss
            stats["csi_nmse"] = float(csi_loss.detach())
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
            torch.save({"step": step, "model": link.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                        "link_rng": generator.get_state(), "numpy_rng": rng.bit_generator.state,
                        "config": config}, out / "training_state.pt")
            if metrics["score"] > best:
                best = metrics["score"]
                save_weights(link, out / "best")
                (out / "best.json").write_text(json.dumps({"step": step, **metrics}, indent=2) + "\n")
            print(json.dumps({"step": step, "validation_score": metrics["score"], "best": best}), flush=True)


if __name__ == "__main__":
    main()
