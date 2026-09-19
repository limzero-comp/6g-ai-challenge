"""Train v8 (pilot-anchored combined model) with chained warm start + EMA.

Follows the external notes' training engineering: warm-start every compatible
weight from the v3 checkpoint (embed columns 0-78 copied, new columns small
random), cosine LR, EMA(0.999) evaluated alongside the raw weights, best
checkpoint tracked per track.
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


def warm_start_from_v3(link, v3_dir, device):
    """Copy every compatible v3 weight; expand the receiver embed (first 79
    columns copied, the 27 new physics columns small-random)."""
    sd = {name: torch.load(Path(v3_dir) / f"{name}.pth", map_location=device,
                           weights_only=True)
          for name in ("encoder", "transmitter", "receiver")}
    link.encoder.load_state_dict(sd["encoder"])
    tx_new, tx_old = link.transmitter.state_dict(), sd["transmitter"]
    for key, value in tx_old.items():
        if key in tx_new and tx_new[key].shape == value.shape:
            tx_new[key] = value
    link.transmitter.load_state_dict(tx_new)
    rx_new, rx_old = link.receiver.state_dict(), sd["receiver"]
    for key, value in rx_old.items():
        if key in rx_new and rx_new[key].shape == value.shape:
            rx_new[key] = value
    embed_new = rx_new["embed.weight"]
    embed_old = rx_old["embed.weight"]                      # [96, 79]
    gen = torch.Generator().manual_seed(20260919)
    embed_new[:, :embed_old.shape[1]] = embed_old
    embed_new[:, embed_old.shape[1]:] = torch.randn(
        embed_new.shape[0], embed_new.shape[1] - embed_old.shape[1],
        generator=gen) * 0.01
    rx_new["embed.weight"] = embed_new
    link.receiver.load_state_dict(rx_new)


class Ema:
    def __init__(self, link, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in link.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, link):
        for key, value in link.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self, link):
        backup = {k: v.detach().clone() for k, v in link.state_dict().items()
                  if k in self.shadow}
        link.load_state_dict(self.shadow, strict=False)
        return backup

    @torch.no_grad()
    def restore(self, link, backup):
        link.load_state_dict(backup, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v8.json")
    parser.add_argument("--data", default="data_train/H_train.npz")
    parser.add_argument("--splits", default="splits/random_seed20260918.npz")
    parser.add_argument("--warm", default="experiments/v3_mps_seed20260918/best")
    parser.add_argument("--out", default="experiments/v8_mps_seed20260919")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
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
                "numpy": np.__version__, "device": args.device, "warm_start": args.warm,
                "design_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    link = Link(load_design(source)).to(args.device)
    warm_start_from_v3(link, args.warm, args.device)
    ema = Ema(link, decay=config.get("ema_decay", 0.999))

    optimizer = torch.optim.AdamW(link.parameters(), lr=config["lr"],
                                  weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"],
                                                           eta_min=config["lr"] * 0.1)
    best_raw, best_ema = -float("inf"), -float("inf")
    for step in range(1, config["steps"] + 1):
        link.train()
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
            target = h
            if "user_perm" in aux:
                idx = aux["user_perm"].reshape(h.shape[0], 2, 1, 1, 1).expand(-1, -1, *h.shape[2:])
                target = torch.gather(h, 1, idx)
            nmse = (estimate - target).abs().square().sum((2, 3, 4)) \
                / target.abs().square().sum((2, 3, 4)).clamp_min(1e-8)
            loss = loss + csi_weight * nmse.mean()
            stats["csi_nmse"] = float(nmse.mean().detach())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(link.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        ema.update(link)
        if step == 1 or step % config.get("log_every", 100) == 0:
            record = {"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), **stats}
            with (out / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % config["eval_every"] == 0 or step == config["steps"]:
            # raw track
            metrics_raw, _ = evaluate(link, channels, parts["val"], samples=config["val_samples"],
                                      repeats=config.get("val_repeats", 2), seed=314159,
                                      batch_size=config.get("eval_batch_size", 16), bootstrap=0)
            # EMA track
            backup = ema.swap(link)
            metrics_ema, _ = evaluate(link, channels, parts["val"], samples=config["val_samples"],
                                      repeats=config.get("val_repeats", 2), seed=314159,
                                      batch_size=config.get("eval_batch_size", 16), bootstrap=0)
            ema.restore(link, backup)
            with (out / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "raw": metrics_raw, "ema": metrics_ema}) + "\n")
            save_weights(link, out / "last")
            if metrics_ema["score"] > best_ema:
                best_ema = metrics_ema["score"]
                backup = ema.swap(link)
                save_weights(link, out / "best_ema")
                ema.restore(link, backup)
                (out / "best_ema.json").write_text(json.dumps({"step": step, **metrics_ema}, indent=2))
            if metrics_raw["score"] > best_raw:
                best_raw = metrics_raw["score"]
                save_weights(link, out / "best")
                (out / "best.json").write_text(json.dumps({"step": step, **metrics_raw}, indent=2))
            print(json.dumps({"step": step, "val_raw": metrics_raw["score"],
                              "val_ema": metrics_ema["score"],
                              "best_raw": best_raw, "best_ema": best_ema}), flush=True)


if __name__ == "__main__":
    main()
