"""Train v9 (v3 + receiver P0 combo: aux channel readout, y-RMS feature).

Identical to train_v8's loop except the auxiliary loss is the receiver's
channel readout: 0.02 x MSE(|aux_h_hat|, |h_own|) per UE (neural_rx's
double_readout recipe). Warm starts from any v3-lineage checkpoint.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v8 import Ema, save_weights, warm_start_from_v3  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v9.json")
    parser.add_argument("--data", default="data_train/H_train.npz")
    parser.add_argument("--splits", default="splits/random_seed20260918.npz")
    parser.add_argument("--warm", default="experiments/v3_long_mps/best_ema")
    parser.add_argument("--out", default="experiments/v9_mps_seed20260920")
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
    aux_weight = config.get("aux_chest_weight", 0.02)

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
        # neural_rx multiloss: BCE at every iteration (equal weight), so the
        # state is supervised from the first update onward.
        iter_llrs = list(link.receiver.iter_llrs)[:-1]     # last is `logits`
        for llr_it in iter_llrs:
            extra, _ = objective(bits, [llr_it], lengths, score_weight=0.0, tail_weight=0.0)
            loss = loss + extra / max(1, len(iter_llrs) + 1)
        csi_weight = config.get("csi_weight", 0.01)
        if csi_weight:
            estimate = aux["h_hat"]
            nmse = (estimate - h).abs().square().sum((2, 3, 4)) \
                / h.abs().square().sum((2, 3, 4)).clamp_min(1e-8)
            loss = loss + csi_weight * nmse.mean()
            stats["csi_nmse"] = float(nmse.mean().detach())
        if aux_weight:
            hists = list(link.receiver.iter_chests)   # per-iteration [B,RE,R*T*2] re/im
            link.receiver.iter_llrs, link.receiver.iter_chests = [], []
            # normalised complex target: h / rms(|h|) per sample, re/im layout
            # [B, RE, (R,T,2)] matching the readout ordering.
            h_own = h[:, 1]
            scale = h_own.abs().square().mean((1, 2, 3)).clamp_min(1e-10).sqrt()
            h_n = (h_own / scale[:, None, None, None])
            ref = torch.cat((h_n.real, h_n.imag), dim=-1).permute(0, 3, 1, 2) \
                .reshape(h.shape[0], 144, 64)
            chest_losses = [(c - ref).square().mean() for c in hists]
            stats["chest_nmse"] = float(chest_losses[-1].detach())
            loss = loss + aux_weight * sum(chest_losses) / len(chest_losses)
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
            metrics_raw, _ = evaluate(link, channels, parts["val"], samples=config["val_samples"],
                                      repeats=config.get("val_repeats", 2), seed=314159,
                                      batch_size=config.get("eval_batch_size", 16), bootstrap=0)
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
