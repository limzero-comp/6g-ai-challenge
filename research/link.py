"""Competition-compatible noisy link. No unobservable CSI is passed to a receiver."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

B_MAX = 1152


def load_design(path):
    spec = importlib.util.spec_from_file_location("candidate_design", str(Path(path).resolve()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complex_noise(shape, device, generator=None):
    re = torch.randn(shape, device=device, generator=generator)
    im = torch.randn(shape, device=device, generator=generator)
    return torch.complex(re, im) * (2.0 ** -0.5)


def normalize_feedback(u):
    energy = u.abs().square().mean(1, keepdim=True)
    if not bool(torch.isfinite(u).all() and (energy > 0).all()):
        raise ValueError("Feedback must be finite and have positive per-sample energy")
    return u / energy.sqrt()


def normalize_downlink(x):
    energy = x.abs().square().sum(1, keepdim=True).mean(2, keepdim=True)
    if not bool(torch.isfinite(x).all() and (energy > 0).all()):
        raise ValueError("Waveform must be finite and have positive per-sample energy")
    return x / energy.sqrt()


def bit_mask(lengths, width=B_MAX):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def per_user_score(bits, logits, lengths):
    mask = bit_mask(lengths, logits.shape[1])
    correct = ((logits >= 0) == bits[:, :logits.shape[1]].bool()) & mask
    return 100.0 * (correct.sum(1) + 0.5 * (B_MAX - lengths)) / B_MAX


class Link(nn.Module):
    def __init__(self, design):
        super().__init__()
        self.encoder = design.Encoder()
        self.transmitter = design.Transmitter()
        self.receiver = design.Receiver()

    def forward(self, h, bits, snr, generator=None, collect_aux=False):
        # h [B,UE,RX,TX,SC]; snr [UE,B]. All SNR values are dB.
        batch = h.shape[0]
        if h.shape[1:] != (2, 2, 16, 144) or snr.shape != (2, batch):
            raise ValueError("Channel/SNR shape mismatch")
        feedback = []
        for ue in range(2):
            u = self.encoder(h[:, ue], snr[ue])
            if u.shape != (batch, 96) or u.dtype != torch.complex64:
                raise ValueError("Encoder contract mismatch")
            u = normalize_feedback(u)
            sigma = 10.0 ** (-(snr[ue] - 10.0) / 20.0)
            feedback.append(u + sigma[:, None] * complex_noise(u.shape, u.device, generator))
        aux = {}
        if collect_aux:
            x, ctrl, aux = self.transmitter(bits, feedback, snr, return_aux=True)
        else:
            x, ctrl = self.transmitter(bits, feedback, snr)
        if x.shape != (batch, 16, 144) or x.dtype != torch.complex64:
            raise ValueError("Transmitter waveform contract mismatch")
        if ctrl.shape != (batch, 5) or not bool(((ctrl == 0) | (ctrl == 1)).all()):
            raise ValueError("Control bits must be binary [batch,5]")
        x = normalize_downlink(x)
        logits, lengths = [], []
        for ue in range(2):
            y = (h[:, ue] * x[:, None]).sum(2)
            y = y + 10.0 ** (-snr[ue, :, None, None] / 20.0) * complex_noise(y.shape, y.device, generator)
            if hasattr(self.receiver, "forward_full"):
                llr = self.receiver.forward_full(y, h[:, ue], ctrl, snr[ue])
                count = self.receiver.valid_lengths(snr[ue], ctrl)
            else:
                llr = self.receiver(y, h[:, ue], ctrl, snr[ue])
                count = torch.full((batch,), llr.shape[1], device=y.device, dtype=torch.long)
            if (llr.ndim != 2 or llr.shape[0] != batch or not 0 < llr.shape[1] <= B_MAX
                    or llr.dtype != torch.float32 or not bool(torch.isfinite(llr).all())):
                raise ValueError("Receiver LLR contract mismatch")
            if not bool(((count > 0) & (count <= llr.shape[1])).all()):
                raise ValueError("Invalid prefix lengths")
            logits.append(llr)
            lengths.append(count)
        return logits, lengths, aux
