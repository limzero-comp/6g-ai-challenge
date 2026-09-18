"""v5: deterministic interpretable link for GAP_TO_70 experiments 1+2.

Fixed statistical beams + constant-modulus Gray-PSK + short-block lossy source
coding. Training-free; every quantity the receiver uses is derivable from its
official inputs (own H, own SNR, 5-bit control word) plus public tables.

Role association without UE identity (GAP_TO_70 section 4.1):
  q_own = 32-level quantisation of own SNR; control word t = min(q0, q1) over
  the two users (31 never emitted: tie keeps t = q). q_own == t -> LOW role,
  own stream on beam 0; q_own > t -> HIGH role, own stream on beam 1. When the
  two users tie, both take LOW and collide on beam 0 (both lose; measured and
  reported separately, expected 3.125% of pairs under uniform SNR).

Power/energy budget: each user transmits one unit-modulus PSK symbol per RE at
stream power 0.5 on its assigned beam, so the waveform has exactly unit energy
per RE and the official per-sample downlink normalisation is the identity. The
receiver forms its 2x2 effective channel exactly (own H times the public beam
table) and applies per-RE MMSE with the exact noise variance, then max-log
demaps, combines repetition slots by LLR summation, and expands through the
selected source-coding table to 1152 source positions.

Source policies (public, keyed by own SNR on both sides; V5_POLICY env selects
one globally, or "ladder" for the per-bucket table BUCKETS):
  prefix_<mod>[_r<k>]   first K=144*log2M/rep source bits verbatim
  maj<m>_<mod>          majority of every m source bits (lossy, all 1152 out)
"""
import math
import os

import torch
from torch import nn

NUM_UE, NUM_RX, NUM_TX, NUM_RE = 2, 2, 16, 144
NUM_BITS = 1152
SNR_UL_GAP_DB = 10.0
STREAM_POWER = 0.5
BEAM_OF_ROLE = (0, 1)        # LOW role -> DFT col 0, HIGH role -> DFT col 1

BUCKETS = [
    (-20.0, "maj9_qpsk"),
    (-15.0, "maj9_qpsk"),
    (-10.0, "maj3_8psk"),
    (-5.0, "maj3_8psk"),
    (0.0, "prefix_qpsk"),
    (5.0, "prefix_8psk"),
    (10.0, "prefix_16psk"),
    (15.0, "prefix_16psk"),
]
_MOD_BITS = {"qpsk": 2, "8psk": 3, "16psk": 4}
_POLICY_NAME = os.environ.get("V5_POLICY", "prefix_16psk")


def _parse_policy(name):
    parts = name.split("_")
    rep = 1
    if parts[-1].startswith("r") and parts[-1][1:].isdigit():
        rep = int(parts[-1][1:])
        parts = parts[:-1]
    mod = parts[-1]
    if mod not in _MOD_BITS:
        raise ValueError(f"unknown modulation {mod}")
    if parts[0] == "prefix":
        return {"mode": "prefix", "m": 1, "mod": mod, "rep": rep}
    if parts[0].startswith("maj") and parts[0][3:].isdigit():
        return {"mode": "maj", "m": int(parts[0][3:]), "mod": mod, "rep": rep}
    raise ValueError(f"unknown policy {name}")


def _policy_for_snr(snr):
    """Public policy per sample; snr [B] -> list of policy dicts.

    V5_POLICY selects a single global policy; "ladder" uses the per-bucket
    table. Both sides resolve identically from own SNR only.
    """
    if _POLICY_NAME != "ladder":
        pol = _parse_policy(_POLICY_NAME)
        return [dict(pol) for _ in range(len(snr))]
    names = []
    for s in snr.tolist():
        chosen = BUCKETS[-1][1]
        for lo, nm in BUCKETS:
            if s >= lo:
                chosen = nm
        names.append(chosen)
    return [_parse_policy(n) for n in names]


def _policy_key(pol):
    return (pol["mode"], pol["m"], pol["mod"], pol["rep"])


def _slot_count(pol):
    return NUM_RE * _MOD_BITS[pol["mod"]] // pol["rep"]


def _info_count(pol):
    return _slot_count(pol) if pol["mode"] == "prefix" else NUM_BITS // pol["m"]


def _quant32(snr):
    return torch.clamp(((snr + 20.0) / 1.25).floor(), 0, 31)


def _ctrl_bits(snr):
    """Official 5-bit word: t = min(q0, q1), LSB first; ties keep t = q."""
    q = _quant32(snr)
    t = q.min(dim=0).values
    return ((t.long()[:, None] >> torch.arange(5, device=snr.device)[None]) & 1).float()


def _beam(device):
    """Public beam table [2, TX]: unit-norm DFT column k duplicated over pols."""
    pos = torch.arange(8, device=device, dtype=torch.float32)
    table = torch.zeros(2, NUM_TX, dtype=torch.complex64, device=device)
    for role, k in enumerate(BEAM_OF_ROLE):
        u = torch.exp(-2j * math.pi * k * pos / 8) / math.sqrt(8)
        table[role] = torch.cat([u, u]) / math.sqrt(2)
    return table


def _psk_points(order, device):
    angles = torch.arange(order, device=device, dtype=torch.float32) \
        * (2 * math.pi / order) + (math.pi / order)
    return torch.polar(torch.ones(order, dtype=torch.float32, device=device), angles).to(torch.complex64)


def _psk_labels(order, device):
    """Gray label bits [order, m], bit order MSB first."""
    m = int(round(math.log2(order)))
    idx = torch.arange(order, device=device)
    gray = idx ^ (idx >> 1)
    bits = ((gray[:, None] >> (m - 1 - torch.arange(m, device=device))[None]) & 1)
    return bits.float(), m


order_map = {"qpsk": 4, "8psk": 8, "16psk": 16}


def _psk_map(bits, order, device):
    """bits [..., n*m] {0,1} -> unit-modulus symbols [..., n]."""
    points = _psk_points(order, device)
    labels, m = _psk_labels(order, device)                    # [order, m]
    d = (bits.reshape(*bits.shape[:-1], -1, m)[:, :, None, :] - labels[None, None]).abs().square().sum(-1)
    idx = d.argmin(-1)                                        # nearest label
    return points[idx]


def _psk_demod_llr(sym, order, noise_var):
    """Max-log LLR of Gray PSK; sym [..., n], noise per-symbol variance (scalar
    tensor broadcastable). Returns [..., n*m], positive = bit 1."""
    points = _psk_points(order, sym.device)
    labels, m = _psk_labels(order, sym.device)                # [order, m]
    s = sym.reshape(-1)
    d2 = (s[:, None] - points[None, :]).abs().square().to(torch.float32)
    metric = (-d2 / noise_var.reshape(-1)[:, None])[:, :, None]   # [N, order, 1]
    lab = labels[None]                                            # [1, order, m]
    pos = torch.where(lab > 0.5, metric, torch.full_like(metric, -1e30)).amax(1)
    neg = torch.where(lab < 1.5, metric, torch.full_like(metric, -1e30)).amax(1)
    return (pos - neg).reshape(*sym.shape[:-1], sym.shape[-1] * m)


class Encoder(nn.Module):
    """Constant unit-energy feedback; the fixed-beam design uses no CSIT."""

    def forward(self, h, snr):
        b = h.shape[0]
        u = torch.ones(b, 96, dtype=torch.complex64, device=h.device)
        return u / (u.abs().square().mean().sqrt() + 1e-12)


class Transmitter(nn.Module):
    def forward(self, bits_list, feedback_list, snr, return_aux=False):
        device = bits_list[0].device
        b = bits_list[0].shape[0]
        beams = _beam(device)                                    # [2, TX]
        q = _quant32(snr)                                        # [UE, B]
        t = q.min(dim=0).values                                  # [B]
        x = torch.zeros(b, NUM_TX, NUM_RE, dtype=torch.complex64, device=device)
        for ue in range(NUM_UE):
            role = (q[ue] <= t).long()                           # 0=LOW, 1=HIGH
            pols = _policy_for_snr(snr[ue])
            sym = torch.zeros(b, NUM_RE, dtype=torch.complex64, device=device)
            for key in set(_policy_key(p) for p in pols):
                sel = [i for i, p in enumerate(pols) if _policy_key(p) == key]
                pol = pols[sel[0]]
                info = _encode_source(bits_list[ue][sel], pol)   # [n,K] {0,1}
                slots = _slot_pattern(info.float(), pol)         # [n,S]
                sym[sel] = _psk_map(slots, order_map[pol["mod"]], device)
            x = x + (STREAM_POWER ** 0.5) * beams[role][:, :, None] * sym[:, None, :]
        ctrl = _ctrl_bits(snr)
        if return_aux:
            return x, ctrl, {}
        return x, ctrl


def _encode_source(bits, pol):
    if pol["mode"] == "prefix":
        return bits[:, :_info_count(pol)]
    m = pol["m"]
    g = bits.reshape(bits.shape[0], NUM_BITS // m, m)
    return (g.mean(-1) >= 0.5).float()


def _slot_pattern(info, pol):
    s = _slot_count(pol)
    k = info.shape[1]
    idx = torch.arange(s, device=info.device) % k
    return info[:, idx]


class Receiver(nn.Module):
    """Exact per-RE MMSE on the public 2-stream effective channel.

    The waveform carries per-user stream power STREAM_POWER on unit beams, so
    with unit-modulus symbols the per-RE energy is exactly 1 and the official
    normalisation is the identity; the receiver knows everything exactly.
    """
    def __init__(self):
        super().__init__()
        # The design has no trained weights; evaluation utilities iterate
        # parameters() to find the device, so register one unused placeholder.
        self._device_anchor = nn.Parameter(torch.zeros(1))

    def forward_full(self, y, h, ctrl, snr):
        # y [B,RX,RE]; h [B,RX,TX,RE]; snr [B] (dB); ctrl [B,5]
        device = y.device
        b = y.shape[0]
        beams = _beam(device)                                    # [2, TX]
        q = _quant32(snr)
        t = (ctrl * (2 ** torch.arange(5, device=device)).float()).sum(-1)
        role = (q <= t).long()                                   # 0=LOW, 1=HIGH
        heff = torch.einsum("brtp,st->brps", h, beams)           # [B,RX,RE,2]
        sigma = (10.0 ** (-snr / 20.0))[:, None, None, None]     # [B,1,1,1]
        # Normalise by noise std so MMSE uses A = He^H He + I:
        # SINR_k = 1/[A^-1]kk - 1 with unit-power streams.
        he = (heff * (STREAM_POWER ** 0.5) / sigma).permute(0, 2, 1, 3)  # [B,RE,RX,2]
        a = (he.conj() * he).sum(2).real                         # [B,RE,2] diag
        c = (he[:, :, :, 0].conj() * he[:, :, :, 1]).sum(2)      # [B,RE] off-diag
        d0 = a[..., 0] + 1.0
        d1 = a[..., 1] + 1.0
        det = (d0 * d1 - c.abs().square()).clamp_min(1e-12)      # [B,RE]
        yf = y.permute(0, 2, 1)                                  # [B,RE,RX]
        s0 = (d1[..., None] * he[..., 0].conj() - c[..., None] * he[..., 1].conj()) * yf
        s1 = (d0[..., None] * he[..., 1].conj() - c.conj()[..., None] * he[..., 0].conj()) * yf
        y0 = s0.sum(-1) / det                                    # [B,RE]
        y1 = s1.sum(-1) / det
        sinr0 = (det / d1 - 1.0).clamp_min(1e-9)
        sinr1 = (det / d0 - 1.0).clamp_min(1e-9)
        my_sym = torch.where((role == 0)[:, None], y0, y1)        # [B,RE]
        my_sinr = torch.where((role == 0)[:, None], sinr0, sinr1)
        # MMSE bias c=sinr/(1+sinr); unbiased symbols, noise var c^2/sinr.
        bias = my_sinr / (1.0 + my_sinr)
        my_sym = my_sym / bias
        noise_eff = (bias ** 2 / my_sinr).clamp_min(1e-9)
        out = torch.zeros(b, NUM_BITS, device=device)
        pols = _policy_for_snr(snr)
        for key in set(_policy_key(p) for p in pols):
            sel = [i for i, p in enumerate(pols) if _policy_key(p) == key]
            pol = pols[sel[0]]
            slot_llr = _psk_demod_llr(my_sym[sel], order_map[pol["mod"]],
                                      noise_eff[sel])            # [n, RE*m]
            info_llr = _combine_slots(slot_llr, pol)             # [n,K]
            out[sel] = _expand_to_source(info_llr, pol)
        return out

    def forward(self, y, h, ctrl, snr):
        return self.forward_full(y, h, ctrl, snr)

    def valid_lengths(self, snr, ctrl=None):
        """Per-sample scored prefix length; lossy codes reconstruct all 1152."""
        pols = _policy_for_snr(snr)
        return torch.tensor([_policy_valid_lengths(p) for p in pols],
                            device=snr.device, dtype=torch.long)


def _combine_slots(slot_llr, pol):
    """[n, S] slot LLRs -> [n, K] info LLRs by LLR summation over repeats."""
    s = _slot_count(pol)
    k = _info_count(pol)
    llr = slot_llr.reshape(slot_llr.shape[0], -1)[:, :s]
    out = torch.zeros(slot_llr.shape[0], k, device=slot_llr.device)
    idx = torch.arange(s, device=slot_llr.device) % k
    out.index_add_(1, idx, llr)
    return out


def _expand_to_source(info_llr, pol):
    b = info_llr.shape[0]
    device = info_llr.device
    if pol["mode"] == "prefix":
        out = torch.zeros(b, NUM_BITS, device=device)
        out[:, :info_llr.shape[1]] = info_llr
        return out
    m = pol["m"]
    expanded = info_llr.repeat_interleave(m, dim=1)
    out = torch.zeros(b, NUM_BITS, device=device)
    out[:, :expanded.shape[1]] = expanded
    return out


def _policy_valid_lengths(pol):
    return NUM_BITS if pol["mode"] == "maj" else min(_info_count(pol), NUM_BITS)
