"""Self-contained, untrained research model for the 6G AI Challenge.

v1: full-rate per-RE 16-QAM initialization, two streams per UE.
v2: learned cross-RE block JSCC with an SNR-selected source prefix.
v3: learned cross-RE block JSCC for all 1152 source bits.

Only this file and its matching three state dictionaries are needed for a
submission. All receivers use only (y, own h, common control bits, own SNR).
They never receive the actual precoder, the other user's channel, or a user ID.
Positive receiver logits mean bit 1. No performance is claimed without training.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


MODEL_VARIANT = "v3"
NUM_RE = 144
NUM_BITS = 1152

# v8: strongest combined candidate (three-agent synthesis, reports/agents_synthesis.md).
# Air-interface ideas, each independently verified by the external explorations:
#   - subband-constant robust RZF (Kimi v6): W per 48-SC subband, the precondition
#     for pilot-based estimation; per-stream learned power ratios (zero-init MLP)
#     multiplied into W so pilots and data share the allocation.
#   - rank-4 Hadamard pilots on 12 fixed REs through the same W (Kimi v5/v6,
#     Qwen pilots): the receiver recovers the per-subband effective channel by
#     least squares instead of guessing it from y.
#   - role protocol (Qwen 4.1): users sorted low-SNR first; the control word
#     carries the min-SNR bin (31 = quantised tie), so each receiver knows
#     which two of the four estimated streams are its own without a user ID.
#   - receiver: v3 features + 29 physical dims (pilot-LS effective channel,
#     assumed-own slice, LS residual, role/tie flags); embed columns 0-78
#     match v3 exactly for warm starting.
# Dropped deliberately: GLM-1's replica replay (superseded by the pilot LS; it
# slowed from-scratch convergence in our v7 run without detectable benefit).


NUM_FEEDBACK = 96
NUM_TX = 16
NUM_RX = 2
NUM_UE = 2
SNR_UL_GAP_DB = 10.0


def _snr_vector(snr, batch, device):
    snr = torch.as_tensor(snr, dtype=torch.float32, device=device).reshape(-1)
    if snr.numel() == 1:
        snr = snr.expand(batch)
    if snr.numel() != batch:
        raise ValueError("Expected one SNR in dB per sample.")
    return snr


def _snr_features(snr):
    """DL SNR, actual UL SNR, and an analog-feedback reliability feature."""
    ul_snr = snr - SNR_UL_GAP_DB
    ul_noise = torch.pow(10.0, -ul_snr / 10.0)
    return torch.stack((snr / 20.0, ul_snr / 20.0, 1.0 / (1.0 + ul_noise)), -1)


def valid_lengths(snr, ctrl=None):
    """Public rate policy, known independently to the TX and each RX.

    The v2 thresholds are an unvalidated starting schedule, not a capacity
    bound or an optimal allocation. v1 and v3 always predict all source bits.
    The common control word quantizes the minimum pair SNR; it carries
    neither a private per-user rate nor a user identity.
    """
    snr = torch.as_tensor(snr, dtype=torch.float32).reshape(-1)
    if MODEL_VARIANT != "v2":
        return torch.full_like(snr, NUM_BITS, dtype=torch.long)
    thresholds = snr.new_tensor([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    lengths = torch.tensor([16, 32, 72, 144, 288, 576, 864, 1152],
                           dtype=torch.long, device=snr.device)
    return lengths[torch.bucketize(snr.contiguous(), thresholds, right=True)]


def _complex(re, im):
    # Explicitly preserve the official complex64 interface under autocast.
    return torch.complex(re.float(), im.float())


def _complex_solve(a, b):
    """Solve a @ x = b for complex a, b.

    torch.linalg.solve has no complex kernel on MPS, so there an exactly
    equivalent real 2n x 2n block system is solved instead; other devices use
    the native complex path with identical numerics.
    """
    if a.device.type != "mps":
        return torch.linalg.solve(a, b)
    ar, ai = a.real, a.imag
    block = torch.cat((torch.cat((ar, -ai), -1), torch.cat((ai, ar), -1)), -2)
    rhs = torch.cat((b.real, b.imag), -2)
    solution = torch.linalg.solve(block, rhs)
    rows = a.shape[-1]
    return _complex(solution[..., :rows, :], solution[..., rows:, :])


def _channel_features(h):
    batch = h.shape[0]
    flat = h.reshape(batch, NUM_RX * NUM_TX, NUM_RE)
    return torch.cat((flat.real, flat.imag), dim=1).float()


def _as_channel(features):
    re, im = features.float().chunk(2, dim=1)
    return _complex(re, im).reshape(-1, NUM_RX, NUM_TX, NUM_RE)


class _LocalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                nn.Linear(width * 2, width))

    def forward(self, x):
        z = self.norm(x)
        z = self.depthwise(z.transpose(1, 2)).transpose(1, 2)
        return x + self.ff(z)


class _GlobalBlock(nn.Module):
    """Dense token mixing: every RE can affect every other RE in one block.

    This is a full-block source/channel code component, not independent
    per-RE modulation with a renamed MLP. Complexity is linear in width.
    """
    def __init__(self, width, token_hidden=72):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        self.token_mix = nn.Sequential(nn.Linear(NUM_RE, token_hidden), nn.GELU(),
                                       nn.Linear(token_hidden, NUM_RE))
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mix = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                         nn.Linear(width * 2, width))

    def forward(self, x):
        x = x + self.token_mix(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mix(self.channel_norm(x))


class Encoder(nn.Module):
    """Analog CSI JSCC, with frequency and all-tap delay representations.

    No antenna geometry is assumed. All 144 delay coordinates enter a learned
    projection; no empirical energy-retention percentage is hard-coded.
    """
    def __init__(self):
        super().__init__()
        width = 32
        self.frequency_in = nn.Conv1d(64, width, 3, padding=1)
        self.delay_in = nn.Conv1d(64, width, 3, padding=1)
        self.frequency_block = _LocalBlock(width)
        self.delay_block = _LocalBlock(width)
        self.frequency_projection = nn.Linear(NUM_RE, 12)
        self.delay_projection = nn.Linear(NUM_RE, 12)
        self.feedback_head = nn.Sequential(nn.Linear(2 * width * 12 + 4, 384),
                                           nn.GELU(), nn.Linear(384, 192))

    def forward(self, h, snr):
        if h.ndim != 4 or tuple(h.shape[1:]) != (NUM_RX, NUM_TX, NUM_RE):
            raise ValueError("Encoder expects h [batch, 2, 16, 144].")
        h = h.to(torch.complex64)
        batch = h.shape[0]
        snr = _snr_vector(snr, batch, h.device)
        scale = h.abs().square().mean((1, 2, 3)).clamp_min(1e-10).sqrt()
        h_unit = h / scale[:, None, None, None]
        delay = torch.fft.ifft(h_unit, dim=-1, norm="ortho")
        frequency = self.frequency_in(_channel_features(h_unit)).transpose(1, 2)
        delay = self.delay_in(_channel_features(delay)).transpose(1, 2)
        frequency = self.frequency_projection(
            self.frequency_block(frequency).transpose(1, 2)).flatten(1)
        delay = self.delay_projection(
            self.delay_block(delay).transpose(1, 2)).flatten(1)
        latent = self.feedback_head(torch.cat((frequency, delay,
                                              scale.log()[:, None], _snr_features(snr)), -1))
        re, im = latent.chunk(2, -1)
        u = _complex(re, im)
        return u / u.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


class Decoder(nn.Module):
    """Denoise the actual noisy UL feedback and reconstruct full-frequency CSI."""
    def __init__(self):
        super().__init__()
        width = 32
        self.embed = nn.Sequential(nn.Linear(192 + 3, 384), nn.GELU(),
                                   nn.Linear(384, width * 12))
        self.delay_expand = nn.Linear(12, NUM_RE)
        self.frequency_expand = nn.Linear(12, NUM_RE)
        self.delay_block = _LocalBlock(width)
        self.frequency_block = _LocalBlock(width)
        self.delay_out = nn.Conv1d(width, 64, 1)
        self.frequency_out = nn.Conv1d(width, 64, 1)

    def forward(self, feedback, snr):
        batch = feedback.shape[0]
        snr = _snr_vector(snr, batch, feedback.device)
        ul_noise = torch.pow(10.0, -(snr - SNR_UL_GAP_DB) / 10.0)
        # Unit-power input, additive variance n: a linear-MMSE-style shrinkage
        # suppresses feedback noise as n grows. Bias/conditioning can learn the
        # statistical prior; this is not a claim of optimal nonlinear MMSE.
        stable = feedback.to(torch.complex64) / (1.0 + ul_noise[:, None])
        inputs = torch.cat((stable.real, stable.imag, _snr_features(snr)), -1)
        latent = self.embed(inputs.float()).reshape(batch, 32, 12)
        delay = self.delay_block(self.delay_expand(latent).transpose(1, 2))
        frequency = self.frequency_block(self.frequency_expand(latent).transpose(1, 2))
        h_delay = _as_channel(self.delay_out(delay.transpose(1, 2)))
        h_frequency = _as_channel(self.frequency_out(frequency.transpose(1, 2)))
        return torch.fft.fft(h_delay, dim=-1, norm="ortho") + h_frequency


class Precoder(nn.Module):
    """Subband-constant robust RZF (external notes' Kimi-line v6) with the v3
    UE-shared residual and a zero-initialised per-stream gain MLP.

    W is computed once per 48-SC subband from the subband-averaged decoder
    estimate and broadcast to every RE of that subband, so each subband's four
    pilot REs see exactly the same precoder as its data REs -- the
    precondition for the receiver's pilot least-squares to be meaningful.
    The gain MLP outputs log-ratios per stream; being multiplied into W before
    the per-RE normalisation it acts as pure stream-power ratios and applies
    to pilot and data REs alike.
    """

    def __init__(self):
        super().__init__()
        self.log_dl_weight = nn.Parameter(torch.tensor(1.0))
        self.log_ul_weight = nn.Parameter(torch.tensor(0.0))
        self.correction = nn.Sequential(nn.Linear(64 * 2 + 3 * 2, 64), nn.GELU(),
                                        nn.Linear(64, 64))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.gain_head = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, NUM_UE * NUM_RX))
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def forward(self, h_hat, snr):
        # h_hat: [B, UE, RX, TX, RE], snr: [UE, B].
        batch = h_hat.shape[0]
        device = h_hat.device
        h = h_hat.permute(0, 4, 1, 2, 3).contiguous()            # B,F,U,R,T
        hs = h.reshape(batch, NUM_RE, NUM_UE, NUM_RX, NUM_TX)
        sub = torch.arange(NUM_RE, device=device) // 48
        acc = torch.zeros(batch, 3, NUM_UE, NUM_RX, NUM_TX, dtype=h.dtype, device=device)
        acc.index_add_(1, sub, hs)
        h_sub = acc / 48.0                                       # B,3,U,R,T
        # Kimi spec: the subband Gram is the AVERAGE OF PER-RE GRAMS (first
        # moment of h^H h), which retains the within-subband channel variance
        # that RZF needs for interference suppression -- not the gram of the
        # averaged channel.
        hs4 = hs.reshape(batch, NUM_RE, NUM_UE * NUM_RX, NUM_TX)
        gram_re = torch.einsum("brea,breb->brab", hs4, hs4.conj())
        gram = torch.zeros(batch, 3, NUM_UE * NUM_RX, NUM_UE * NUM_RX,
                           dtype=gram_re.dtype, device=gram_re.device)
        gram.index_add_(1, sub, gram_re)
        gram = gram / 48.0                                       # B,3,4,4
        g = h_sub.reshape(batch, 3, NUM_UE * NUM_RX, NUM_TX)
        snr_bu = snr.transpose(0, 1).float()                     # B,U
        noise_dl = torch.pow(10.0, -snr_bu / 10.0)
        noise_ul = torch.pow(10.0, -(snr_bu - SNR_UL_GAP_DB) / 10.0)
        feedback_uncertainty = noise_ul / (1.0 + noise_ul)
        row_energy = h_sub.abs().square().sum(-1).mean((1, 3))   # B,U
        alpha = (F.softplus(self.log_dl_weight) * noise_dl +
                 F.softplus(self.log_ul_weight) * feedback_uncertainty * row_energy + 1e-3)
        alpha = alpha.repeat_interleave(NUM_RX, dim=1)[:, None, :]
        w = _complex_solve(gram + torch.diag_embed(alpha), g)    # B,3,4,T
        w = w.conj().transpose(-2, -1)                           # B,3,T,4
        per_user = h_sub.reshape(batch, 3, NUM_UE, NUM_RX * NUM_TX)
        features = torch.cat((per_user.real, per_user.imag), -1)
        symmetric_context = features.mean(2, keepdim=True).expand_as(features)
        snr_features = _snr_features(snr_bu)[:, None].expand(-1, 3, -1, -1)
        snr_context = snr_features.mean(2, keepdim=True).expand_as(snr_features)
        correction = self.correction(torch.cat((features, symmetric_context,
                                                snr_features, snr_context), -1).float())
        correction = correction.reshape(batch, 3, NUM_UE, NUM_RX, NUM_TX, 2)
        correction = _complex(correction[..., 0], correction[..., 1])
        correction = correction.reshape(batch, 3, NUM_UE * NUM_RX, NUM_TX)
        w = w + 0.1 * correction.transpose(-2, -1)               # B,3,T,4
        pair = torch.stack((snr_bu.min(1).values, snr_bu.max(1).values,
                            snr_bu[:, 0] - snr_bu[:, 1]), -1)    # B,3
        log_gains = self.gain_head(torch.cat((pair, _snr_features(snr_bu)), -1).float())
        w = w * torch.exp(log_gains)[:, None, None, :]           # ratios, zero-init = identity
        w_re = w[:, sub]                                         # B,144,T,4
        w_re = w_re / w_re.abs().square().sum((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        return w_re


class Precoder(nn.Module):
    """Four-stream robust RZF with a UE-shared, permutation-equivariant residual.

    Stream order inside each UE follows its two observed receive antennas.
    User-index embeddings and user-specific modules are deliberately absent.
    """
    def __init__(self):
        super().__init__()
        self.log_dl_weight = nn.Parameter(torch.tensor(1.0))
        self.log_ul_weight = nn.Parameter(torch.tensor(0.0))
        self.correction = nn.Sequential(nn.Linear(64 * 2 + 3 * 2, 64), nn.GELU(),
                                        nn.Linear(64, 64))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, h_hat, snr):
        # h_hat: [B, UE, RX, TX, RE], snr: [UE, B].
        batch = h_hat.shape[0]
        h = h_hat.permute(0, 4, 1, 2, 3).contiguous()  # B,F,U,R,T
        g = h.reshape(batch, NUM_RE, NUM_UE * NUM_RX, NUM_TX)
        gram = g @ g.conj().transpose(-2, -1)
        snr_bu = snr.transpose(0, 1).float()
        noise_dl = torch.pow(10.0, -snr_bu / 10.0)
        noise_ul = torch.pow(10.0, -(snr_bu - SNR_UL_GAP_DB) / 10.0)
        feedback_uncertainty = noise_ul / (1.0 + noise_ul)
        row_energy = h.abs().square().sum(-1).mean((1, 3))  # B,U
        alpha = (F.softplus(self.log_dl_weight) * noise_dl +
                 F.softplus(self.log_ul_weight) * feedback_uncertainty * row_energy + 1e-3)
        alpha = alpha.repeat_interleave(NUM_RX, dim=1)[:, None, :]
        inverse_times_g = _complex_solve(gram + torch.diag_embed(alpha), g)
        w = inverse_times_g.conj().transpose(-2, -1)  # B,F,T,U*R
        # A conjugated RZF solution is G^H (G G^H + A)^-1 since A is real.
        per_user = h.reshape(batch, NUM_RE, NUM_UE, NUM_RX * NUM_TX)
        features = torch.cat((per_user.real, per_user.imag), -1)
        symmetric_context = features.mean(2, keepdim=True).expand_as(features)
        snr_features = _snr_features(snr_bu)[:, None].expand(-1, NUM_RE, -1, -1)
        snr_context = snr_features.mean(2, keepdim=True).expand_as(snr_features)
        correction = self.correction(torch.cat((features, symmetric_context,
                                                 snr_features, snr_context), -1).float())
        correction = correction.reshape(batch, NUM_RE, NUM_UE, NUM_RX, NUM_TX, 2)
        correction = _complex(correction[..., 0], correction[..., 1])
        correction = correction.reshape(batch, NUM_RE, NUM_UE * NUM_RX, NUM_TX)
        w = w + 0.1 * correction.transpose(-2, -1)
        return w / w.abs().square().sum((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()


def _qam16(signs):
    """Gray 16-QAM per spatial stream, with bit 1 giving positive signs."""
    bits = signs.reshape(*signs.shape[:-1], NUM_RX, 4)
    re = bits[..., 0] * (2.0 - bits[..., 1]) / math.sqrt(10.0)
    im = bits[..., 2] * (2.0 - bits[..., 3]) / math.sqrt(10.0)
    return torch.stack((re, im), dim=-1).flatten(-2)


class _LocalModulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(8 + 3, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, bits, snr, lengths):
        signs = (2.0 * bits.float() - 1.0).reshape(-1, NUM_RE, 8)
        condition = _snr_features(snr)[:, None].expand(-1, NUM_RE, -1)
        return _qam16(signs) + self.residual(torch.cat((signs, condition), -1))


class _BlockModulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(8 + 8 + 3 + 1, 64)
        self.blocks = nn.Sequential(_LocalBlock(64), _GlobalBlock(64), _GlobalBlock(64))
        self.output = nn.Linear(64, 4)
        # Nonzero initialization makes the cross-RE path active immediately.
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        self.systematic_gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, bits, snr, lengths):
        mask = (torch.arange(NUM_BITS, device=bits.device)[None] < lengths[:, None]).float()
        signs = ((2.0 * bits.float() - 1.0) * mask).reshape(-1, NUM_RE, 8)
        mask = mask.reshape(-1, NUM_RE, 8)
        conditions = torch.cat((_snr_features(snr), lengths[:, None].float() / NUM_BITS), -1)
        conditions = conditions[:, None].expand(-1, NUM_RE, -1)
        features = torch.cat((signs, mask, conditions), -1)
        learned = self.output(self.blocks(self.embed(features)))
        return learned + self.systematic_gain * _qam16(signs)


PILOT_SUB_POS = (0, 12, 24, 36)                              # 4 pilots per 48-SC subband
PILOT_POS = tuple(s * 48 + k for s in range(3) for k in PILOT_SUB_POS)
_H4 = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0]]) / 2.0           # rows = unit-power codewords


def _sort_users(snr_dl):
    """Permutation placing the lower-SNR user first (role protocol).

    All UE-indexed modules are permutation-equivariant (shared weights, no
    user embedding), so sorting is a pure relabelling and keeps v3 weights
    valid. Returns perm [B,2] with perm[:,0] = lower-SNR UE index.
    """
    return torch.argsort(snr_dl.transpose(0, 1), dim=1, stable=True)


def _gather_ue(tensor, perm):
    """Reorder the UE dimension (dim 1) of tensor [B,U,...] per-sample."""
    idx = perm.reshape(tensor.shape[0], NUM_UE, *([1] * (tensor.ndim - 2)))
    return torch.gather(tensor, 1, idx.expand(-1, -1, *tensor.shape[2:]))


class Transmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self._decoder = Decoder()
        self._precoder = Precoder()
        self._modulator = _LocalModulator() if MODEL_VARIANT == "v1" else _BlockModulator()
        self.register_buffer("pilot_pattern", _H4.to(torch.complex64).repeat(3, 1))   # [12,4], one H4 per subband
        self.register_buffer("pilot_index", torch.tensor(PILOT_POS, dtype=torch.long))

    def forward(self, bits_list, feedback_list, snr_dl, return_aux=False):
        if len(bits_list) != NUM_UE or len(feedback_list) != NUM_UE:
            raise ValueError("Expected exactly two UEs.")
        batch = bits_list[0].shape[0]
        device = bits_list[0].device
        snr_dl = torch.as_tensor(snr_dl, device=device, dtype=torch.float32)
        if tuple(snr_dl.shape) != (NUM_UE, batch):
            raise ValueError("Expected snr_dl [2, batch].")
        for bits, feedback in zip(bits_list, feedback_list):
            if tuple(bits.shape) != (batch, NUM_BITS):
                raise ValueError("Each source must have shape [batch, 1152].")
            if tuple(feedback.shape) != (batch, NUM_FEEDBACK):
                raise ValueError("Each feedback must have shape [batch, 96].")
        lengths = torch.stack([valid_lengths(snr_dl[i]) for i in range(NUM_UE)], 1)
        perm = _sort_users(snr_dl)                                # [B,2], low-SNR user first

        symbols = []
        estimates = []
        for i in range(NUM_UE):
            z = self._modulator(bits_list[i], snr_dl[i], lengths[:, i])
            z = z.reshape(batch, NUM_RE, NUM_RX, 2)
            z = _complex(z[..., 0], z[..., 1])
            z = z / z.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
            symbols.append(z)
            estimates.append(self._decoder(feedback_list[i], snr_dl[i]))
        h_hat = _gather_ue(torch.stack(estimates, 1), perm)       # B,U,R,T,F sorted
        snr_sorted = torch.gather(snr_dl, 0, perm.transpose(0, 1))  # rows follow the sort
        w = self._precoder(h_hat, snr_sorted)                     # B,F,T,4
        streams = torch.stack(symbols, 1)                         # B,U,F,R
        streams = _gather_ue(streams, perm)
        streams = streams.reshape(batch, NUM_RE, NUM_UE * NUM_RX)
        streams = streams / streams.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        # Known rank-4 Hadamard pilots through the same subband-constant W.
        streams[:, self.pilot_index, :] = self.pilot_pattern[None].expand(batch, -1, -1)
        x = (w @ streams.unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        power = x.abs().square().sum(1, keepdim=True).mean(-1, keepdim=True)
        x = x / power.clamp_min(1e-12).sqrt()
        # v3 min-SNR control word; code 31 now additionally flags a quantised
        # tie (both users in the same 1.25 dB bin), Qwen's collision protocol.
        q = torch.floor((snr_dl + 20.0) * 0.8).long().clamp(0, 31)
        mode = torch.floor((snr_dl.min(dim=0).values + 20.0) * (32.0 / 40.0)).long().clamp(0, 31)
        mode = torch.where(q[0] == q[1], torch.full_like(mode, 31), mode)
        ctrl = ((mode[:, None] >> torch.arange(5, device=device)) & 1).float()
        if return_aux:
            return x, ctrl, {"h_hat": h_hat, "precoder": w, "user_perm": perm,
                             "valid_lengths": lengths, "symbols": streams}
        return x, ctrl


class Receiver(nn.Module):
    """Pilot-anchored neural receiver (combines the strongest verified levers).

    On top of the v3 features (h/y/scales/condition, 79 dims) it adds 29 dims
    of physics features derived only from official inputs and public constants:

    - per-subband effective-channel least-squares estimate from the four known
      Hadamard pilots (16 dims, broadcast over the subband's REs);
    - the role-protocol's assumed-own slice of that estimate (8 dims; streams
      0-1 for the lower-SNR role, 2-3 for the higher, zeros on a quantised
      tie) plus role and tie flags (2 dims);
    - the pilot least-squares fit residual per subband (3 dims), which exposes
      how well the subband-constant model actually fits.

    The embed is laid out so its first 79 input columns match v3 exactly:
    warm starting copies those columns and small-random-initialises the rest.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("pilot_pattern", _H4.to(torch.complex64).repeat(3, 1))   # [12,4], one H4 per subband
        self.register_buffer("pilot_index", torch.tensor(PILOT_POS, dtype=torch.long))
        self.register_buffer("sub_of_re", torch.arange(NUM_RE) // 48)
        width = 96
        self.embed = nn.Linear(64 + 4 + 2 + 3 + 5 + 1 + 27, width)
        position = torch.arange(NUM_RE).float()[:, None]
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe = torch.zeros(NUM_RE, width)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", 0.02 * pe[None])
        blocks = [_LocalBlock(width), _LocalBlock(width), _GlobalBlock(width),
                  _GlobalBlock(width), _GlobalBlock(width)]
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))

    def valid_lengths(self, snr, ctrl=None):
        return valid_lengths(snr, ctrl)

    def _pilot_features(self, y, snr, y_scale):
        """Least-squares effective-channel estimate from the known pilots."""
        batch = y.shape[0]
        device = y.device
        s = self.pilot_pattern.to(device).reshape(3, 4, 4)         # [3,4,4]
        y_pil = y[:, :, self.pilot_index].permute(0, 2, 1).reshape(batch, 3, 4, NUM_RX)
        # Orthonormal codeword rows: Heff_est = Y @ conj(S) (ridge = identity).
        heff = torch.einsum("bckr,ckj->bcrj", y_pil, s.conj())       # [B,3,RX,4]
        fit = torch.einsum("bcrj,ckj->bckr", heff, s)
        resid = (y_pil - fit).abs().square().mean((2, 3))          # [B,3]
        resid = (resid / y_scale.square().mean(1, keepdim=True).clamp_min(1e-10)).clamp_min(1e-12).log()
        return heff, resid

    def forward_full(self, y, h, ctrl_bits, snr):
        batch = y.shape[0]
        if tuple(y.shape) != (batch, NUM_RX, NUM_RE):
            raise ValueError("Receiver expects y [batch, 2, 144].")
        if tuple(h.shape) != (batch, NUM_RX, NUM_TX, NUM_RE):
            raise ValueError("Receiver expects own h [batch, 2, 16, 144].")
        if tuple(ctrl_bits.shape) != (batch, 5):
            raise ValueError("Receiver expects the common control word [batch, 5].")
        snr = _snr_vector(snr, batch, y.device)
        h = h.to(torch.complex64)
        y = y.to(torch.complex64)
        h_scale = h.abs().square().mean((1, 2)).clamp_min(1e-10).sqrt()  # B,F
        noise = torch.pow(10.0, -snr / 10.0)
        y_scale = (NUM_TX * h_scale.square() + noise[:, None]).clamp_min(1e-10).sqrt()
        h_features = _channel_features(h / h_scale[:, None, None]).transpose(1, 2)
        y_unit = (y / y_scale[:, None]).transpose(1, 2)
        y_features = torch.cat((y_unit.real, y_unit.imag), -1)
        scale_features = torch.stack((h_scale.log(), y_scale.log()), -1)
        lengths = self.valid_lengths(snr, ctrl_bits)
        condition = torch.cat((_snr_features(snr), ctrl_bits.float(),
                               lengths[:, None].float() / NUM_BITS), -1)
        condition = condition[:, None].expand(-1, NUM_RE, -1)

        # ---- role protocol (public: own q vs the control word) ----
        q_own = torch.floor((snr + 20.0) * 0.8).clamp(0, 31)
        t = (ctrl_bits * (2 ** torch.arange(5, device=y.device)).float()).sum(-1)
        tie = t >= 31.0 - 1e-3
        low_role = (~tie) & (q_own <= t)                           # own = streams 0-1
        role_flag = (~low_role).float()[:, None]                   # 0 low, 1 high
        tie_flag = tie.float()[:, None]

        # ---- pilot least-squares effective channel ----
        heff, resid = self._pilot_features(y, snr, y_scale)        # [B,3,RX,4], [B,3]
        sub_scale = y_scale.reshape(batch, 3, 48).mean(-1)          # [B,3]
        heff_n = heff / sub_scale[:, :, None, None].clamp_min(1e-10).to(heff.dtype)
        sub = self.sub_of_re.to(y.device)
        heff_per_re = heff_n[:, sub]                                # [B,144,RX,4]
        heff_feat = torch.cat((heff_per_re.real, heff_per_re.imag), -1).reshape(batch, NUM_RE, -1)
        # the two assumed-own stream columns per role (zeroed on a tie)
        cols = torch.where(low_role[:, None],
                           torch.tensor([0, 1], device=y.device),
                           torch.tensor([2, 3], device=y.device))   # [B,2]
        own_slice = heff_per_re.gather(
            3, cols[:, None, None, :].expand(-1, NUM_RE, NUM_RX, 2))
        own_slice = torch.cat((own_slice.real, own_slice.imag), -1).reshape(batch, NUM_RE, -1)
        own_slice = own_slice * (~tie)[:, None, None].float()
        resid_feat = resid[:, sub][:, :, None]                       # [B,144,1]

        phys = torch.cat((heff_feat, own_slice, resid_feat,
                          role_flag[:, :, None].expand(-1, NUM_RE, 1),
                          tie_flag[:, :, None].expand(-1, NUM_RE, 1)), -1)
        features = torch.cat((h_features, y_features, scale_features, condition,
                              phys), -1).float()
        hidden = self.embed(features) + self.position
        return self.output(self.blocks(hidden)).reshape(batch, NUM_BITS).float()

    def forward(self, y, h, ctrl_bits, snr):
        snr = _snr_vector(snr, y.shape[0], y.device)
        lengths = self.valid_lengths(snr, ctrl_bits)
        if not torch.equal(lengths, lengths[:1].expand_as(lengths)):
            raise ValueError("Mixed prefix lengths cannot form an official dense LLR batch. "
                             "Use batch_size=1, bucket by rate, or forward_full plus valid_lengths.")
        return self.forward_full(y, h, ctrl_bits, snr)
