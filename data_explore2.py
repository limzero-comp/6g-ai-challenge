"""Round-2 data exploration: compression floor, delay-tap positions, azimuth
structure, and a perfect-CSI scheme simulation scored by the competition metric.

CPU only. Run: python data_explore2.py  (expects ./data_train/H_train.npz)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figs', exist_ok=True)
rng = np.random.default_rng(7)
log_lines = []


def log(s=''):
    print(s, flush=True)
    log_lines.append(str(s))


log('loading data...')
d = np.load('./data_train/H_train.npz')
H = d['real'].astype(np.float32) + 1j * d['imag'].astype(np.float32)  # (N, ue, rx, tx, sc)
N, UE, RX, TX, SC = H.shape
NS = 5000
idx = rng.choice(N, NS, replace=False)
Hs = H[idx]
del H, d
log(f'subsample {Hs.shape}')

# ---------- delay transform ----------
G = np.fft.ifft(Hs, axis=-1)[..., :16]   # (NS, ue, rx, tx, 16)
P = (np.abs(G) ** 2)

# ============ B. delay tap positions ============
peak_tap = P.argmax(-1).ravel()
centroid = (P * np.arange(16)).sum(-1) / (P.sum(-1) + 1e-9)
log(f'\n[B] peak delay tap fraction (taps 0-7): {(np.bincount(peak_tap, minlength=16)[:8] / peak_tap.size).round(4)}')
log(f'energy centroid per link: mean {centroid.mean():.2f}, std {centroid.std():.2f}, 95pct {np.percentile(centroid, 95):.2f}')

# ============ A. PCA compression floor (delay domain) ============
X = np.stack([G.real, G.imag], axis=-1).reshape(NS, -1).astype(np.float32)  # (NS, 2048)
Xc = X - X.mean(0, keepdims=True)
log('\n[A] PCA on delay-domain vec(H), 16 taps kept (2048 real dims)')
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = S ** 2
cum = np.cumsum(ev) / ev.sum()
for q in (0.9, 0.95, 0.99, 0.999):
    log(f'  dims for {q * 100:.1f}% energy: {(cum < q).sum() + 1}')
Xn = (Xc ** 2).sum()
for K in (192, 96, 48, 24):
    nmse = max(1.0 - cum[K - 1], 1e-12)
    log(f'  PCA top-{K:4d} dims -> NMSE = {nmse:.5f} ({-10 * np.log10(nmse):.1f} dB)')

# ============ C. azimuth / polarization structure ============
Hp = G.reshape(NS, UE, RX, 2, 8, 16)     # (NS, ue, rx, pol, pos, tap)
az = np.fft.fft(Hp, axis=-3)             # 8-point DFT over positions
az_p = (np.abs(az) ** 2).sum(-1).sum(2)  # (NS, ue, pol, pos)
dom_pol = az_p.max(-1).argmax(-1)        # (NS, ue)
dom_az = az_p.max(2).argmax(-1)          # (NS, ue)
log(f'\n[C] dominant-pol agreement between users: {(dom_pol[:, 0] == dom_pol[:, 1]).mean():.2f}; '
    f'dominant-azimuth-bin agreement: {(dom_az[:, 0] == dom_az[:, 1]).mean():.2f}')
pr = az_p / (az_p.sum(-1, keepdims=True) + 1e-9)
ent = -(pr * np.log(pr + 1e-12)).sum(-1) / np.log(8)
log(f'azimuth spectrum normalized entropy: mean {ent.mean():.3f} (1.0 = isotropic)')
pol_share = az_p.sum(-1)                 # (NS, ue, pol)
share0 = pol_share[..., 0] / (pol_share.sum(-1) + 1e-9)
log(f'pol0 power share: mean {share0.mean():.3f} (0.5 = balanced)')

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(peak_tap, bins=np.arange(-0.5, 8.5), density=True)
ax[0].set_title('peak delay tap (per link)'); ax[0].set_xlabel('tap')
ax[1].plot(np.arange(1, 2049), cum)
for K in (48, 96, 192):
    ax[1].axvline(K, ls='--', lw=0.7)
ax[1].set_xscale('log'); ax[1].set_title('PCA cumulative energy (2048 dims)')
ax[1].set_xlabel('top-K dims')
ax[2].hist(ent.ravel(), bins=60, density=True)
ax[2].set_title('azimuth entropy per (sample, ue)')
fig.tight_layout(); fig.savefig('figs/fig7_structure.png', dpi=130); plt.close(fig)

# ============ D. perfect-CSI scheme simulation under competition scoring ============
NSIM = 3000
idx2 = rng.choice(NS, NSIM, replace=False)
Hm = Hs[idx2].transpose(0, 4, 1, 2, 3).reshape(NSIM * SC, UE, RX, TX)
H1, H2 = Hm[:, 0], Hm[:, 1]
M = H1.shape[0]

# MU-ZF: project each user's channel onto the FULL null space of the other (14-dim)
Vh1 = np.linalg.svd(H1, full_matrices=True)[2]
Vh2 = np.linalg.svd(H2, full_matrices=True)[2]
N2 = Vh2[:, 2:, :].conj().transpose(0, 2, 1)   # (M, 16, 14) null basis of H2
N1 = Vh1[:, 2:, :].conj().transpose(0, 2, 1)
E1 = np.einsum('mrt,mtn->mrn', H1, N2)         # (M, 2, 14)
E2 = np.einsum('mrt,mtn->mrn', H2, N1)
zf1 = np.linalg.svd(E1, full_matrices=False, compute_uv=False)[:, 0] ** 2
zf2 = np.linalg.svd(E2, full_matrices=False, compute_uv=False)[:, 0] ** 2
s_1 = np.linalg.svd(H1, full_matrices=False, compute_uv=False)[:, 0] ** 2
s_2 = np.linalg.svd(H2, full_matrices=False, compute_uv=False)[:, 0] ** 2
su_gain = np.maximum(s_1, s_2)                 # best user matched-filter gain
del Hm, H1, H2, Vh1, Vh2, N1, N2, E1, E2

B_MAX = 1152
KS = (2, 4, 6, 8)
P_MU = 0.5

# ---- Monte-Carlo BER tables for square M-QAM (Gray mapping), vs per-symbol SINR (dB) ----
def build_ber_table(mc=200000, db_min=-10.0, db_max=40.0):
    dbs = np.arange(db_min, db_max + 0.5, 1.0)
    tables = {}
    for k in KS:
        Mp = 2 ** k
        m = int(np.sqrt(Mp))
        i, j = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
        gray = lambda x: x ^ (x >> 1)
        gi, gj = gray(i), gray(j)
        bits_i = ((gi[:, :, None] >> np.arange(k // 2 - 1, -1, -1)) & 1)
        bits_j = ((gj[:, :, None] >> np.arange(k // 2 - 1, -1, -1)) & 1)
        labels = np.concatenate([bits_i, bits_j], axis=-1).reshape(-1, k)  # (Mp, k)
        c = np.sqrt(3.0 / (2 * (Mp - 1)))
        pts = c * ((2 * i - m + 1) + 1j * (2 * j - m + 1)).ravel().astype(np.float64)
        ber = np.zeros_like(dbs)
        for bi, db in enumerate(dbs):
            n0 = 10 ** (-db / 10.0)
            true_idx = rng.integers(0, Mp, mc)
            x = pts[true_idx] + np.sqrt(n0 / 2) * (rng.standard_normal(mc) + 1j * rng.standard_normal(mc))
            ih = np.clip(np.rint(x.real / c + (m - 1) / 2), 0, m - 1).astype(int)
            jh = np.clip(np.rint(x.imag / c + (m - 1) / 2), 0, m - 1).astype(int)
            dec_idx = ih * m + jh
            ber[bi] = (labels[dec_idx] != labels[true_idx]).sum() / (mc * k)
        tables[k] = ber
    return dbs, tables

log('building Monte-Carlo BER tables (QPSK/16/64/256QAM)...')
ber_dbs, ber_tables = build_ber_table()


def ber_lookup(k, sinr_db):
    return np.interp(sinr_db, ber_dbs, ber_tables[k], left=0.5, right=0.0)


def score_fixed(snr_db, gain, pwr):
    sigma2 = 10 ** (-snr_db / 10.0)
    sinr_db = 10 * np.log10(np.maximum(gain * pwr / sigma2, 1e-12))
    ber = np.maximum(ber_lookup(8, sinr_db), 1e-6)
    return 100.0 * (1 - ber)


def score_adaptive(snr_db, gain, pwr):
    sigma2 = 10 ** (-snr_db / 10.0)
    sinr_db = 10 * np.log10(np.maximum(gain * pwr / sigma2, 1e-12))
    best = np.full(sinr_db.shape, 50.0)
    for k in KS:
        ber = np.maximum(ber_lookup(k, sinr_db), 1e-6)
        B = B_MAX * k // 8
        best = np.maximum(best, 100.0 * (B * (1 - ber) + 0.5 * (B_MAX - B)) / B_MAX)
    return best


snr1 = rng.uniform(-20, 20, M)
snr2 = rng.uniform(-20, 20, M)

mu1_f, mu2_f = score_fixed(snr1, zf1, P_MU), score_fixed(snr2, zf2, P_MU)
mu1_a, mu2_a = score_adaptive(snr1, zf1, P_MU), score_adaptive(snr2, zf2, P_MU)
su_served = np.where(s_1 >= s_2,
                     score_adaptive(snr1, su_gain, 1.0),
                     score_adaptive(snr2, su_gain, 1.0))
unserved = np.full(M, 50.0)
mu_pair = mu1_a + mu2_a
su_best = su_served
pick_mu = mu_pair >= su_best
oracle_u1 = np.where(pick_mu, mu1_a, unserved)
oracle_u2 = np.where(pick_mu, mu2_a, su_served)

schemes = {
    'MU-ZF fixed rate': np.concatenate([mu1_f, mu2_f]),
    'MU-ZF adaptive': np.concatenate([mu1_a, mu2_a]),
    'SU greedy adaptive': np.concatenate([su_served, unserved]),
    'Oracle switch adaptive': np.concatenate([oracle_u1, oracle_u2]),
}

log('\n[D] perfect-CSI scheme comparison (competition scoring)')
log(f'{"scheme":24s} {"eff":>7s} {"fair(p10)":>10s} {"final=0.7e+0.3f":>16s}')
for name, sc in schemes.items():
    eff = sc.mean()
    fair = np.percentile(sc, 10)
    log(f'{name:24s} {eff:7.2f} {fair:10.2f} {0.7 * eff + 0.3 * fair:16.2f}')

# Shannon-style ceiling: capacity-achieving coding across REs.
# Sending C bits/RE (C <= 8) with Bc=B: mu = 100*(0.5 + 0.5*B/Bmax) = 100*(0.5 + C/16)
def score_capacity(snr_db, gain, pwr):
    sigma2 = 10 ** (-snr_db / 10.0)
    c = np.log2(1 + gain * pwr / sigma2)
    return 100.0 * (0.5 + np.minimum(8.0, c) / 16.0)

cap_mu1 = score_capacity(snr1, zf1, P_MU)
cap_mu2 = score_capacity(snr2, zf2, P_MU)
cap_su_served = score_capacity(np.where(s_1 >= s_2, snr1, snr2), su_gain, 1.0)
pick_mu_cap = (cap_mu1 + cap_mu2) >= (cap_su_served + 50.0)
cap_rows = {
    'MU-ZF + ideal coding': np.concatenate([cap_mu1, cap_mu2]),
    'SU fullpower + ideal': np.concatenate([cap_su_served, np.full(M, 50.0)]),
    'Oracle + ideal coding': np.concatenate([np.where(pick_mu_cap, cap_mu1, np.full(M, 50.0)),
                                             np.where(pick_mu_cap, cap_mu2, cap_su_served)]),
}
log('')
for name, sc in cap_rows.items():
    eff = sc.mean()
    fair = np.percentile(sc, 10)
    log(f'{name:24s} {eff:7.2f} {fair:10.2f} {0.7 * eff + 0.3 * fair:16.2f}')
log(f'capacity-oracle picks MU in {pick_mu_cap.mean() * 100:.1f}% of (sample,sc)')

log(f'\n gain check: median zf1={np.median(zf1):.1f}, zf2={np.median(zf2):.1f}, su(s1^2)={np.median(su_gain):.1f}')
log(f'oracle picks MU in {(pick_mu).mean() * 100:.1f}% of (sample,sc)')

mean_snr = (snr1 + snr2) / 2
bins = np.arange(-20, 25, 5)
frac_mu = [(pick_mu[(mean_snr >= bins[i]) & (mean_snr < bins[i + 1])]).mean() for i in range(len(bins) - 1)]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(np.arange(len(frac_mu)), frac_mu)
ax[0].set_xticks(np.arange(len(frac_mu)))
ax[0].set_xticklabels([f'[{bins[i]},{bins[i+1]})' for i in range(len(bins) - 1)], rotation=30, fontsize=8)
ax[0].set_ylim(0, 1); ax[0].set_title('P(oracle picks MU) vs mean user SNR')
for name, sc in schemes.items():
    ax[1].hist(sc, bins=120, density=True, alpha=0.5, label=name, range=(40, 100))
ax[1].set_title('per-user score distributions'); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig('figs/fig8_switch.png', dpi=130); plt.close(fig)

with open('explore2_summary.txt', 'w') as f:
    f.write('\n'.join(log_lines) + '\n')
log('\nDONE')
