"""Explore H_train.npz channel statistics (CPU only).

Outputs: figs/*.png + explore_summary.txt
Run: python data_explore.py  (expects ./data_train/H_train.npz)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figs', exist_ok=True)
rng = np.random.default_rng(0)
log_lines = []


def log(s=''):
    print(s)
    log_lines.append(str(s))


log('loading data...')
d = np.load('./data_train/H_train.npz')
H = d['real'].astype(np.float32) + 1j * d['imag'].astype(np.float32)  # (N, ue, rx, tx, sc)
N, UE, RX, TX, SC = H.shape
log(f'shape {H.shape}, dtype {H.dtype}, storage fp16 (real/imag)')
log(f'raw real dtype {d["real"].dtype}, min {d["real"].min():.4f}, max {d["real"].max():.4f}')

NS = 20000  # subsample for heavy per-sample analyses
idx = rng.choice(N, NS, replace=False)
Hs = H[idx]  # (NS, ue, rx, tx, sc)

# ============ 1. basic power stats ============
p = np.abs(H) ** 2
log(f'\n[1] basic: E|h|^2 = {p.mean():.4f}, |h| percentiles 1/50/99 = '
    f'{np.percentile(np.abs(H), [1, 50, 99]).round(3)}')
per_entry = p.mean(axis=(0, 1, 2, 4))  # (tx,)
log(f'per-TX-antenna mean power: {per_entry.round(3)}')
log(f'per-RX-antenna mean power: {p.mean(axis=(0, 1, 3, 4)).round(3)}')
# LoS / deterministic component check: sample-mean of entries vs its std error
m = H[:1000].mean(axis=0)
se = H[:1000].std(axis=0) / np.sqrt(1000)
z = np.abs(m) / (se + 1e-9)
log(f'LoS check: max |E[h]|/(std/sqrt(N)) = {z.max():.1f} '
    f'({(z > 4).mean() * 100:.2f}% entries > 4sigma; ~5% expected if purely random)')

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(np.abs(H).ravel()[::37], bins=200, density=True)
ax[0].set_title('|h| histogram')
ax[1].plot(p.mean(axis=(0, 1, 2, 3)))
ax[1].set_title('mean power per subcarrier')
ax[1].set_xlabel('subcarrier')
ax[2].bar(range(TX), p.mean(axis=(0, 1, 2, 4)))
ax[2].set_title('mean power per TX antenna')
ax[2].set_xlabel('tx antenna')
fig.tight_layout()
fig.savefig('figs/fig1_power.png', dpi=130)
plt.close(fig)

# ============ 2. delay domain (PDP) ============
# ifft over subcarriers -> delay taps
G = np.fft.ifft(Hs, axis=-1)
pdp = (np.abs(G) ** 2).mean(axis=(0, 1, 2, 3))  # (144,)
pdp_db = 10 * np.log10(pdp / pdp.max() + 1e-12)
# delay spread from normalized PDP
t = np.arange(SC)
t0 = (t * pdp).sum() / pdp.sum()
t2 = ((t - t0) ** 2 * pdp).sum() / pdp.sum()
rms_ds = np.sqrt(t2)
log(f'\n[2] delay: RMS delay spread = {rms_ds:.2f} taps (grid {SC}); '
    f'PDP > -20dB taps = {(pdp_db > -20).sum()}')
log(f'taps 0-5 dB relative: {pdp_db[:6].round(1)}')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(pdp_db)
ax.axhline(-20, color='r', ls='--', lw=0.8)
ax.set_xlabel('delay tap'); ax.set_ylabel('dB'); ax.set_title('power delay profile (avg)')
fig.tight_layout(); fig.savefig('figs/fig2_delay.png', dpi=130); plt.close(fig)

# ============ 3. frequency correlation ============
h1 = Hs[:, 0]  # (NS, rx, tx, sc) one UE
num_off = SC // 2
fc = np.zeros(num_off)
for off in range(1, num_off + 1):
    a = h1[..., :-off]
    b = h1[..., off:]
    num = np.abs((a * np.conj(b)).mean())
    den = np.sqrt((np.abs(a) ** 2).mean() * (np.abs(b) ** 2).mean())
    fc[off - 1] = num / den
# coherence bw: smallest offset where |corr| < 0.5
below = np.where(fc < 0.5)[0]
coh = below[0] + 1 if len(below) else -1
log(f'\n[3] freq corr: |rho(offset)| first 5 = {fc[:5].round(3)}; '
    f'rho(47) = {fc[47]:.3f}, rho(48) = {fc[48]:.3f}')
log(f'coherence offset (|rho|<0.5): {coh} subcarriers '
    f'(subband = 48 sc)')

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(np.arange(1, num_off + 1), fc)
ax.axhline(0.5, color='r', ls='--', lw=0.8, label='0.5')
ax.axvline(48, color='g', ls='--', lw=0.8, label='subband=48')
ax.set_xlabel('offset (subcarriers)'); ax.set_ylabel('|corr|'); ax.legend()
ax.set_title('frequency correlation')
fig.tight_layout(); fig.savefig('figs/fig3_freqcorr.png', dpi=130); plt.close(fig)

# ============ 4. TX spatial correlation ============
xa = h1.reshape(NS * RX, TX, SC)  # (M, tx, sc)
C = np.zeros((TX, TX))
for a in range(TX):
    for b in range(a, TX):
        num = np.abs((xa[:, a] * np.conj(xa[:, b])).mean())
        den = np.sqrt((np.abs(xa[:, a]) ** 2).mean() * (np.abs(xa[:, b]) ** 2).mean())
        C[a, b] = C[b, a] = num / den
log(f'\n[4] TX spatial corr: mean |rho(a,b)| = {C[np.triu_indices(TX, 1)].mean():.3f}, '
    f'adjacent antennas = {np.mean([C[i, i+1] for i in range(TX-1)]):.3f}')

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(C, vmin=0, vmax=1, cmap='viridis')
fig.colorbar(im)
ax.set_title('TX antenna correlation'); ax.set_xlabel('tx a'); ax.set_ylabel('tx b')
fig.tight_layout(); fig.savefig('figs/fig4_spatial.png', dpi=130); plt.close(fig)

# ============ 5. SVD / streams ============
Hm = Hs.transpose(0, 4, 1, 2, 3).reshape(NS * SC, UE, RX, TX)  # (M, ue, rx, tx)
del Hs, h1
U, S, Vh = np.linalg.svd(Hm[:, 0], full_matrices=False)  # user0: (M, 2, 16) -> S (M, 2)
s1, s2 = S[:, 0], S[:, 1]
log(f'\n[5] SVD (user0, per sc): mean s1^2 = {(s1**2).mean():.2f}, mean s2^2 = {(s2**2).mean():.2f}')
r = s2 / (s1 + 1e-9)
log(f's2/s1 percentiles 10/50/90 = {np.percentile(r, [10, 50, 90]).round(3)}; '
    f'frac s2/s1 < 0.3 (single-stream dominant): {(r < 0.3).mean():.3f}')
# per-sample fluctuation of best-beam gain across sc (rate adaptation headroom)
g = (s1 ** 2).reshape(NS, SC)
cv = g.std(axis=1) / (g.mean(axis=1) + 1e-9)
log(f'per-sample across-sc fluctuation of s1^2 (std/mean): '
    f'10/50/90 pct = {np.percentile(cv, [10, 50, 90]).round(2)}')

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(10 * np.log10(s1 ** 2 + 1e-9), bins=150, density=True, alpha=0.7, label='s1^2 dB')
ax[0].hist(10 * np.log10(s2 ** 2 + 1e-9), bins=150, density=True, alpha=0.7, label='s2^2 dB')
ax[0].legend(); ax[0].set_title('singular value power (dB)')
ax[1].hist(r, bins=150, density=True)
ax[1].set_title('s2/s1 ratio'); ax[1].set_xlabel('s2/s1')
fig.tight_layout(); fig.savefig('figs/fig5_svd.png', dpi=130); plt.close(fig)

# ============ 6. inter-user correlation (MU pairing) ============
v1 = Vh[:, 0, :].conj()  # (M, tx) dominant beam of user0 (v = Vh[0].conj())
H2 = Hm[:, 1]            # (M, rx, tx)
proj = np.einsum('mrt,mt->mr', H2, v1)           # H2 @ v1
interf = (np.abs(proj) ** 2).sum(-1) / ((np.abs(H2) ** 2).reshape(NS * SC, -1).sum(-1) + 1e-9)
log(f'\n[6] MU: frac of user1-beam energy inside user2 channel subspace: '
    f'10/50/90 pct = {np.percentile(interf, [10, 50, 90]).round(3)}')
log(f'frac samples-SC with interference > 0.5 (bad for pairing): {(interf > 0.5).mean():.3f}')

a = Hm[:, 0].reshape(NS * SC, -1)
b = Hm[:, 1].reshape(NS * SC, -1)
rho = np.abs((a * np.conj(b)).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
log(f'cross-user |rho| (vec H1 vs vec H2): 10/50/90 pct = {np.percentile(rho, [10, 50, 90]).round(3)}')

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(interf, bins=150, density=True)
ax[0].set_title('beam-energy fraction leaking into other UE'); ax[0].set_xlabel('frac')
ax[1].hist(rho, bins=150, density=True)
ax[1].set_title('cross-user |corr| of vec(H)'); ax[1].set_xlabel('|rho|')
fig.tight_layout(); fig.savefig('figs/fig6_users.png', dpi=130); plt.close(fig)

with open('explore_summary.txt', 'w') as f:
    f.write('\n'.join(log_lines) + '\n')
log('\nDONE')
