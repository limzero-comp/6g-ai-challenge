"""Smoke test: data loading + full link forward/backward on the server env."""
import time
import numpy as np
import torch
import torch.nn as nn
from modelDesign import Encoder, Transmitter, Receiver

NUM_UE = 2
NUM_UPLINK_SUBCARRIERS = 96
NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_CTRL_BITS = 5
NUM_DOWNLINK_TX = 16
NUM_MAX_BITS = 144 * 8
SNR_DL_RANGE = [-20, 20]
SNR_UL_GAP = 10
DEVICE = torch.device('mps')

print(f'torch {torch.__version__} | numpy {np.__version__} | device {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'gpu: {torch.cuda.get_device_name(0)}')


class MU_MIMO_Link(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder()
        self._transmitter = Transmitter()
        self._receiver = Receiver()
        self._num_ue = NUM_UE

    def forward(self, h_list, b_list, snr_dl, snr_ul):
        batch_size = h_list[0].shape[0]
        H_list = [torch.as_tensor(h_i, device=DEVICE) for h_i in h_list]
        b_list = [b.to(DEVICE) for b in b_list]
        snr_dl = snr_dl.to(DEVICE)
        snr_ul = snr_ul.to(DEVICE)

        I_list = []
        for i in range(self._num_ue):
            U = self._encoder(H_list[i], snr_dl[i])
            assert U.shape == (batch_size, NUM_UPLINK_SUBCARRIERS) and U.dtype == torch.complex64
            energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
            assert torch.all(energy > 0)
            U = U / torch.sqrt(energy)
            g = torch.complex(torch.randn_like(U, dtype=torch.float32) / 1.4142, torch.randn_like(U, dtype=torch.float32) / 1.4142)
            I_list.append(U + g * torch.sqrt(torch.reshape(10 ** (-snr_ul[i] / 10.0), [-1, 1])))

        X, b_ctrl = self._transmitter(b_list, I_list, snr_dl)
        assert X.shape == (batch_size, NUM_DOWNLINK_TX, NUM_DOWNLINK_DATA_SUBCARRIERS) and X.dtype == torch.complex64
        assert b_ctrl.shape == (batch_size, NUM_DOWNLINK_CTRL_BITS)
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2), keepdim=True)
        assert torch.all(energy > 0)
        X = X / torch.sqrt(energy)

        c_list = []
        X_exp = X.unsqueeze(1)
        for i in range(self._num_ue):
            Y = torch.sum(H_list[i] * X_exp, dim=2)
            g = torch.complex(torch.randn_like(Y, dtype=torch.float32) / 1.4142, torch.randn_like(Y, dtype=torch.float32) / 1.4142)
            Y = Y + g * torch.sqrt(torch.reshape(10 ** (-snr_dl[i] / 10.0), [-1, 1, 1]))
            llr = self._receiver(Y, H_list[i], b_ctrl, snr_dl[i])
            assert llr.ndim == 2 and 0 < llr.shape[1] <= NUM_MAX_BITS and llr.dtype == torch.float32
            assert torch.isfinite(llr).all()
            c_list.append(llr)
        return c_list


t0 = time.time()
d = np.load('./data_train/H_train.npz')
print(f'npz keys: {list(d.keys())}, load time {time.time()-t0:.1f}s')
H = d['real'].astype(np.float32) + 1j * d['imag'].astype(np.float32)
print(f'H_train shape: {H.shape}, dtype: {H.dtype}')
print(f'|H| mean: {np.abs(H).mean():.4f}, |H| std: {np.abs(H).std():.4f}')

mu_link = MU_MIMO_Link().to(DEVICE)
print(f'params: {sum(p.numel() for p in mu_link.parameters())/1e6:.2f}M')

BATCH = 100
idx = np.random.choice(H.shape[0], BATCH, replace=False)
h_batch = H[idx]
h_list = [torch.as_tensor(h_batch[:, i], device=DEVICE) for i in range(NUM_UE)]
b_list = [torch.randint(0, 2, (BATCH, NUM_MAX_BITS), dtype=torch.float32, device=DEVICE) for _ in range(NUM_UE)]
snr_dl = SNR_DL_RANGE[0] + (SNR_DL_RANGE[1] - SNR_DL_RANGE[0]) * torch.rand(NUM_UE, BATCH, device=DEVICE)
snr_ul = snr_dl - SNR_UL_GAP

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(mu_link.parameters(), lr=1e-4)

# warmup + timed forward/backward
for step in range(3):
    t0 = time.time()
    llr_list = mu_link(h_list, b_list, snr_dl, snr_ul)
    loss = sum(criterion(llr_list[j], b_list[j]) for j in range(NUM_UE))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    acc = sum(((llr_list[j] >= 0).float() == b_list[j]).float().mean().item() for j in range(NUM_UE)) / NUM_UE
    print(f'step {step}: loss {loss.item():.4f} acc {acc:.4f} | {(time.time()-t0)*1000:.0f} ms')

mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == 'cuda' else 0
print(f'peak gpu mem: {mem:.2f} GB')
print('SMOKE TEST OK')
