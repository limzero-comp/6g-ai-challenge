"""Reproducible CPU-only audit of the supplied 6G channels; never trains a model.

Example:
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python analysis/audit_data.py \
      --data /path/to/H_train.npz --out reports

PCA here is a statistical diagnostic, fitted only to the audit training split.
All reported reconstruction errors include omitted frequency/delay components.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np


def quantiles(x):
    return {str(q): float(np.percentile(x, q)) for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)}


def nmse_summary(error, power):
    ratio = error / np.maximum(power, 1e-30)
    return {"ratio_of_total_energy": float(error.sum() / power.sum()),
            "mean_per_example_nmse": float(ratio.mean()), "per_example_quantiles": quantiles(ratio)}


def progress(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def complex_chunk(real, imag, selection):
    return real[selection].astype(np.float32) + 1j * imag[selection].astype(np.float32)


def inspect_all(real, imag, chunk):
    n = len(real)
    energy = np.empty((n, real.shape[1]), dtype=np.float64)
    channel_power = np.zeros(real.shape[1:], dtype=np.float64)
    total = {"shape": list(real.shape), "real_dtype": str(real.dtype), "imag_dtype": str(imag.dtype),
             "axes": ["sample", "UE", "RX", "TX", "subcarrier"], "n_values_per_component": int(real.size)}
    counts = {k: {"nonfinite": 0, "zeros": 0, "min": float("inf"), "max": -float("inf")} for k in ("real", "imag")}
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        for name, array in (("real", real[sl]), ("imag", imag[sl])):
            counts[name]["nonfinite"] += int((~np.isfinite(array)).sum())
            counts[name]["zeros"] += int((array == 0).sum())
            counts[name]["min"] = min(counts[name]["min"], float(array.min()))
            counts[name]["max"] = max(counts[name]["max"], float(array.max()))
        h = complex_chunk(real, imag, sl)
        p = np.abs(h) ** 2
        energy[sl] = p.mean(axis=(2, 3, 4), dtype=np.float64)
        channel_power += p.sum(axis=0, dtype=np.float64)
        if start % (20 * chunk) == 0:
            progress("quality scan %d/%d" % (start, n))
    total["components"] = counts
    total["zero_energy_sample_ue_count"] = int((energy == 0).sum())
    total["mean_element_power"] = float(energy.mean())
    total["sample_ue_mean_power_quantiles"] = quantiles(energy)
    total["per_ue_mean_power"] = energy.mean(0).tolist()
    total["per_tx_mean_power"] = (channel_power / n).mean(axis=(0, 1, 3)).tolist()
    total["per_rx_mean_power"] = (channel_power / n).mean(axis=(0, 2, 3)).tolist()
    total["ue_power_ratio_quantiles"] = quantiles(energy[:, 0] / np.maximum(energy[:, 1], 1e-30))
    blocks = np.array_split(np.arange(n), 20)
    total["ordered_20_blocks"] = [{"start": int(b[0]), "end_exclusive": int(b[-1] + 1),
                                      "mean_power_per_ue": energy[b].mean(0).tolist(),
                                      "p10_power_per_ue": np.percentile(energy[b], 10, axis=0).tolist()}
                                     for b in blocks]
    total["adjacent_sample_energy_correlation_per_ue"] = [float(np.corrcoef(energy[:-1, u], energy[1:, u])[0, 1]) for u in range(2)]
    return total, energy


def structure(h):
    # Unitary IFFT preserves power exactly; last axis is frequency.
    g = np.fft.ifft(h, axis=-1, norm="ortho").astype(np.complex64)
    p = np.abs(g) ** 2
    per = p.sum(axis=(2, 3))
    total = per.sum(-1)
    pdp = per.sum(axis=(0, 1)) / total.sum()
    out = {"sample_count": len(h), "pdp_energy_fraction_144": pdp.tolist(),
           "leading_delay_captured_energy": {}, "trailing_delay_energy_fraction": {},
           "subband_mean_channel_nmse": {}}
    for k in (1, 2, 4, 8, 16, 32, 64):
        out["leading_delay_captured_energy"][str(k)] = quantiles(per[..., :k].sum(-1) / total)
    for k in (1, 4, 8, 16):
        out["trailing_delay_energy_fraction"][str(k)] = float(per[..., -k:].sum() / total.sum())
    for size in (12, 24, 48, 72, 144):
        grouped = h.reshape(*h.shape[:-1], 144 // size, size)
        err = np.abs(grouped - grouped.mean(-1, keepdims=True)) ** 2
        out["subband_mean_channel_nmse"][str(size)] = nmse_summary(err.sum(axis=(2, 3, 4, 5)), (np.abs(h) ** 2).sum(axis=(2, 3, 4)))
    out["frequency_correlation"] = {}
    for lag in (1, 2, 4, 8, 12, 24, 48, 72):
        a, b = h[..., :-lag], h[..., lag:]
        cov = np.sum(a.conj() * b, dtype=np.complex128)
        den = np.sqrt(np.sum(np.abs(a) ** 2, dtype=np.float64) * np.sum(np.abs(b) ** 2, dtype=np.float64))
        out["frequency_correlation"][str(lag)] = {"magnitude": float(abs(cov) / den), "phase_rad": float(np.angle(cov))}
    # TX indices 0..7 and 8..15 are a statistical grouping, not verified hardware metadata.
    hp = g.reshape(len(h), 2, 2, 2, 8, 144)
    beams = np.fft.fft(hp, axis=4, norm="ortho")  # axis=4 is position, axis=3 is group.
    bp = (np.abs(beams) ** 2).sum(axis=(2, 5))  # sample,UE,group,beam
    probs = bp / bp.sum(-1, keepdims=True)
    ent = -(probs * np.log(np.maximum(probs, 1e-30))).sum(-1) / np.log(8)
    pooled = bp.sum(axis=(0, 1, 2)); pooled /= pooled.sum()
    dom = bp.sum(2).argmax(-1)
    out["spatial"] = {"fft_axis": "position axis=4 of [sample,UE,RX,group,position,delay]",
                       "per_sample_ue_group_beam_entropy": quantiles(ent),
                       "pooled_beam_energy_fraction": pooled.tolist(),
                       "pooled_beam_entropy": float(-(pooled * np.log(pooled)).sum() / np.log(8)),
                       "users_dominant_beam_agreement": float((dom[:, 0] == dom[:, 1]).mean()),
                       "group0_energy_fraction": float(bp[:, :, 0].sum() / bp.sum())}
    # Covariance on TX, averaged over sampled channels / RX / frequency.
    spatial_cov = np.einsum("nurts,nurvs->tv", h.conj(), h, optimize=True)
    diag = np.sqrt(np.real(np.diag(spatial_cov)))
    corr = np.abs(spatial_cov) / np.outer(diag, diag)
    out["spatial"]["tx_abs_correlation"] = corr.tolist()
    out["spatial"]["adjacent_within_group_correlation"] = float(np.mean([corr[i, i + 1] for i in list(range(7)) + list(range(8, 15))]))
    out["spatial"]["cross_group_mean_abs_correlation"] = float(corr[:8, 8:].mean())
    # Quantization bounds refer to unavailable pre-rounded data, not an exact error measurement.
    step_real = np.abs(np.spacing(np.abs(h.real).astype(np.float16)).astype(np.float64))
    step_imag = np.abs(np.spacing(np.abs(h.imag).astype(np.float16)).astype(np.float64))
    spacing_sq = (step_real ** 2 + step_imag ** 2).sum()
    signal_sq = (np.abs(h) ** 2).sum(dtype=np.float64)
    out["fp16_rounding_scale"] = {"uniform_rounding_model_nmse": float(spacing_sq / (12 * signal_sq)),
                                  "half_spacing_squared_bound_nmse": float(spacing_sq / (4 * signal_sq)),
                                  "caveat": "Estimate/bound for ordinary nearest rounding without overflow; original unrounded samples unavailable."}
    return out, g


def pca_diagnostics(g, train_n, test_n):
    # Separate original sample indices ensure a UE's paired counterpart cannot cross this split.
    train = g[:train_n, 0]
    test = g[train_n:train_n + test_n]  # evaluate both UEs separately
    train_pdp = (np.abs(train) ** 2).sum(axis=(0, 1, 2))
    tapsets = {"leading16": np.arange(16), "train_selected16": np.sort(np.argsort(train_pdp)[-16:]),
               "leading32": np.arange(32)}
    out = {"train_original_samples": train_n, "holdout_original_samples": test_n,
           "fit_ue": 0, "feedback_budget_complex": 96, "feedback_budget_real": 192,
           "metric": "full 144-frequency reconstruction, including omitted delay energy; no feedback noise",
           "variants": {}}
    for name, taps in tapsets.items():
        progress("PCA " + name)
        a = train[..., taps].reshape(train_n, -1)
        x = np.concatenate((a.real, a.imag), axis=1).astype(np.float64)
        mean = x.mean(0)
        x -= mean
        covariance = x.T @ x / (len(x) - 1)
        ev, basis = np.linalg.eigh(covariance)
        ev, basis = np.maximum(ev[::-1], 0), basis[:, ::-1].copy()
        cumulative = ev.cumsum() / ev.sum()
        row = {"selected_taps": taps.tolist(), "real_dimension": x.shape[1],
               "train_dims_for_selected_tap_variance": {str(q): int(np.searchsorted(cumulative, q) + 1) for q in (.9, .95, .99)},
               "holdout": {}}
        for ue in (0, 1):
            a = test[:, ue][..., taps]
            a = a.reshape(test_n, -1)
            xt = np.concatenate((a.real, a.imag), axis=1).astype(np.float64)
            xt -= mean
            energy = (np.abs(test[:, ue]) ** 2).sum(axis=(1, 2, 3), dtype=np.float64)
            kept_energy = (np.abs(test[:, ue][..., taps]) ** 2).sum(axis=(1, 2, 3), dtype=np.float64)
            omitted = np.maximum(energy - kept_energy, 0)
            coeff = xt @ basis[:, :384]
            err0 = (xt ** 2).sum(1)
            ue_out = {"omitted_delay_nmse": nmse_summary(omitted, energy), "rank": {}}
            for rank in (24, 48, 96, 192, 384):
                residual = np.maximum(err0 - (coeff[:, :rank] ** 2).sum(1), 0) + omitted
                ue_out["rank"][str(rank)] = nmse_summary(residual, energy)
            row["holdout"][str(ue)] = ue_out
        out["variants"][name] = row
    return out


def qam_table(rng, mc):
    dbs = np.arange(-30, 41, 1, dtype=float)
    tables = {}
    checks = {}
    for k in (2, 4, 6, 8):
        m = 2 ** (k // 2)
        ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        gi, gj = ii ^ (ii >> 1), jj ^ (jj >> 1)
        bits = np.concatenate((((gi[..., None] >> np.arange(k // 2)) & 1), ((gj[..., None] >> np.arange(k // 2)) & 1)), axis=-1).reshape(-1, k)
        c = np.sqrt(3 / (2 * (2 ** k - 1)))
        points = c * ((2 * ii - m + 1) + 1j * (2 * jj - m + 1)).ravel()
        def demod(z):
            # The 2*c denominator is required because adjacent levels differ by 2*c.
            ih = np.clip(np.rint((z.real / c + m - 1) / 2), 0, m - 1).astype(int)
            jh = np.clip(np.rint((z.imag / c + m - 1) / 2), 0, m - 1).astype(int)
            return ih * m + jh
        checks[str(k)] = {"noiseless_errors": int((demod(points) != np.arange(2 ** k)).sum()),
                          "average_constellation_power": float((np.abs(points) ** 2).mean())}
        true = rng.integers(0, 2 ** k, size=mc)
        noise = (rng.standard_normal(mc) + 1j * rng.standard_normal(mc)) / np.sqrt(2)
        ber = [float((bits[demod(points[true] + noise * 10 ** (-db / 20))] != bits[true]).mean()) for db in dbs]
        tables[k] = np.array(ber)
    return dbs, tables, checks


def reference_simulation(h, rng, mc):
    n, ue, rx, tx, sc = h.shape
    hm = h.transpose(0, 4, 1, 2, 3).reshape(-1, ue, rx, tx)
    su, zf = [], []
    identities = np.eye(tx, dtype=np.complex64)
    for u in range(2):
        own, other = hm[:, u], hm[:, 1 - u]
        _, _, vh = np.linalg.svd(other, full_matrices=False)
        projection = identities - vh.conj().transpose(0, 2, 1) @ vh
        projected = own @ projection
        zf.append((np.linalg.svd(projected, compute_uv=False)[:, 0] ** 2).reshape(n, sc))
        su.append((np.linalg.svd(own, compute_uv=False)[:, 0] ** 2).reshape(n, sc))
    zf, su = np.stack(zf, 1), np.stack(su, 1)
    snr = rng.uniform(-20, 20, size=(n, 2, 1))  # fixed for all 144 REs in each sample and UE
    dbs, tables, checks = qam_table(rng, mc)
    def per_qam(gain, power, k):
        db = snr + 10 * np.log10(np.maximum(gain * power, 1e-30))
        ber = np.interp(db, dbs, tables[k], left=.5, right=0.)
        return 50 + (100 * k / 8) * (.5 - ber)
    fixed = per_qam(zf, .5, 8).mean(-1)
    mu_options = np.stack([per_qam(zf, .5, k).mean(-1) for k in (2, 4, 6, 8)], -1)
    # One constellation per entire sample/UE, not oracle selection at every RE.
    adaptive = mu_options.max(-1)
    su_options = np.stack([per_qam(su, 1., k).mean(-1) for k in (2, 4, 6, 8)], -1).max(-1)
    selected = su_options.argmax(-1)
    su_score = np.full((n, 2), 50.)
    su_score[np.arange(n), selected] = su_options[np.arange(n), selected]
    switch = np.where((adaptive.sum(-1) >= su_score.sum(-1))[:, None], adaptive, su_score)
    c_mu = np.log2(1 + zf * .5 * 10 ** (snr / 10)).mean(-1)
    cap_mu = 50 + 6.25 * np.minimum(8, c_mu)
    cap_su_options = 50 + 6.25 * np.minimum(8, np.log2(1 + su * 10 ** (snr / 10)).mean(-1))
    chosen = cap_su_options.argmax(-1)
    cap_su = np.full((n, 2), 50.)
    cap_su[np.arange(n), chosen] = cap_su_options[np.arange(n), chosen]
    cap_switch = np.where((cap_mu.sum(-1) >= cap_su.sum(-1))[:, None], cap_mu, cap_su)
    schemes = {"perfect_csi_mu_zf_256qam": fixed, "perfect_csi_mu_zf_sample_adaptive_qam": adaptive,
               "perfect_csi_su_adaptive_qam": su_score, "sum_score_oracle_mu_su_qam": switch,
               "mu_zf_ideal_reliable_prefix_reference": cap_mu, "su_ideal_reliable_prefix_reference": cap_su,
               "sum_score_oracle_ideal_reliable_prefix_reference": cap_switch}
    rows = {}
    for name, s in schemes.items():
        mean, p10 = float(s.mean()), float(np.percentile(s, 10))
        rows[name] = {"efficiency": mean, "fairness_p10": p10, "weighted_score": .7 * mean + .3 * p10}
    return {"original_samples": n, "ber_table_symbols_per_point": mc, "qam_checks": checks,
            "snr_policy": "one independent Uniform[-20,20] draw per sample and UE, broadcast across 144 RE",
            "aggregation": "expected bit accuracy averaged over all 144 RE first, then p10 across sample-UE scores",
            "limitations": ["Perfect CSI and perfect beam-aware receiver are unavailable to a real submitted transmitter/receiver.",
                            "Expected BER scores omit finite 1152-bit realization variance; this is not the official evaluator.",
                            "Reliable-prefix Shannon reference is not an upper bound for a bitwise accuracy/Hamming distortion objective.",
                            "The sum-score oracle optimizes each pair's mean, not the global p10 or competition score.",
                            "96 noisy feedback uses, pilot cost and control-bit feasibility are not represented."],
            "median_gain": {"mu_zf": float(np.median(zf)), "su_matched": float(np.median(su))},
            "schemes": rows, "qam_ber_table": {"db": dbs.tolist(), **{str(k): v.tolist() for k, v in tables.items()}}}


def beam_frequency_diagnostics(h):
    # Perfect CSI SVD is only used for deterministic diagnostics, not trained models.
    hc = h.transpose(0, 1, 4, 2, 3)
    _, s, vh = np.linalg.svd(hc, full_matrices=False)
    beam = vh[..., 0, :].conj()
    out = {"second_to_first_singular_value_ratio": quantiles(s[..., 1] / s[..., 0]), "principal_beam_overlap": {}, "subband_mean_beam_gain_ratio": {}}
    for lag in (1, 12, 24, 48, 72):
        overlap = np.abs(np.sum(beam[:, :, :-lag].conj() * beam[:, :, lag:], -1)) ** 2
        out["principal_beam_overlap"][str(lag)] = quantiles(overlap)
    for width in (12, 24, 48, 72, 144):
        mean_h = hc.reshape(len(h), 2, 144 // width, width, 2, 16).mean(3)
        mean_vh = np.linalg.svd(mean_h, full_matrices=False)[2]
        mean_beam = np.repeat(mean_vh[..., 0, :].conj(), width, axis=2)
        gain = np.abs(np.einsum("nufrt,nuft->nufr", hc, mean_beam)) ** 2
        gain = gain.sum(-1)
        ratio = gain.sum(-1) / (s[..., 0] ** 2).sum(-1)
        out["subband_mean_beam_gain_ratio"][str(width)] = quantiles(ratio)
    return out


def write_report(result, path):
    q = result["quality"]; st = result["structure"]; pca = result["pca"]; ref = result["reference_simulation"]
    f = lambda x: "%.5f" % x
    lines = ["# 独立数据审计（未训练任何模型）", "",
             "本报告由 `analysis/audit_data.py` 从真实 NPZ 重新计算，原始数据未修改；PCA 只用于压缩诊断。完整数值与抽样索引见 `data_audit.json`。", "",
             "## 数据与复现范围", "",
             "- 原始路径：`%s`。" % result["metadata"]["data_path"],
             "- shape `%s`；轴为 sample / UE / RX / TX / SC；real、imag 均为 `%s`。" % (q["shape"], q["real_dtype"]),
             "- 全量扫描 %d 条；结构分析 %d 条；PCA UE0训练 %d 条，独立原始样本留出 %d 条（分别评估两UE）；固定随机种子 %d。" % (q["shape"][0], st["sample_count"], pca["train_original_samples"], pca["holdout_original_samples"], result["metadata"]["seed"]),
             "- 非有限值 real=%d / imag=%d；零能量 sample-UE=%d；平均元素功率 %.6f。" % (q["components"]["real"]["nonfinite"], q["components"]["imag"]["nonfinite"], q["zero_energy_sample_ue_count"], q["mean_element_power"]),
             "- sample-UE 平均功率 p1/p10/p50/p90/p99 = %s。" % " / ".join(f(q["sample_ue_mean_power_quantiles"][str(x)]) for x in (1,10,50,90,99)),
             "- 抽样完全重复条数：%d（只验证本次抽样，不能证明全数据无重复）。" % result["sample_duplicate_check"]["duplicates"],
             "- 相邻样本能量相关：%s；20个顺序分块均值见 JSON，可据此判断顺序切分是否稳健。" % q["adjacent_sample_energy_correlation_per_ue"], "",
             "## 频率、时延与阵位结构", "",
             "前4/8/16/32个时延抽头的样本-UE能量份额中位数为 %s。" % " / ".join(f(st["leading_delay_captured_energy"][str(k)]["50"]) for k in (4,8,16,32)),
             "尾端4个 IFFT 抽头仍占全局能量 %.4f%%。离散频段/非整数时延可产生环绕泄漏，尾部不能直接归为量化噪声。" % (100 * st["trailing_delay_energy_fraction"]["4"]),
             "fp16 均匀舍入模型 NMSE≈%.3e，半间距平方界≈%.3e；未获得舍入前数据，不能测量实际量化误差。" % (st["fp16_rounding_scale"]["uniform_rounding_model_nmse"], st["fp16_rounding_scale"]["half_spacing_squared_bound_nmse"]),
             "正确的8阵位 FFT 轴为 `[sample,UE,RX,group,position,delay]` 的 axis=4；两组是否确为物理双极化仍是建模假设。每样本/UE/组的谱熵中位数 %.4f，全体池化谱熵 %.4f；二者回答不同问题。" % (st["spatial"]["per_sample_ue_group_beam_entropy"]["50"], st["spatial"]["pooled_beam_entropy"]), "",
             "| 每个子带 SC 数 | 平均H重建全频NMSE（能量加权） | 均值H的主波束增益 / 逐SC最优（中位） |", "|---:|---:|---:|"]
    for k in (12,24,48,72,144):
        lines.append("| %d | %.5f | %.5f |" % (k, st["subband_mean_channel_nmse"][str(k)]["ratio_of_total_energy"], result["beam_frequency"]["subband_mean_beam_gain_ratio"][str(k)]["50"]))
    lines += ["", "平均功率变化小不能推出信道方向不变；应结合上表波束损失选择预编码粒度。主波束方向可能有相位/奇异值退化问题，因此同时报告实际响应增益。", "",
              "## 单UE PCA：必须把截断损失加回", "",
              "每 UE 原始为 2×16×144=4608 复数；16 tap 为1024实维。旧脚本把两UE拼接成2048实维，再与单UE192实维反馈预算比较，预算口径不一致。以下只在 UE0 拟合，独立样本测试；模型未见留出样本。选tap版本只按训练集PDP选tap。PCA没有上行噪声、也没有每条反馈功率归一化，因此只是确定性压缩参照。", "",
              "| tap 方案 | 留出UE | 96实维 full NMSE | 192实维 full NMSE | 192维样本NMSE p90 | 丢弃tap误差 |", "|---|---:|---:|---:|---:|---:|"]
    for name,row in pca["variants"].items():
        for u, vals in row["holdout"].items():
            lines.append("| %s | %s | %.5f | %.5f | %.5f | %.5f |" % (name,u,vals["rank"]["96"]["ratio_of_total_energy"],vals["rank"]["192"]["ratio_of_total_energy"],vals["rank"]["192"]["per_example_quantiles"]["90"],vals["omitted_delay_nmse"]["ratio_of_total_energy"]))
    lines += ["", "## 反馈信道才是关键瓶颈", "", "假设复数 AWGN、单位平均功率、96次使用，容量参考为 `96 log2(1+10^((SNR_DL-10)/10))` bit/用户。该渐近容量不保证96符号有限码长可实现，也不等于192个可靠实数。", "", "| DL dB | UL dB | 每96符号容量 bit |", "|---:|---:|---:|"]
    for row in result["uplink_capacity_reference"]:
        lines.append("| %d | %d | %.3f |" % (row["snr_dl_db"],row["snr_ul_db"],row["capacity_bits_per_96_complex_uses"]))
    lines += ["", "低SNR用户的CSI反馈应极度收缩，依赖统计先验/固定波束与接收端已知真实H；中高SNR才提高反馈细节。对全SNR共用一个追求低NMSE的反馈目标并不稳妥。", "",
              "## 修正后的 perfect-CSI 参考实验", "",
              "QAM判决已使用 `(x/c + m-1)/2`，所有星座无噪声判决零错误且单位平均能量。每个原始样本/UE只采一次SNR，144RE平均后计算p10；SU选择考虑各自SNR并正确归属用户，MU/SU比较包含未服务用户50分。", "",
              "| 方案 | 效率 | p10 | 0.7效率+0.3公平 |", "|---|---:|---:|---:|"]
    for name,row in ref["schemes"].items():
        lines.append("| %s | %.3f | %.3f | %.3f |" % (name,row["efficiency"],row["fairness_p10"],row["weighted_score"]))
    lines += ["", "这些数值是有限抽样的设计参照，不是官方得分预测，更不是物理上界。它们使用完美CSI/理想等效接收、未计反馈和导频成本、以期望BER代替1152个有限比特的真实正确率。Shannon可靠前缀参考只比较一种编码目标，不能为按位正确率建立上界；全局p10还会受非线性评分与比特噪声影响。", "",
              "## 由数据支持的建模顺序", "",
              "1. 首先建立明确可解调的波形与共享控制码，检验低SNR反馈失效时仍可工作。",
              "2. 采用SNR条件化反馈、训练集固定降维先验和噪声感知重建；前16tap以外的泄漏要显式考虑。",
              "3. 用细粒度频率特征/波束生成避免把48SC均值当作无损；可从12或24SC组开始做消融。",
              "4. 同时保留固定QAM、分层或冗余编码、MU与受控SU方案；当前审计不足以宣布永远MU、放弃调制自适应或公平性已到物理墙。",
              "5. 后续训练时按原始sample划分，固定验证SNR与随机比特，多seed报告均值和p10；不要将同一pair的两个UE拆入不同集合。", "",
              "复现参数、耗时与所有抽样索引均保存于 JSON。图见 `data_audit_overview.png`。"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def make_plot(result, energy, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0,0].hist(energy.ravel(), bins=100, density=True); ax[0,0].set(xlabel="Mean |H|^2 per sample/UE",title="All-sample power distribution")
    blocks=result["quality"]["ordered_20_blocks"]
    ax[0,1].plot([b["mean_power_per_ue"] for b in blocks]); ax[0,1].set(title="Ordered blocks: power",xlabel="Block (20 equal parts)")
    pdp=np.array(result["structure"]["pdp_energy_fraction_144"])
    ax[0,2].semilogy(np.arange(144),pdp); ax[0,2].set(title="144-tap IFFT energy",xlabel="Delay tap",ylabel="Energy fraction")
    im=ax[1,0].imshow(result["structure"]["spatial"]["tx_abs_correlation"],vmin=0,vmax=1); fig.colorbar(im,ax=ax[1,0]); ax[1,0].set(title="TX absolute correlation",xlabel="TX",ylabel="TX")
    for name,row in result["pca"]["variants"].items():
        vals=row["holdout"]["0"]["rank"]
        ax[1,1].plot([int(k) for k in vals],[v["ratio_of_total_energy"] for v in vals.values()],marker="o",label=name)
    ax[1,1].set(xscale="log",yscale="log",xlabel="Retained real dimensions",ylabel="Full-frequency holdout NMSE",title="Single-UE PCA incl. tail"); ax[1,1].legend(fontsize=8)
    cap=result["uplink_capacity_reference"]
    ax[1,2].semilogy([v["snr_dl_db"] for v in cap],[v["capacity_bits_per_96_complex_uses"] for v in cap],marker="o"); ax[1,2].set(xlabel="DL SNR (dB); UL = DL - 10",ylabel="bits per 96 complex uses",title="AWGN capacity reference")
    fig.tight_layout(); fig.savefig(out / "data_audit_overview.png",dpi=150); plt.close(fig)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path); parser.add_argument("--out",type=Path,default=Path("reports"))
    parser.add_argument("--seed",type=int,default=20260918); parser.add_argument("--sample",type=int,default=4000)
    parser.add_argument("--pca-train",type=int,default=2500); parser.add_argument("--pca-test",type=int,default=1000)
    parser.add_argument("--sim-samples",type=int,default=1000); parser.add_argument("--ber-symbols",type=int,default=80000)
    parser.add_argument("--chunk",type=int,default=512)
    args=parser.parse_args(); start=time.time(); args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    progress("loading original fp16 arrays")
    archive=np.load(args.data)
    real,imag=archive["real"],archive["imag"]
    if real.shape != imag.shape or real.ndim != 5 or real.shape[1:] != (2,2,16,144):
        raise ValueError("Unexpected data layout: %s / %s" % (real.shape,imag.shape))
    n=min(args.sample,len(real))
    if args.pca_train+args.pca_test > n:
        raise ValueError("--sample must cover disjoint --pca-train and --pca-test")
    result={"metadata":{"data_path":str(args.data.resolve()),"seed":args.seed,"script":"analysis/audit_data.py","trained_models":False,"numpy_version":np.__version__}}
    result["quality"],energy=inspect_all(real,imag,args.chunk)
    idx=rng.choice(len(real),size=n,replace=False)
    h=complex_chunk(real,imag,idx)
    result["sample_indices"]=idx.tolist()
    hashes=[hashlib.blake2b(x.tobytes(),digest_size=16).hexdigest() for x in h]
    result["sample_duplicate_check"]={"n_checked":n,"duplicates":n-len(set(hashes)),"method":"128-bit BLAKE2b of exact promoted complex64 sample bytes"}
    del real,imag,archive
    progress("frequency/delay/spatial statistics")
    result["structure"],g=structure(h)
    progress("principal beam frequency diagnostics")
    result["beam_frequency"]=beam_frequency_diagnostics(h[:min(1000,n)])
    result["pca"]=pca_diagnostics(g,args.pca_train,args.pca_test)
    del g
    result["uplink_capacity_reference"]=[{"snr_dl_db":d,"snr_ul_db":d-10,"capacity_bits_per_96_complex_uses":float(96*np.log2(1+10**((d-10)/10)))} for d in range(-20,21,5)]
    progress("corrected QAM reference simulation")
    result["reference_simulation"]=reference_simulation(h[:min(args.sim_samples,n)],rng,args.ber_symbols)
    result["metadata"]["elapsed_seconds"]=time.time()-start
    result["metadata"]["plot_created"]=make_plot(result,energy,args.out)
    (args.out/"data_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_report(result,args.out/"data_audit.md")
    progress("DONE %.1f seconds" % (time.time()-start))


if __name__ == "__main__":
    main()
