# 文献调研：多用户 MIMO 端到端传输 × AI（2026-09-19）

> 调研方式：5 路并行检索（arXiv / IEEE / GitHub / 3GPP / ITU），所有 GitHub 链接经调研 agent 当场验证非 404（github.com 网页直连不稳时走 GitHub API）；标注 ★ 数为当时快照。本机二次抽查：TurboAE 确认 PyTorch 官方 113★。GitHub API 二次复核时遭限流，未复核项以 agent 验证为准。
> 结论先行：反馈压缩（CsiNet 线）文献最成熟但我们已证明不是瓶颈；**与我们赛题同构、且带官方代码的最有价值资源是 NVlabs/neural_rx（MU-MIMO 神经接收机）与学习式短块长码线（TurboAE/ECCT）**。

## 0. 领域定位

本赛题 = "深度学习 × 物理层（AI-native air interface）"，横跨四条成熟研究线 + 一条标准线：
① CSI 反馈压缩（CsiNet 家族）②端到端学习/神经接收机（O'Shea、DeepRx、Sionna）③JSCC/语义通信/速率自适应（DeepJSCC、NTSCC、rateless）④学习式信道编码与 MU 预编码（TurboAE/ECCT、WMMSE unfolding/GNN）⑤3GPP Rel-18/19 AI 空口 + ITU AI/ML in 5G Challenge。

---

## 1. CSI 反馈压缩（f_enc 相关；文献最卷，我们已证明非瓶颈）

### 带开源代码（均已验证）

| 论文 | 出处 | 方法一句话 | 代码 |
|---|---|---|---|
| CsiNet | IEEE WCL 2018 | 开山：CSI 压缩重建 = autoencoder | [sydney222/Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 官方 TF 317★ |
| CRNet | IEEE ICC 2020 | 多尺度卷积 + 分层 latent，多压缩率 | [Kylin9511/CRNet](https://github.com/Kylin9511/CRNet) 官方 PyTorch 94★ |
| CSITransformer | IEEE WOCC 2021 | 早期 Transformer CSI 压缩 | [WilliamYangXu/CSITransformer](https://github.com/WilliamYangXu/CSITransformer) 官方 PyTorch 47★ |
| CLNet | IEEE TWC | 复数域输入 + 空间注意力，轻量 | [SIJIEJI/CLNet](https://github.com/SIJIEJI/CLNet) 官方 PyTorch 79★ |
| DualNet-MP | IEEE WCL 2021 | 幅相双支路，复用上行幅度只传相位 | [max821002/DualNet-MP](https://github.com/max821002/DualNet-MP) 官方 TF 9★ |
| DCRNet | 2022 | 空洞卷积扩感受野 | [tangshunpu/DCRNet](https://github.com/tangshunpu/DCRNet) 官方 PyTorch 31★ |
| **变长反馈 + 量化联合优化**（Nerini et al.） | IEEE TWC 2022 | 按信道条件分配反馈比特数 + 熵编码，率失真联合优化 | [matteonerini/ml-based-csi-feedback](https://github.com/matteonerini/ml-based-csi-feedback) 官方 MATLAB 25★ |
| SALDR | ~2023 | 自注意力 + 密集精炼，多压缩率 | [XS96/SALDR](https://github.com/XS96/SALDR) 官方 TF 7★ |
| TCLNet | arXiv:2601.06588 (2026) | Transformer-CNN 有损 + 语言模型无损熵编码 | [TG-Wireless/TCLNet](https://github.com/TG-Wireless/TCLNet) 官方 PyTorch（新） |

### 无代码但重要
CsiNet+（TWC 2020，必比 baseline）、CsiNet-Pro（TVT 2021）、TransNet（WCL 2022，全注意力）；CsiMamba / MambaCSP（SSM 线，~2025）；条件扩散生成式 CSI 解码（H. Kim 2025）；GPT-2 反馈（Cui, GLOBECOM 2025）/ LLMCsiNet（2026）。追踪入口：[AI4Wireless/LAM4PHY_6G](https://github.com/AI4Wireless/LAM4PHY_6G)（LLM×物理层论文大全 141★）。

### 3GPP TR 38.843（Rel-18）要点
CSI 压缩 = 双端模型；中间 KPI SGCS/NMSE，最终 KPI 吞吐；O1/O1_28B 场景 32TRX 双极化；**跨部署泛化是公认难点**；双端压缩留在 study 阶段，单端 CSI prediction 升级 Rel-19 正式 WI。无统一官方公开数据集，社区事实标准仍是 COST2100。

### 对我们
P0 消融已证明反馈噪声仅 ~1 分 → 此线只取两个点：**Nerini 变长+量化联合优化**（SNR 条件化反馈比特分配，应对上行 -10dB）与 **TransNet/CLNet 的复数域+注意力 encoder**（若做 f_enc 加长训练再考虑）。

---

## 2. 端到端学习 / 神经接收机（我们当前瓶颈所在，最对口）

### 带开源代码（均已验证）

| 论文/资源 | 出处 | 方法一句话 | 代码 |
|---|---|---|---|
| O'Shea & Hoydis | IEEE TCCN 2017 (arXiv:1702.00832) | 开山：收发链路 = autoencoder | 思想并入 Sionna 教程 |
| Dörner et al. | IEEE JSTSP 2018 (arXiv:1707.03384) | 首次真实 SDR 空口训练 autoencoder，交替训练法 | 同上 |
| **Sionna** | NVIDIA 持续维护 | 可微链路级仿真库：5G NR 编码/MIMO/OFDM/E2E 教程 | [NVlabs/sionna](https://github.com/NVlabs/sionna) 官方 ~1610★；**v2.x 已迁移 PyTorch**（agent 查 release 报告） |
| **Neural Receiver for 5G NR MU-MIMO**（Cammerer et al.） | GLOBECOM 2023 (arXiv:2312.02601)；实时版 arXiv:2409.02912 | CNN+GNN 整 slot 处理，输入 Y+LS 信道估计+SNR → LLR，可变用户数/带宽免重训 | [NVlabs/neural_rx](https://github.com/NVlabs/neural_rx) **官方 105★，Sionna 2.x，含 MU-MIMO 与 pilotless E2E notebook** |
| Aït Aoudia & Hoydis（pilotless OFDM） | arXiv:2009.05261 | 联合学习导频叠加+几何调制+神经接收机 | 收录于 NVlabs/neural_rx |
| DeepRx（Honkala et al.） | IEEE TWC 2021 (arXiv:2005.01494) | 全卷积一体化接收机，导频+SNR 条件输入 → LLR | 官方未开源；第三方 [rikluost/ofdm-plutosdr-pytorch](https://github.com/rikluost/ofdm-plutosdr-pytorch) 22★、[j991222/MIMO_JCESD](https://github.com/j991222/MIMO_JCESD) 32★ |
| DeepRx MIMO | IEEE TWC 2022 (arXiv:2010.16283) | MIMO 检测 + 可学习乘性变换 | 同上 |
| EqDeepRx | arXiv:2602.11834 (2026) | 并行 LMMSE/RZF 先验层 + 共享神经块，跨 MIMO 配置免重训 | 未开源（结构可照抄） |
| Machine LLRning | GLOBECOM W 2019 (arXiv:1907.01512) | 学习软解调，跨调制阶数泛化 | 论文教程级实现 |

### 无代码但重要
"Learning to Detect"（Samuel et al., TSP 2019，检测展开法奠基）；Model-Based Deep Learning（Shlezinger et al., arXiv:2012.08405 / 2205.02640，方法论：物理模型进层）；贝叶斯模块化 MIMO 接收机（Raviv et al. 2023）；LOREN（WCNC 2026，**每码率一个 LoRA 适配器覆盖多码率**，arXiv:2602.10770）；FM-Receiver（2026，基础模型式接收机趋势）。

### 对我们
**这是最该抄的三件套**：
1. `NVlabs/neural_rx`：接口与我们接收机完全一致（Y + H + SNR → LLR），GNN 处理 MU 干扰耦合，官方端到端训练管线，Sionna 2.x/PyTorch 可直接改造。
2. EqDeepRx 结构：把可微 LMMSE/等效信道层作为先验 + 神经残差——正好打我们"接收端等效信道"的 ~8 分缺口，且 model-based 设计比黑盒省数据。
3. LOREN 的码率条件化（每码率 LoRA / 条件输入）：匹配我们 5-ctrl 比特选速率、"未传比特按 0.5 计分"的评分设计。

---

## 3. JSCC / 语义通信 / 速率自适应（跨 RE 编码的理论近亲）

### 带开源代码（均已验证）

| 论文 | 出处 | 方法一句话 | 代码 |
|---|---|---|---|
| DeepJSCC-f | IEEE JSAIT 2020 | 反馈驱动的渐进式 JSCC | [kurka/deepJSCC-feedback](https://github.com/kurka/deepJSCC-feedback) 官方 TF 102★（注：DeepJSCC 原文无官方 repo） |
| NECST | ICML 2019 | 离散 VAE 联合信源信道编码 | [ermongroup/necst](https://github.com/ermongroup/necst) 官方 PyTorch |
| Dynamic JSCC | arXiv 2022 | 按信道动态调速率的 JSCC | [mingyuyng/Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC) 官方 PyTorch 152★ |
| **OFDM-guided Deep JSCC** | arXiv 2024 | 直接在 OFDM 资源网格上做 JSCC | [mingyuyng/OFDM-guided-JSCC](https://github.com/mingyuyng/OFDM-guided-JSCC) 官方 PyTorch —— 与"跨 RE 编码"最接近的开源参考 |
| NTSCC | IEEE JSAC 2022 | 超先验熵模型条件编码，SNR 变速率，软解码 | [wsxtyrdd/NTSCC_JSAC22](https://github.com/wsxtyrdd/NTSCC_JSAC22) 官方 PyTorch 109★ |
| NTSCC++ | IEEE JSAC 2024 | 增强版速率自适应 | [wsxtyrdd/NTSCC_plus](https://github.com/wsxtyrdd/NTSCC_plus) 官方 PyTorch 30★ |
| WITT | ICASSP 2023 | ViT 语义传输 | [KeYang8/WITT](https://github.com/KeYang8/WITT) 官方 PyTorch 214★ |
| SwinJSCC | IEEE TCCN 2024 | Swin 主干 + SNR/Rate 双 ModNet 自适应 | [semcomm/SwinJSCC](https://github.com/semcomm/SwinJSCC) 官方 PyTorch 201★ |
| DiffJSCC / CDDM | ICC/JSAC 2024 | 扩散模型辅助 JSCC / 信道去噪 | [mingyuyng/DiffJSCC](https://github.com/mingyuyng/DiffJSCC) 50★、[CDDM](https://github.com/Wireless3C-SJTU/CDDM-channel-denoising-diffusion-model-for-semantic-communication) 82★ |
| 代码总汇 | — | 语义通信开源代码索引 | [yang-hsiao/OpenSemanticComm](https://github.com/yang-hsiao/OpenSemanticComm) 360★ |

### 无代码但重要
DeepJSCC-l（带宽灵活渐进，2021）；DeepJSCC-Q（量化数字版）；**DeepJSCC-MIMO**（Wu, Shao, Gündüz et al., TWC 2024, arXiv:2301.11362，JSCC 直上面向 MIMO+预编码，结构最对口）；Rateless Autoencoder Codes（arXiv:2301.12231）；NTRSCC 广播信道 rateless JSCC（arXiv:2603.21616，机制与本赛题几乎同构）；Spinal codes（SIGCOMM 2012）。

### 对我们
我们的约束是"纯信息比特（无信源）、K 比特铺满固定 144×2 流 RE、接收端出 LLR、ctrl 选速率、未传按 0.5 计分"——更像**短块长好码 + 速率适配**。可借鉴：① OFDM-guided-JSCC 的资源网格映射写法；② NTSCC 的"速率/信道条件 → 条件化编码"结构；③ RaptorQ（[cberner/raptorq](https://github.com/cberner/raptorq)，RFC 6330）的"系统比特 + 增量校验铺满剩余 RE"直接可落地并与神经码叠加。

---

## 4. 学习式信道编码 + MU 预编码

### A 信道编码（带代码，已验证）

| 论文 | 出处 | 方法 | 代码 |
|---|---|---|---|
| **TurboAE** | NeurIPS 2019 | CNN+交织器 Turbo 结构，端到端学习编码/译码，短块长优势区 | [yihanjiang/turboae](https://github.com/yihanjiang/turboae) 官方 **PyTorch** 113★（已二次确认） |
| Sequential RNN Decoder | ICLR 2018 | Viterbi/BCJR 展开为 GRU | [yihanjiang/Sequential-RNN-Decoder](https://github.com/yihanjiang/Sequential-RNN-Decoder) 官方 TF 68★ |
| **ECCT** | NeurIPS 2022 (arXiv:2203.14966) | 校验矩阵掩码 Transformer 软译码，通吃 BCH/极化/LDPC 短码 | [yoniLc/ECCT](https://github.com/yoniLc/ECCT) 官方 PyTorch 61★ |
| LEARN | ICC 2019 | 可变速率自适应神经码 | [yihanjiang/learn](https://github.com/yihanjiang/learn) 官方 3★ |
| 低精度神经极化译码 | IEEE JSAIT 2021 | BP 译码剪枝+量化 | [IgWod/low-precision-nnd](https://github.com/IgWod/low-precision-nnd) 官方 21★ |
| LDPC-BP 复现合集 | — | DNN-BP 经典基线 | [Leo-Chu/Deep-learning-for-LDPC-decoding](https://github.com/Leo-Chu/Deep-learning-for-LDPC-decoding) 第三方 77★ |

无代码：综合征 DNN 译码（Bennatan et al., TCOM）、Neural Polar Decoders 系列（2023-2025）、学习式 IR-HARQ（Göktepe 2023）、有限块长学习码理论（Bernardo 2023）。

### B 预编码（带代码，已验证）

| 论文 | 出处 | 方法 | 代码 |
|---|---|---|---|
| GNN 资源分配 | JSAC 2021 (arXiv:2007.07632) | 干扰图 GNN 输出功率/波束，跨拓扑泛化 | [yshenaw/GNN4Com](https://github.com/yshenaw/GNN4Com) 官方 115★；GNN 通信论文总汇 [jwwthu/GNN-Communication-Networks](https://github.com/jwwthu/GNN-Communication-Networks) 605★ |
| WMMSE deep unfolding | ICASSP 2021 (arXiv:2006.14204) | WMMSE 逐层展开成可训练网络，近最优低复杂度 | [lpkg/WMMSE-deep-unfolding](https://github.com/lpkg/WMMSE-deep-unfolding) 官方 PyTorch 106★ |
| 分布式反馈+MU 预编码（Sohrabi, Yu） | IEEE TWC 2021 (arXiv:2007.06512) | 本地 CSI → 本地预编码的 DL 框架 | [foadsohrabi/DL-DSC-FDD-Massive-MIMO](https://github.com/foadsohrabi/DL-DSC-FDD-Massive-MIMO) 官方 MATLAB 51★ |
| DeepMIMO | v2-v4 | 射线追踪 MIMO 信道数据集框架 | [DeepMIMO](https://github.com/DeepMIMO/DeepMIMO) 官方 |

无代码：Learning to Optimize（Sun, Shi et al., TSP 2018）；低复杂度 MU 预编码 DNN（arXiv:2207.03765）；CsiFBnet（JSAC 2020，反馈→波束增益目标而非重建）。

### 文献综合结论：解析 vs 学习式预编码
完美 CSI+高 SNR 时 RZF/SVD 已近最优，学习式主要赢在**低中 SNR（ZF 噪声放大→MMSE 折中）、CSI 不完美/有噪反馈的鲁棒性、以及与收发端端到端联合训练时**（优化目标直指端到端比特正确率而非代理 MSE）。模型驱动（unfolding/GNN）泛化与样本效率优于纯黑盒。→ 我们当前 ZF 基线下，预编码不是第一杠杆；若动预编码，优先"RZF + 可微 MMSE 折中"而非黑盒。

---

## 5. 综述 / 标准化 / 比赛 / 数据集

### 综述（arXiv 均验证）
- Model-Based Deep Learning（Shlezinger et al., Proc. IEEE 2023, arXiv:2012.08405）—— 设计方法论首选
- Overview of DL-based CSI Feedback in Massive MIMO（东南大学等, IEEE COMST 2022, arXiv:2206.14383）—— 与本赛题方法面重合度最高
- Semantic Communications: Principles and Challenges（Qin Zhijin 等, IEEE Network, arXiv:2201.01389）
- Toward a 6G AI-Native Air Interface（Nokia Bell Labs, arXiv:2012.08285）
- Model-free Training of End-to-end Communication Systems（IEEE JSAC 2019, arXiv:1812.05929，MIMO 端到端训练技巧）

### 标准化
- 3GPP TR 38.843（Rel-18 SI）：三大用例 CSI feedback / beam management / positioning 均有增益；Rel-19 AI/ML for NG-RAN 转规范性（LCM），空口用例继续 study；AI-native 空口落向 Rel-20/6G（首版规范约 2028-29）。
- ITU-R M.2160-0（2023）：IMT-2030 六场景含 "AI and Communication"；IMT-2030(6G) 推进组 2025-08 成立 6G AI 特设组——本赛事的政策背景。

### 同类比赛
- **ITU AI/ML in 5G Challenge（2020-2024）**：获奖方案按规则必须开源 → [github.com/ITU-AI-ML-in-5G-Challenge](https://github.com/ITU-AI-ML-in-5G-Challenge)，全历届存档 [Challenge_Archive](https://github.com/ITU-AI-ML-in-5G-Challenge/Challenge_Archive)；获奖示例 [Channel-Estimation-ML-DOJO-2](https://github.com/ITU-AI-ML-in-5G-Challenge/Channel-Estimation-ML-DOJO-2)、基线 [lasseufpa/ITU-Challenge-ML5G-PHY](https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY)。
- 本系列赛（OPPO/IMT-2030，DataFountain 平台）：2022-23 首届（信道建模/波束预测）、2024 届（非正交导频+数据叠加接收机）、2025 届（控制语义系统）、2026 本届。**未检索到往届完整开源冠军代码**，方案散见 DataFountain 论坛/CSDN。

### 公开数据集/仿真器
- [Sionna](https://github.com/NVlabs/sionna)（可微 E2E + 3GPP TR 38.901 信道，v2.x PyTorch）
- [DeepMIMO](https://github.com/DeepMIMO/DeepMIMO) + [deepmimo.net](https://deepmimo.net/)（CsiNet 系 O1 场景出处）
- [DeepSense 6G](https://www.deepsense6g.net/)（实测多模态，波束预测基准）
- [QuaDRiGa](https://github.com/FraunhoferHHI/QuaDRiGa)（Fraunhofer，3GPP 校准级信道生成）
- AerialIMT：未能验证公开存在（可能为赛方内部名），以赛题页说明为准。

---

## 6. 落地优先级（结合我们现状：v3 线上 62.38，缺口在接收端等效信道 + 速率自适应）

1. **抄 `NVlabs/neural_rx` 的接收机骨干与训练技巧**（接口同构，MU-MIMO，官方 PyTorch）——对应"接收端等效信道"缺口。
2. **跨 RE 编码升级**：TurboAE/ECCT 的学习式短码 + RaptorQ 式"系统比特+增量校验"，把 v3 的全长度跨 RE 编码换成多速率版本（ctrl 选码率，LOREN 式条件化）。
3. **反馈侧只在需要时取**：Nerini 变长反馈（SNR 条件比特分配）——P0 已证明反馈仅 ~10 分且压缩近饱和，投入有限。
4. 预编码保持 ZF/RZF 基线，仅在与接收机联合训练时考虑可微 MMSE 折中（WMMSE unfolding 思路）。
5. 自建仿真/增广：Sionna 2.x + DeepMIMO/QuaDRiGa，做 B 榜鲁棒性预研与数据增广。
