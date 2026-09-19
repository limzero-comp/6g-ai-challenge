# 43 个参考代码库深读笔记（2026-09-19）

> 方法：5 路并行 agent 实际阅读 `third_party/` 下全部 43 个库的模型定义与训练代码（非仅 README），按"对赛题可借鉴点"提取，标 P0（立刻试）/ P1（值得做）/ P2（参考）。背景：v3 线上 62.38，缺口主项 = 接收端等效信道利用（~8 分）与速率自适应；P0 消融已证反馈非瓶颈。文献背景见 [literature_survey_2026-09.md](literature_survey_2026-09.md)。

## TL;DR — 按我们缺口排序的 P0 行动清单

| # | 来源 | 动作 | 打哪个缺口 |
|---|---|---|---|
| 1 | neural_rx | f_rx 加**逐样本功率归一化**（y 缩放同尺度 no），几行改动 | 接收端（跨 SNR 泛化） |
| 2 | neural_rx | f_rx 加**辅助信道读出**（ReadoutChEst：Linear→重建 h 或 ŷ，loss + 0.02×MSE，multiloss 逐迭代） | 接收端等效信道（主缺口，文献证明加速收敛+提效） |
| 3 | neural_rx | **变 MCS masking 读出**：readout 输出 max_bits，按 b_ctrl gather 子集，BCE 掩掉未传比特 | 速率自适应 + 0.5 计分对齐 |
| 4 | neural_rx | **迭代式状态更新**（StateInit → 4-8 次带残差的 sep-conv 更新）替换/增强现有 Transformer 结构；sepconv2d 在我们 1×144 场景退化为 depthwise Conv1d(k=3)+1×1 | 接收端 |
| 5 | sionna | 移植 `lmmse_equalizer`（含 no_eff 有效噪声方差）作为解析对照/蒸馏教师/预处理分支 | 接收端等效信道 |
| 6 | NTSCC 系 | **速率档条件化 = 每档独立适配头 + rate token embedding**（离散档索引选择，而非连续标量）；32 档恰好 5 ctrl bit | 速率自适应 |
| 7 | SwinJSCC/NTSCC_plus | **SNR → ModNet/FiLM 门控**（标量 SNR → MLP → sigmoid 逐通道门，条件分支 detach），被三库复用的最成熟 SNR 条件化 | 全模块 SNR 泛化 |
| 8 | TurboAE | v3 跨 RE 编码改 **TurboAE-CNN 骨架**：系统支+两奇偶支+固定置换 gather 交织器（零参数可微），多码率=通道数；f_rx 侧加 3-6 半迭代、外信息相减的涡轮头 | 跨 RE 编码增益 |
| 9 | ECCT | f_rx loss 换 **`y·sign(x)` 缩放软目标 + 每 sample 随机 SNR**（几乎零成本，低 SNR 稳定） | 训练配方 |
| 10 | WMMSE-unfolding | 预编码升级：RZF 初始化 + 2-4 层闭式 u/w + learned-step PGD（无矩阵求逆），αᵢ 按用户 SNR 加权天然优化 p10 | 预编码/MU 干扰 |
| 11 | DualNet-MP | f_enc：**软量化**（阶梯 sigmoid 可微）+ 幅相分离非对称预算；训练注入逐维对齐的多档 SNR 上行噪声 | 反馈鲁棒（低成本补充） |

---

## 1. e2e_receiver（最对口：f_rx 缺口）

### neural_rx（NVIDIA 官方，TF2+Keras，权重不可直接加载，只搬架构与配方）
- **架构**（`utils/neural_rx.py`）：y[·,sc,sym,2Nrx] → 逐样本功率归一化（:816-824）→ StateInit（SeparableConv2D 3×3 ×[128,128]，输入 concat[y, pe, h_hat]）（:22-145）→ 8 次迭代（各迭代独立权重，共享会掉点）AggregateUserStates（Dense(64→d_s) 后**对其它活跃用户状态取均值** = MU 消息传递，active_tx 掩码，任意用户数/带宽免重训）（:147-251）+ UpdateState（sepconv[128,128]+残差）（:253-369）→ ReadoutLLR（Dense(128)→bits/RE）。位置编码 = 每 RE 到最近导频的时/频距（:1110-1160）。
- **变 MCS 两方案**（:831-872）：每 MCS 独立 IO 层 + one-hot mask 选择；或 `var_mcs_masking` 单读出层出 max_bits 按 MCS gather——**速率自适应直接模板**。变用户数：TX 端非活跃用户能量乘 0，RX 端掩码（`e2e_model.py:349-351`）。
- **训练**（`config/nrx_large_var_mcs_64qam_masking.cfg`）：Adam **1e-3**、10M 步两段、batch 128、XLA；SNR 逐样本 U[min,max] 且 **MCS 条件 SNR 偏移**（QPSK/16QAM/64QAM +0/4/7dB）；用户数三角分布采样；loss = 逐迭代 BCE + **0.02/0.01×MSE(h_hat)**（double_readout 辅助信道读出显著加速收敛）；每 1000 步存档；`weights/` 有 17 组预训练权重（TF pickle）。
- **移植清单**（对接我们 f_rx = [y(4)+h(64)+log10 no(1)]=69 维 → Transformer×6）：①H 已知 → 走 initial_chest 路径，pe 可省；②sepconv2d → depthwise Conv1d(k=3)+1×1（我们无时间维）；③先省略用户聚合层（单用户配置 `e2e_large.cfg` 证明可行），16 天线已在 h 特征里；④readout 出 max_bits/RE，按 b_ctrl gather，BCE 掩未传比特；⑤依赖全标准 torch 算子（~200 行），torch.compile 替代 XLA。

### MIMO_JCESD（PyTorch，研究级，数据需 MATLAB 预生成）
- 18 通道输入 [Y(8)+H_ls(8)+X0(2)]（`deeprx/model/mmsenet.py:1894-1902`）；deeprx 版 = 11 残差块、分组+膨胀 3×3 conv（膨胀 (1,1)→(2,3)→(3,6)、通道 64→256→64）；hypernet 版 = 超网络生成维纳滤波参数。训练无亮点（Adam 1e-4，StepLR）。
- **P1** 分组膨胀卷积骨干（分辨率无关，可与 Transformer 混合）；**P2** 由均衡符号 G(1-G) 构造近似 LLR 作蒸馏教师。

### ofdm-plutosdr-pytorch（小而干净，PyTorch，可跑）
- 输入仅 Y 的 I/Q 2 通道（H 不进网络），5 个预激活残差块（LayerNorm+ReLU+3×3+skip）（`models_local.py:26-68`）。
- **P0** 逐样本 loss 按 **SINR 归一加权**（cell 16）——容量分给高 SNR 样本，吻合效率+公平双目标；**P1** 对数 LR 衰减（1e-3→2e-5）、仿真+实测混训思想。

### sionna 2.1.0（**src/sionna/phy 已是纯 PyTorch**，算子可逐行抄）
- `mimo/equalization.py::lmmse/zf/mf_equalizer`（返回 **no_eff 有效噪声方差**，含 whiten_interference 稳定形式）；`mimo/detection.py::KBest/EP/MMSEPIC`；`ofdm/channel_estimation.py`（LS+插值）。
- **P0** lmmse_equalizer+no_eff 作等效信道预处理/蒸馏教师；教程 `Neural_Receiver.ipynb`（5×128ch 残差块，输入 [re,im,log10 no]×天线）与我们输入组装几乎一致可对齐。

## 2. coding_precoding（f_tx 编码 + 预编码）

### TurboAE（NeurIPS'19 官方，PyTorch，可直接跑）
- 编码器 = 三支 SameShapeConv1d（k=5×2层+ELU+Linear→1）：系统支 / 奇偶支 / **交织后奇偶支**（`encoders.py:306-377`），拼接后 power constraint（全块 mean/std 归一，`encoders.py:102-125`）；**交织器 = 纯 gather(p_array) 零参数可微**（`interleavers.py:6-21`）；**多码率 = code_rate_k 作输入通道数**。
- 译码器 DEC_LargeCNN：6 半迭代，每半迭代输入 [r_sys, r_par, prior] → 5 层 CNN → **外信息相减** x_plr−prior（`decoders.py:157-269`）。
- **FTAE 含噪反馈版**（`ftae_ae.py:295-376`）：学习式 fb_enc + fb_z 加噪回传，编码器吃 [bits, r1, x1, r2, x2] 增量冗余——与我们有噪上行结构同源。
- **训练**：enc:dec **交替 1:5 epoch 且用不同 SNR 区间**（enc 1dB / dec −1.5~2dB，σ 域线性混合 `channels.py:22-25`）；loss 只用 BCE（作者：maxBCE/focal/soft_ber 均试败）；Adam 1e-3（+Lookahead）。

### ECCT（NeurIPS'22 官方，PyTorch，173 行）
- 校验矩阵→注意力掩码（同校验式内变量/校验节点两两可注意，`Model.py:142-166`）；输入 [|y|, syndrome] 逐 token；d=32、8 头、6 层。
- **doped noise 技巧**：训练只送全零码字加噪（AWGN 对称性下等价，`Main.py:31-57`）；loss = 对 **y·sign(x) 缩放软目标**的 BCE（`Model.py:136-140`）——P0 直接抄进 f_rx。

### WMMSE deep unfolding（TF1 notebook，Pellaco et al.）
- V 更新改 PGD（∇=−2αᵢwᵢuᵢhᵢ+2Avᵢ，A=Σαᵢwᵢ|uᵢ|²hᵢhᵢᴴ）**无求逆/特征分解**，复数用 2×2 实矩阵；每层 = 闭式 u + 闭式 w + K=4 个 PGD step，**可学习参数 = step_size 与 Nesterov momentum**；功率约束用范数投影；**损失 = 每层 WSR 求和（深监督）**；αᵢ 可调即泛化 RZF。
- **P0** 以我们逐 RE RZF 为初始化加 2-4 层展开；αᵢ 按用户 SNR/速率加权 → 天然优化 p10 公平项。

### GNN4Com / Globecom2019（IGCNet）
- 节点特征=[本链路增益,1]，边特征=[H_ij,H_ji,1]；EdgeConv=MLP([src,dst,edge])+max 聚合；**无监督和速率 loss**（SINR 显式计算）；IGCNet 5 轮"强度作为特征回灌"与 turbo 迭代同构；UE-AP 二部异构图可 train K≠test K。
- **P1** 2 用户×144 RE 建图（用户=节点、RE=边特征），EdgeConv 出每 RE 功率比，替代现功率比 MLP；DGL 依赖用 torch scatter 重写。

### 其它
- `learn`：空库（代码从未发布，README 自述 TurboAE 才是正解）。速率自适应落到 turboae 的 code_rate_k/打孔。
- `DL-DSC-FDD-Massive-MIMO`：**本地 clone 为空**（上游 master 无提交），按论文仅 P2 观点。
- `low-precision-nnd`（int8 量化译码）、`Deep-learning-for-LDPC-decoding`（BP 边 lifting 成稠密矩阵向量化，P1 避免循环）、`Sequential-RNN-Decoder`（TF1 前身）：P2。

## 3. jscc（速率条件化的成熟机制）

### 结论 (a)：b_ctrl 速率条件化的最成熟做法
**离散档位 = 每档索引的独立适配头（indexed weights）+ 档位 embedding（rate token）**，而非连续标量——速率是结构参数（哪些 RE 有符号），连续条件化有档间混淆风险。SNR（连续量）才用 FiLM/ModNet。推荐组合 = **NTSCC_plus 的整体结构**（`compatible_ntscc.py`）：b_ctrl 选档 → 索引选头 + rate token；SNR → ModNet 门控。

### 结论 (b)：动态速率决策依据
文献最有效：①内容复杂度/熵（NTSCC 逐位置 −log2 p）；②瞬时 SNR（Dynamic_JSCC 策略网络 + λ·E[发送量] 惩罚端到端学出"信道差就少发"）。对应到我们：f_tx 的"反馈质量"代理（上行 SNR、|U| 统计）扮演解码置信度角色，值得作为策略头输入；训练中把 0.7×eff+0.3×p10 直接做成奖励/权重。

### 各库要点
- **NTSCC_JSAC22**（PyTorch+compressai）：熵驱动逐 patch 速率——`symbol_num=Σ(−log2 p)·η` → searchsorted 到 16 档（`layer/jscc_encoder.py:90-95`）；每档独立线性适配器+掩码截断（:9-41）；rate_token 加到特征（:96-98）；速率索引走容量可达编码单独传（开销 log2(档数)/像素）。loss=[mse_ntc+λ·bpp]+mse_ntscc，λ=64。
- **NTSCC_plus**：`forward_multiple_snr` 逐样本 SNR 张量过信道（`channel/channel.py:43-66`）；ChannelModNet+AdaptiveModulator（SNR→MLP→sigmoid 逐通道门控，`net/compatible_ntscc.py:11-55`）；"速率硬选档 + SNR 软门控共存同一网络" = 架构模板。
- **SwinJSCC**：AdaptiveModulator（`net/encoder.py:192-205`）：SNR→MLP→sigmoid→M 维门，级联 7 段乘性注入，**条件分支对主特征 detach**；Rate ModNet 按门控强度 top-k 选特征维（速率=特征选择）。SNR/速率逐 batch 随机抽（`network.py:54-63`）。
- **Dynamic_JSCC**：**唯一给出"学发多少"可微训练配方的库**——[池化 latent+SNR]→MLP→Gumbel-Softmax（温度退火 t=exp(−0.015)/epoch，下限 0.005）+ straight-through 硬掩码（`models/networks.py:315-332`）；**温度计编码（累积和）= 嵌套前缀速率档**（:335-341），低速率是高速率前缀——与跨 RE 铺满 + 0.5 计分天然契合；G_n=4 组恒发兼作隐式信令。
- **OFDM-guided-JSCC**：网格只是发送排布，卷积在 latent 空间做；CE/EQ = **"LMMSE 基线 + 学习残差"**（`JSCCOFDM_model.py:151-186`）+ 信道估计误差显式损失；**SNR_cal='ins' 噪声按实际收功率逐样本算** = 隐式 SNR 抖动增广（P1）。
- **deepJSCC-feedback**（TF）：多轮渐进残差传输 + Combiner 融合，逐层训练后冻结；嵌套比特排布思想用于跨 RE 编码（P1）。
- `necst`/`WITT`/`DiffJSCC`/`CDDM`：P2（扩散类推理重，不适合 LLR 任务；CDDM 思想=接收端可学习去噪前端）。`OpenSemanticComm` 为索引。

## 4. csi_feedback（f_enc 低成本改进；P0 吝啬，反馈非瓶颈）

- **库共性问题**：全部 CsiNet 系（2×32×32 稀疏图进、MSE 出），**无一实现 SNR 条件化、无一建模上行噪声**（例外见下）。
- **DualNet-MP**（TF，最贴我们场景）：幅相分离非对称压缩（幅度 CR=4/相位 CR=8，`DualNet_MP.py:39-40`）；解码器拼接 UL 观测作边信息（:138-141）；**软量化进训练**（阶梯 sigmoid(25·(x−level)) + 可学习逐维缩放，:174-201）；两阶段训练。
- **ml-based-csi-feedback**（MATLAB，PCA+kmeans，非 NN）：**逐维 SNR 对齐的含噪训练**（`main.m:27-35`，每维方差 1/(Λ·SNR) 再 sqrt(Λ) 归一）——P0 直接搬进 f_enc 训练循环；贪心逐维比特分配（`func_allocate_bits.m`）；PCA 基 DFT 域稀疏化+Gram-Schmidt（与我们固定 DFT 基同源）。
- **CRNet**：非对称长核（1×9/9×1 沿时延轴+窄核天线轴）双路径 CRBlock——匹配"前 4 抽头 95% 能量"；每档 CR 单独训（非统一训练）；warmup30+cosine 至 5e-5。
- **CLNet**："复数输入"实为 real/imag 2 通道（无幅相耦合约束，结论性知识）；SpatialGate 约 20 参数几乎免费；encoder/decoder 分存部署与我们一致。
- **CSITransformer**：逐行 token+单层注意力，**位置编码注释掉实测更好**；NMSELoss（直接优化 NMSE）。
- `DCRNet`（膨胀卷积）/`SALDR`（嵌套多 CR+加权损失）/`Python_CsiNet`（历史基线）/`TCLNet`（README-only，代码未放出）/:`LAM4PHY_6G`（纯索引）：P2。
- 修复记录：CRNet/CLNet/CSITransformer/DCRNet 克隆时工作区文件缺失，已 `git checkout -- .` 恢复。

## 5. datasets_sim + itu_challenge（B 榜增广 + 工程技巧）

- **DeepMIMO Python v4**（Apache-2.0）：B 榜增广**首选**。三步生成：`dm.download('asu_campus_3p5')` → `dm.load` → `dataset.compute_channels()`（`docs/quickstart.md:25-33`）；`ChannelParameters(bs_antenna={'shape':[16,1]}, ue_antenna={'shape':[1,1]}, ofdm={'subcarriers':144,...})` 原生出 [n_ue,1,16,144]，两 UE 组 MU。场景 MB 级下载；双极化仅 MATLAB 版。注意：簇状射线信道与赛题瑞利型有 gap，需逐样本功率归一 + 官方数据混训（建议增广预训练→官方微调）。
- **QuaDRiGa v2.8.1**（MATLAB/Octave，非商用）：3GPP 38.901 校准参数内置免下载、原生双极化，`@qd_channel/fr.m` 直接出 [Rx,Tx,Carrier,Snapshot]——统计上更"泛"，代价是工具链。
- **DeepSense6G**：仓库已 reset 成占位（11 字节 README），**放弃**。
- **Channel-Estimation-ML-DOJO-2**（2020 NCSU 获奖方案，**无 NN**，稀疏 OMP）：可立即迁移的技巧——①噪声白化（Cholesky 预处理，`Whitening.m`）；②超采样 DFT 角度网格 + AoA/AoD/ToF 乒乓交替精炼；③**瑞利最大值分位置信门限** `median(|res|)·sqrt(−2log(1−0.98^(1/N)))` 替代硬阈值；④路径参数化 (AoA/AoD/ToF/α) 可作为 f_enc 压缩目标或接收端 model-based 等效信道前端（直击 8 分缺口）。
- **ITU-Challenge-ML5G-PHY**（Oulu 基线）：**等效信道 |W^H·H·F| 码本扫描作标签/中间表示**的组织方式——可仿此在训练管线加 (f,W) 扫描，把 H_eff/速率作辅助监督（与速率自适应缺口同构）。
- **Challenge_Archive**：2024 "Optimal Multi-user scheduling"（64 用户×52 子载波）方案在 `3DML-Wireless/SMART-Scheduler`（MU 调度+Jain 公平性），值得单独 clone。

---

## 附：与我们系统的对接注意

1. **框架迁移**：neural_rx/WMMSE-unfolding/deepJSCC-feedback/necst 等为 TF2/TF1，只搬架构与配方；sionna 2.x、TurboAE、ECCT、NTSCC 系、SwinJSCC 为 PyTorch 可直接改。TF pickle 权重一律不可直接加载。
2. **维度适配**：我们 f_rx 是 1×144（无时间维）→ 所有 2D sepconv 退化为 depthwise Conv1d + 1×1；H 已知（无导频问题）→ 走 initial_chest/h_hat 路径。
3. **评分对齐**：速率自适应相关改造必须保持"BCE 只对实传比特、未传比特不输出噪声 LLR"，对齐 0.5 计分；公平 p10 通过 αᵢ/采样加权进 loss（neural_rx 的 MCS 条件 SNR 偏移 + ofdm-plutosdr 的 SINR 加权是现成配方）。
4. **推理预算**：所有推荐结构参数量 <1e6，1000s 限制无压力；neural_rx 8 迭代 132PRB 双用户 ≈3.1ms/A100 可作量级参照。
