# v8（导频锚定合体版）收尾说明

v8 未产生正式评测结果：两臂对照实验（子带恒定 W vs 逐 RE W）在 10k 步曲线重合
（55.28 vs 55.34），deficit 来自 12 个导频 RE 占用的 96 个孤儿源位（BER 49.8%=全损）
与 RX 新物理特征的重学成本，在 40-60k 步预算内无法回收。三家 Agent 的 ~65 分建立
在 15 万-38 万步链式训练之上。

证据存档：
- reports/results/v8_falsified/（逐位诊断 + 逐RE W 两臂曲线）
- reports/results/v8_subband_run/（子带 W 曲线）
- reports/results/v7_partial/（副本物理复现部分曲线）

当前主攻：v3 长训 100k 步（experiments/v3_long_mps，热启动 + EMA 双轨）。
experiments/v8_mps_seed20260919/ 下的 eval_val*.json 为墓碑文件（让过期巡检自终止）。
