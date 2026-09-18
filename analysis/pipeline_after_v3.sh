#!/bin/zsh
# Post-v3 pipeline: evaluate v3 -> ablate v3 -> train v2 -> evaluate v2 -> ablate v2 -> summarize.
# Run from repo root after v3 training exits. Each stage logs to experiments/pipeline.log.
set -x
PY=.venv/bin/python
V3=experiments/v3_mps_seed20260918
V2=experiments/v2_mps_seed20260918

echo "=== stage 1: v3 formal eval (val 2000x3 batch1) ==="
$PY -m research.evaluate --design $V3/modelDesign.py --weights $V3/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 2000 --repeats 3 --batch-size 1 --device mps \
  --out $V3/eval_val.json

echo "=== stage 2: v3 feedback ablation ==="
$PY analysis/ablation_feedback.py --design $V3/modelDesign.py --weights $V3/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 1000 --repeats 2 --batch-size 16 --device mps \
  --out $V3/ablation_feedback.json

echo "=== stage 3: v2 training (40k steps) ==="
$PY -m research.train --config configs/v2.json --data data_train/H_train.npz \
  --splits splits/random_seed20260918.npz --out $V2 --device mps --run-training

echo "=== stage 4: v2 formal eval (val 2000x3 batch1) ==="
$PY -m research.evaluate --design $V2/modelDesign.py --weights $V2/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 2000 --repeats 3 --batch-size 1 --device mps \
  --out $V2/eval_val.json

echo "=== stage 5: v2 feedback ablation ==="
$PY analysis/ablation_feedback.py --design $V2/modelDesign.py --weights $V2/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 1000 --repeats 2 --batch-size 16 --device mps \
  --out $V2/ablation_feedback.json

echo "=== stage 6: summarize ==="
$PY analysis/summarize_experiments.py

echo "=== PIPELINE DONE ==="
