#!/bin/zsh
# Resume v6 pipeline: m3 formal eval -> ladder fine-tune -> ladder formal eval.
set -x
V3W=experiments/v3_mps_seed20260918/best

echo "=== stage A: m3 formal eval (val 2000x3 batch1) ==="
V6_M=3 .venv/bin/python -m research.evaluate \
  --design models/modelDesign_v6.py --weights experiments/v6_finetune_m3/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 2000 --repeats 3 --batch-size 1 --device mps \
  --out experiments/v6_m3_eval_formal.json

echo "=== stage B: ladder fine-tune (15k steps) ==="
.venv/bin/python analysis/finetune_v6.py --m ladder --steps 15000 --out experiments/v6_finetune_ladder

echo "=== stage C: ladder formal eval (val 2000x3 batch1) ==="
V6_M=ladder .venv/bin/python -m research.evaluate \
  --design models/modelDesign_v6.py --weights experiments/v6_finetune_ladder/best \
  --data data_train/H_train.npz --splits splits/random_seed20260918.npz \
  --split val --samples 2000 --repeats 3 --batch-size 1 --device mps \
  --out experiments/v6_ladder_eval_formal.json

echo "=== V6 RESUME DONE ==="
