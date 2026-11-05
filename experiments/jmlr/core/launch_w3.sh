#!/bin/bash
# Wave 3: three additional VMAS Dispersion seeds (n=16) to test the distinctiveness->relational finding.
cd "$(dirname "$0")"   # run_e3.py lives next to this script
for s in 20 21 22; do
  g=$(( (s-20) % 4 ))
  CUDA_VISIBLE_DEVICES=$g nohup ${PY:-python} run_e3.py --scenario dispersion --n-agents 16 --seed $s --iters 200 \
    --rollout-steps 64 --num-envs 128 --save-at 0,5,10,25,40,50,75,100,150,200 --out matrices_w3 --device cuda:0 \
    --max-minutes 14 > logs/w3_dispersion_n16_s$s.log 2>&1 &
done
