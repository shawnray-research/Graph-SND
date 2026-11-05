#!/bin/bash
cd "$(dirname "$0")"   # run_e3.py lives next to this script
PY=${PY:-python}
SAVE="0,10,25,50,100,150,200,250,300,400"
run() { # gpu scenario n seed
  CUDA_VISIBLE_DEVICES=$1 nohup $PY run_e3.py --scenario $2 --n-agents $3 --seed $4 --iters 400 \
    --rollout-steps 64 --num-envs 128 --save-at $SAVE --out matrices_w2 --device cuda:0 --max-minutes 23 \
    > logs/w2_$2_n$3_s$4.log 2>&1 &
}
run 0 navigation 16 10
run 1 navigation 32 10
run 2 dispersion 16 10
run 3 navigation 16 11
sleep 1; ps aux | grep run_e3 | grep -v grep | wc -l
