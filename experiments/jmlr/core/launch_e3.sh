#!/bin/bash
cd "$(dirname "$0")"   # run_e3.py lives next to this script
mkdir -p logs matrices
PY=${PY:-python}
SAVE="0,5,10,25,50,75,100,150,200"
run() { # gpu scenario n seed iters
  CUDA_VISIBLE_DEVICES=$1 nohup $PY run_e3.py --scenario $2 --n-agents $3 --seed $4 --iters $5 \
    --rollout-steps 64 --num-envs 128 --save-at $SAVE --out matrices --device cuda:0 --max-minutes 13 \
    > logs/$2_n$3_s$4.log 2>&1 &
}
run 0 navigation 8 0 200
run 0 navigation 8 1 200
run 0 navigation 16 0 200
run 3 navigation 16 1 200
run 3 navigation 16 2 200
run 3 navigation 32 0 200
run 1 navigation 32 1 200
run 1 dispersion 16 0 200
run 2 dispersion 16 1 200
run 2 navigation 64 0 200
# random-init frozen teams (replicates old exp3 synthetic panel)
for n in 50 100 200 500; do
  CUDA_VISIBLE_DEVICES=0 nohup $PY run_e3.py --scenario navigation --n-agents $n --seed 0 --iters 0 \
    --rollout-steps 16 --num-envs 8 --save-at 0 --out matrices_init --device cuda:0 \
    > logs/init_n$n.log 2>&1 &
done
sleep 2; ps aux | grep run_e3 | grep -v grep | wc -l
