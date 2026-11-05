#!/usr/bin/env bash
# Balanced-design (Algorithm 1 style: fresh uniformly random 5-regular graph per call) DiCo grid, n=50,
# plus 3 all-pairs reruns (seed 0) as a software-stack check. Evaluation rollouts disabled (their
# all-pairs SND callback OOMed on shared 16 GB GPUs on 2026-11-02); training metrics and the post-hoc
# all-pairs audit (4,096 observations, every iteration) are unchanged.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FORK=${FORK:-$HOME/ControllingBehavioralDiversity-fork} \
PY=${PY:-python} \
OUTB=${OUTB:-$HOME/dico_n50} GPUS="0 1 2 3" \
CELLS="0:0.12:balanced 1:0.12:balanced 2:0.12:balanced 0:0.12:full 0:0.14:balanced 1:0.14:balanced 2:0.14:balanced 0:0.14:full 0:0.15:balanced 1:0.15:balanced 2:0.15:balanced 0:0.15:full" \
bash "$(dirname "$0")/run_dico_grid_v2.sh"
