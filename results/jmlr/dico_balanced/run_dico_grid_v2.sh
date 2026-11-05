#!/usr/bin/env bash
# DiCo on VMAS Dispersion, n=50, with the paper's audited protocol: 167 PPO
# iterations, 120 envs, 12,000 frames per batch, post-hoc all-pairs SND audit at
# every iteration on 4,096 observations. One job at a time per GPU.
#
# Required env:
#   FORK   path to Graph-SND/ControllingBehavioralDiversity-fork (completed with upstream DiCo)
#   PY     python of the venv (torch 2.8.0+cu128, vmas 1.5.2, benchmarl 1.5.2, torchrl 0.10.1)
#   OUTB   output root; runs land in $OUTB/seed{S}/snd{0pXX}/{balanced|full|bern}/
#   GPUS   space-separated physical GPU ids, e.g. "0 1"
#   CELLS  space-separated seed:snd:estimator triples, e.g. "0:0.12:balanced 1:0.12:full"
# Estimators: balanced = fresh uniformly random 5-regular graph per call (m = 125),
#             bern = Bernoulli p = 0.1 (122.5 pairs in expectation), full = all 1,225 pairs.
set -u
: "${FORK:?}" "${PY:?}" "${OUTB:?}" "${GPUS:?}" "${CELLS:?}"
cd "$FORK" || exit 1
export PYTHONPATH="$FORK:$FORK/.."
mkdir -p "$OUTB"
read -r -a GPU_LIST <<< "$GPUS"
read -r -a CELL_LIST <<< "$CELLS"
run_cell() {
  local gpu=$1 seed=$2 snd=$3 est=$4 tag out estargs
  tag=${snd/./p}; out="$OUTB/seed$seed/snd$tag/$est"
  if [ -s "$out/graph_snd_log.csv" ] && [ "$(wc -l < "$out/graph_snd_log.csv")" -ge 168 ]; then
    echo "$(date -Is) skip complete $seed $snd $est" >> "$OUTB/progress.log"; return 0
  fi
  mkdir -p "$out"
  case "$est" in
    balanced) estargs="model.diversity_estimator=expander model.diversity_expander_d=5" ;;
    bern)     estargs="model.diversity_estimator=graph_p01 model.diversity_p=0.1" ;;
    full)     estargs="model.diversity_estimator=full model.diversity_p=1.0" ;;
    *) echo "unknown estimator $est" >> "$OUTB/progress.log"; return 1 ;;
  esac
  echo "$(date -Is) start gpu=$gpu seed=$seed snd=$snd est=$est" >> "$OUTB/progress.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu "$PY" het_control/run_scripts/run_dispersion_ippo.py \
    seed=$seed task.n_agents=50 task.n_food=50 task.share_reward=true \
    experiment.max_n_iters=167 experiment.on_policy_n_envs_per_worker=120 \
    experiment.on_policy_collected_frames_per_batch=12000 experiment.render=false \
    experiment.train_device=cuda:0 experiment.sampling_device=cuda:0 experiment.buffer_device=cpu \
    "experiment.loggers=[]" experiment.evaluation=false model.desired_snd=$snd $estargs \
    graph_snd_posthoc_full_snd_interval=1 graph_snd_posthoc_full_snd_subsample=4096 \
    hydra.run.dir="$out" > "$out/train.log" 2>&1
  echo "$(date -Is) end gpu=$gpu seed=$seed snd=$snd est=$est rc=$?" >> "$OUTB/progress.log"
}
worker() {
  local gpu=$1 idx=$2 step=${#GPU_LIST[@]} i c seed snd est
  for ((i=idx; i<${#CELL_LIST[@]}; i+=step)); do
    c=${CELL_LIST[$i]}; IFS=: read -r seed snd est <<< "$c"
    run_cell "$gpu" "$seed" "$snd" "$est"
  done
}
for ((g=0; g<${#GPU_LIST[@]}; g++)); do worker "${GPU_LIST[$g]}" "$g" & done
wait
echo "$(date -Is) all done" >> "$OUTB/progress.log"
