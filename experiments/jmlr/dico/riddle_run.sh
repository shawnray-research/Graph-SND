#!/usr/bin/env bash
# Riddle (2x RTX 4090): install the DiCo stack under /usr1/shawnr (NOT /home, which is
# nearly full), assemble the code, then run the remaining DiCo cells to completion.
set -uo pipefail
ROOT=/usr1/shawnr/gsnd_dico
mkdir -p "$ROOT/tmp" && cd "$ROOT" || exit 1
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/pipcache" XDG_CACHE_HOME="$ROOT/cache" HOME_CACHE="$ROOT/cache"
echo "$(date -Is) setup start"
[ -d venv ] || python3 -m venv venv
venv/bin/python -m pip install -q --upgrade pip
venv/bin/python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || { echo "torch install failed"; exit 1; }
venv/bin/python -m pip install -q vmas==1.5.2 benchmarl==1.5.2 torchrl==0.10.1 tensordict==0.10.0 \
  hydra-core==1.3.8 omegaconf tqdm matplotlib networkx pyyaml || { echo "deps install failed"; exit 1; }
venv/bin/python -c "import torch, vmas, benchmarl; print('stack ok', torch.__version__, torch.cuda.device_count())"
[ -d gsnd ] || git clone -q --depth 1 -b paper-v4 https://github.com/shawnray-research/Graph-SND.git gsnd
[ -d dico ] || git clone -q https://github.com/proroklab/ControllingBehavioralDiversity.git dico
(cd dico && git checkout -q 6406cef)
F=gsnd/ControllingBehavioralDiversity-fork
cp -rn dico/het_control/. "$F/het_control/"
sed 's/config_name="navigation_ippo_config"/config_name="dispersion_ippo_config"/' \
  "$F/het_control/run_scripts/run_navigation_ippo.py" > "$F/het_control/run_scripts/run_dispersion_ippo.py"
sed -e 's/diversity_estimator: knn/diversity_estimator: full/' -e '/diversity_knn_k: 3/d' \
  "$F/het_control/conf/dispersion_ippo_knn_config.yaml" > "$F/het_control/conf/dispersion_ippo_config.yaml"
grep -q minibatch_advantage "$F/het_control/conf/algorithm/ippo.yaml" || \
  printf '\nminibatch_advantage: False\n' >> "$F/het_control/conf/algorithm/ippo.yaml"
grep -q collect_with_grad "$F/het_control/conf/experiment/het_control_experiment.yaml" || cat >> "$F/het_control/conf/experiment/het_control_experiment.yaml" <<'EOF'

buffer_device: cpu
collect_with_grad: False
parallel_collection: False
adam_extra_kwargs: {}
off_policy_use_prioritized_replay_buffer: False
off_policy_prb_alpha: 0.6
off_policy_prb_beta: 0.4
evaluation_static: False
project_name: benchmarl
wandb_extra_kwargs: {}
restore_map_location: null
checkpoint_at_end: False
keep_checkpoints_num: 3
exclude_buffer_from_checkpoint: False
EOF
sed -i 's/^share_rew:/share_reward:/' "$F/het_control/conf/task/vmas/dispersion.yaml"
rm -rf "$ROOT/tmp" "$ROOT/pipcache"
echo "$(date -Is) setup done; launching runs"
FORK="$ROOT/$F" PY="$ROOT/venv/bin/python" OUTB="$ROOT/runs/n50" GPUS="0 1" \
CELLS="1:0.14:balanced 2:0.14:balanced 2:0.15:balanced 0:0.12:full 0:0.14:full 0:0.15:full" \
  bash "$ROOT/gsnd/experiments/jmlr/dico/run_dico_grid.sh" 2>/dev/null || \
FORK="$ROOT/$F" PY="$ROOT/venv/bin/python" OUTB="$ROOT/runs/n50" GPUS="0 1" \
CELLS="1:0.14:balanced 2:0.14:balanced 2:0.15:balanced 0:0.12:full 0:0.14:full 0:0.15:full" \
  bash "$ROOT/run_dico_grid.sh"
echo "$(date -Is) riddle runs finished"
