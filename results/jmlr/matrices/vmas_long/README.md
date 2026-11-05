# Long VMAS runs (n = 16): Section 5.4, Figure 4, Supplement "Longer runs"

Seven teams trained with `experiments/jmlr/core/run_e3.py` and the settings of the main anatomy runs
(`launch_e3.sh`: 64 rollout steps x 128 environments = 8,192 environment steps per PPO iteration,
4,096-observation pools for the SND tables), but with 42-49-minute wall-clock budgets on an otherwise idle
machine (`mamba`: Intel i9-9820X CPU or one RTX 3080; PyTorch 2.5.1, VMAS 1.5.2) instead of 13-minute budgets
on shared GPUs:

| Runs | Command-line flags |
|---|---|
| Dispersion, independent critics (IPPO), seeds 0, 1, 2 | `--scenario dispersion --n-agents 16 --device cpu` |
| Navigation, independent critics (IPPO), seeds 0, 1 | `--scenario navigation --n-agents 16` (seed 0 on cuda:0, seed 1 on cpu) |
| Dispersion, centralized critic (MAPPO-style), seeds 0, 1 | `--critic central --scenario dispersion --n-agents 16` (seed 0 on cuda:0, seed 1 on cpu) |

Common flags: `--iters 5000 --rollout-steps 64 --num-envs 128 --save-at 0,5,10,25,50,75,100,150,200,250,300,400,500,...`
plus `--max-minutes` (42-49). With `--critic central`, agent i's value network receives the concatenated observations
of all n agents; the policies stay decentralized and unshared, and the PPO update is unchanged.

Files: `<task>_n16_s<seed>[_mappo]_<kind>_it<iter>.npz` with keys `D` (16 x 16 SND table), `task`, `iter`, `seed`,
`reward`, `kind` (`trained` snapshots, plus `outlier1` and `twogroup` variants of the final team), and
`<run>_log.json` (reward and wall-clock time at each saved iteration).

Analysis (from the repository root):

```bash
python experiments/jmlr/core/long_runs.py "results/jmlr/matrices/vmas_long/*.npz" \
    results/jmlr/fig_marl_long_data.json results/jmlr/core/long_runs_summary.json
python experiments/jmlr/fig_marl.py results/jmlr/fig_marl_data.json jmlr/figures/fig_marl results/jmlr/fig_marl_long_data.json
```
