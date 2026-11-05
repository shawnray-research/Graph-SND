# Balanced-design DiCo runs (Table `tab:dico`, Section 5.8, Appendix C.5)

DiCo diversity control on VMAS Dispersion with n = 50 agents, independent PPO, 167 iterations
(120 environments, 12,000 frames per batch). The SND estimate inside DiCo is the randomized balanced
design: a fresh random 5-regular comparison graph at every call (`networkx.random_regular_graph`,
125 of the 1,225 pairs). A post-hoc all-pairs SND audit on 4,096 held-out observations is logged at
every iteration and never fed back to the controller.

| Folder | Runs |
|---|---|
| `runs/seed{0,1,2}/snd0p12/balanced`, `runs/seed{0,1,2}/snd0p14/balanced` | balanced design, complete (167 iterations) |
| `runs/seed0/snd0p12/full`, `runs/seed0/snd0p14/full` | all-pairs control on the same software stack (stack check), complete |
| `runs/seed{0,1,2}/snd0p15/balanced`, `runs/seed0/snd0p15/full` | still training when the paper was submitted (78-117 and 29 of 167 iterations); excluded from every number |

Each run folder has `graph_snd_log.csv` (one row per iteration: applied SND, post-hoc all-pairs SND,
reward, per-call metric time, iteration time) and the Hydra overrides in `.hydra/`.
`launch_dico_v2.sh` and `run_dico_grid_v2.sh` are the launch scripts (machine `canebrake`, shared Quadro RTX 5000 GPUs,
PyTorch 2.8.0, TorchRL 0.10.1, BenchMARL 1.5.2, VMAS 1.5.2, `experiment.evaluation=false`).
`progress.log` records start and end times.

Reproduce the numbers in the paper (from the repository root):

```bash
python experiments/jmlr/dico/dico_balanced_analysis.py   # writes results/jmlr/dico_balanced/summary.json
```

It compares the balanced runs, cell by cell (seed x set point, final 50 iterations), with the earlier
Bernoulli (p = 0.1) and all-pairs runs in `results/dico_n50_posthoc_full_snd/`.
