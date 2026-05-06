# Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning

This repository accompanies the paper **Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent Reinforcement Learning** by Shawn Ray
(Carnegie Mellon University). It provides the Graph-SND core package, unit tests,
training and experiment scripts, committed result artifacts used by the
paper, and a focused fork of the DiCo code path for the closed-loop
diversity-control experiments. The full paper source lives in `Paper/`.

- Project repository: <https://github.com/shawnray-research/Graph-SND>
- Author: Shawn Ray, Carnegie Mellon University, <shawnray@cmu.edu>

Graph-SND is a sparse aggregation layer for System Neural Diversity
(SND). Instead of averaging behavioral distances over all
$\binom{n}{2}$ agent pairs, it averages over the edges of a graph
$G$. The same implementation supports:

- complete-graph recovery of full SND;
- fixed-graph localized diversity measurement;
- Bernoulli and fixed-size random edge sampling for scalable SND
  estimation;
- random-regular expander aggregation for deterministic sparse
  approximations.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test,plot]'
python -m pytest tests -q
```

The core tests cover graph generators, SND/Graph-SND estimators,
Wasserstein identities, concentration radii, and the batched policy code
used for the $n=100$ PPO scaling run.

## Core Package

- `graphsnd/metrics.py`: full SND, Graph-SND, Horvitz-Thompson
  estimation, finite-population sample means, and concentration radii.
- `graphsnd/graphs.py`: complete graphs, Bernoulli samples, fixed-size
  samples, $k$-nearest-neighbor graphs, random regular graphs, and
  spectral diagnostics.
- `graphsnd/wasserstein.py` and `graphsnd/tvd.py`: behavioral distances
  used in the Gaussian/Wasserstein and categorical/TVD experiments.
- `graphsnd/batched_policies.py`: stacked independent per-agent policies
  used to make the $n=100$ PPO run practical without changing the policy
  factorization.

## Reproducing Paper Evidence

The committed CSV/JSON summaries and plotting scripts let readers
inspect the exact numeric evidence without rerunning all GPU
experiments. The commands below regenerate the main lightweight figures
and tables from CSVs included in `results/`.

```bash
# Core recovery, unbiasedness, concentration, and small-n timing plots.
python experiments/exp1_plots.py --results results/exp1

# GPU timing figure from committed timing CSV.
python experiments/plot_timing_n500.py \
  --exp2-csv results/exp2/timing_n100_250_500.csv \
  --out results/exp2/timing_n500.pdf

# Expander sparsification figures from committed CSVs.
python experiments/exp3_plots.py \
  --csv results/exp3/expander_distortion.csv

# DiCo n=50 Bernoulli-vs-full summary table and figure from included
# per-iteration CSV logs.
python experiments/n50_bern_vs_full_comparison.py \
  --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_setpoint_sweep \
  --out-dir results/dico_n50_bern_vs_full

# DiCo n=50 post-hoc complete-graph SND audit.
python experiments/n50_posthoc_full_snd_validation.py \
  --root ControllingBehavioralDiversity-fork/results/neurips_final_n50_posthoc_full_snd \
  --out-dir results/dico_n50_posthoc_full_snd
```

Several experiments require a CUDA GPU to rerun from scratch. The
committed summaries are included so the numerical claims remain
inspectable on CPU-only machines.

## Claim-to-Code Map

| Paper claim | Code / artifact |
| :--- | :--- |
| Complete graph recovers SND exactly | `tests/test_metrics.py`, `experiments/exp1_metric_comparison.py`, `results/exp1/recovery.csv` |
| Horvitz-Thompson estimator is unbiased | `graphsnd/metrics.py::ht_estimator`, `tests/test_metrics.py`, `results/exp1/unbiasedness.csv` |
| Uniform samples concentrate with $O(1/\sqrt{m})$ radius | `graphsnd/metrics.py::hoeffding_bound`, `tests/test_metrics.py`, `results/exp1/concentration.csv` |
| Runtime scales with $\lvert E \rvert$ rather than $\binom{n}{2}$ | `graphsnd/metrics.py::graph_snd_from_rollouts`, `experiments/exp2_timing_scaling.py`, `results/exp2/timing_n100_250_500.csv` |
| Random regular graphs give accurate sparse aggregators | `graphsnd/graphs.py::random_regular_edges`, `experiments/exp3_expander_distortion.py`, `results/exp3/expander_distortion.csv` |
| Graph-SND can replace full SND in DiCo | `ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`, selected DiCo integration files, `results/dico_n50_bern_vs_full/`, `results/dico_n50_posthoc_full_snd/` |

## DiCo Integration

`ControllingBehavioralDiversity-fork/` contains a focused fork of the
DiCo code path used for the closed-loop diversity-control experiments.
Only the Graph-SND integration files needed to inspect or reapply the
modification are included. The upstream DiCo notice and license are
preserved in that bundle. For the exact integration summary, see
`ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md`.

## Paper

The full paper source is in `Paper/`:

- `Paper/main.tex` -- main document (compiles in `[preprint]` mode).
- `Paper/references.bib` -- bibliography.
- `Paper/neurips_2026.sty` -- NeurIPS 2026 style file.
- `Paper/figures/` -- all figure PDFs (Type-1/TrueType fonts).
- `Paper/main.pdf` -- compiled PDF (when present).

To rebuild from source:

```bash
cd Paper
tectonic main.tex   # or: latexmk -pdf main.tex
```

## Citation

If you use Graph-SND in your work, please cite the paper. See
[`CITATION.cff`](CITATION.cff) for machine-readable metadata. A BibTeX
entry will be provided once the camera-ready / arXiv version is posted.

## License

MIT, see [`LICENSE`](LICENSE). The DiCo fork retains its upstream
license; see
[`ControllingBehavioralDiversity-fork/LICENSE`](ControllingBehavioralDiversity-fork/LICENSE).

## Contact

Shawn Ray, Carnegie Mellon University, <shawnray@cmu.edu>.
