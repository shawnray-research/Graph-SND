# Measuring Pairwise Diversity with Linearly Many Comparisons

This repository accompanies **Measuring Pairwise Diversity with Linearly Many
Comparisons** by Shawn Ray
(Carnegie Mellon University). The work extends and supersedes the preprint
*Graph-SND: Sparse Aggregation for Behavioral Diversity in Multi-Agent
Reinforcement Learning* (arXiv:2605.05020).

- Project repository: <https://github.com/shawnray-research/Graph-SND>
- Author: Shawn Ray, Carnegie Mellon University, <shawnray@cmu.edu>

Diversity is usually measured as the **average pairwise dissimilarity** of a
population, for example:

- System Neural Diversity of a multi-agent team;
- ensemble disagreement;
- Self-BLEU or mean pairwise LPIPS of generated samples;
- nucleotide diversity.

Exact evaluation costs $\binom{n}{2}$ comparisons. This package provides:

- **`anatomy(D)`**: the canonical split of any dissimilarity table into a
  *level* (the average itself), per-individual *distinctiveness* $a_i$, and
  *relational structure* $R$ (roles, clusters), with their variance shares.
- **`estimate_mean_dissimilarity(d, n, k)`**: a randomized *balanced design*
  that compares every individual exactly $k$ times ($nk/2$ comparisons). It is
  exactly unbiased, distinctiveness cancels from its error, and its variance is
  known in closed form. For every metric, $O(n+\varepsilon^{-2})$ comparisons give
  relative error $\varepsilon$ (`median_of_designs`), which is optimal; the
  universal bound uses the tight inequality $\sigma^2_{\mathrm{int}}\le\frac{n-2}{n}\mu^2$.
- **`separable_mean(X)`**: exact linear-time evaluation when the dissimilarity
  is squared-Euclidean-type or $\ell_1$ (e.g. total variation), where no sampling
  is needed.

```python
import numpy as np
from graphsnd.designs import anatomy, estimate_mean_dissimilarity

# d(i, j): any (possibly expensive) dissimilarity between members i and j
est, pairs, vals = estimate_mean_dissimilarity(d, n, k=2, rng=0)  # n*k/2 calls

an = anatomy(D)   # full table available? an.mu, an.a, an.R, an.relational_share
```

The design API needs only NumPy (`tests/test_designs.py` checks the exact
laws, including exhaustive enumeration of all relabelings at small $n$).

## Paper source and reproduction

The manuscript (JMLR format) lives in `jmlr/`: `main.tex` (the article) and
`supplement.tex` (the supplementary material) with their inputs, `refs.bib`, `figures/`,
the compiled `main.pdf` and `supplement.pdf`, and `cover_letter.md`. The two documents
cross-reference each other (`xr-hyper`), so build them from inside `jmlr/` with
`sh build.sh` (tectonic; it compiles main, supplement, then main again).

Experiment code is in `experiments/jmlr/`, and the resulting tables, logs and every recorded
behavioral-distance matrix are in `results/jmlr/`:

- `verify_theory.py`: numerical checks of every closed-form identity in the paper.
- `timing_scale.py`, `accuracy_scale.py`, `fig_scale.py`: GPU cost of all-pairs vs balanced SND
  up to n = 10,000, and measured vs closed-form accuracy up to n = 5,000.
- `closed_loop.py`, `fig_control.py`: simulated diversity control (10 populations per setting).
- `anatomy_real.py`, `fig_marl.py`, `fig_gallery.py`, `fig1_teaser.py`: anatomy figures.
- `samples/`: generated text scored by neural judges (a cross-encoder and Claude Opus 5.5), the
  temperature curve, Fashion-MNIST sample sets and the order-2 Vendi score (Section 5.6); outputs in
  `results/jmlr/samples/`.
- `null_shares.py`: distinctiveness shares of unstructured Gaussian populations (Section 5.4).
- `core/`: law validation, design comparison, VMAS training and matrix export, model
  populations (n = 128 and 1,024), confidence intervals and the fixed-design bound check
  (see `core/README.md`).

The original Graph-SND tooling (graph-edge aggregation, Bernoulli/expander
estimators, VMAS training and DiCo integration) is kept below.

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

## Earlier preprint (Graph-SND)

The source of the earlier Graph-SND preprint (arXiv:2605.05020), superseded by the
JMLR manuscript above, is in `Paper/`:

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

If you use Graph-SND in your work, please cite the following paper.
Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff).

**Preprint (arXiv):**
```bibtex
@misc{ray2026graphsnd,
  title         = {Graph-{SND}: Sparse Aggregation for Behavioral Diversity
                   in Multi-Agent Reinforcement Learning},
  author        = {Shawn Ray},
  year          = 2026,
  eprint        = {2605.05020},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.05020},
  url           = {https://arxiv.org/abs/2605.05020}
}
```

## License

MIT, see [`LICENSE`](LICENSE). The DiCo fork retains its upstream
license; see
[`ControllingBehavioralDiversity-fork/LICENSE`](ControllingBehavioralDiversity-fork/LICENSE).

## Contact

Shawn Ray, Carnegie Mellon University, <shawnray@cmu.edu>.
