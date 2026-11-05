"""Late-window statistics for the balanced-design DiCo runs (Table tab:dico, Sec. 5.8).

Compares the randomized balanced design (fresh random 5-regular graph per call,
125 of 1,225 pairs) against the paper's Bernoulli (p = 0.1) and all-pairs runs,
cell by cell (seed x set point), over the final 50 of 167 iterations.

Usage (from the repository root):
    python experiments/jmlr/dico/dico_balanced_analysis.py
Inputs : results/jmlr/dico_balanced/runs/seed{S}/snd{0pXX}/{balanced,full}/graph_snd_log.csv
         results/dico_n50_posthoc_full_snd/n50_posthoc_full_snd_per_cell.csv  (earlier runs)
Output : results/jmlr/dico_balanced/summary.json (and a printed table)
Incomplete runs (< 167 iterations) are listed but excluded from every comparison.
"""
import csv
import glob
import json
import math
import os
import statistics as st

LATE = 50
FULL_ITERS = 167
ROOT = "results/jmlr/dico_balanced/runs"
PAPER = "results/dico_n50_posthoc_full_snd/n50_posthoc_full_snd_per_cell.csv"
OUT = "results/jmlr/dico_balanced/summary.json"
T975_DF5 = 2.5706


def summarize(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("iter", "") != ""]
    late = rows[-LATE:]
    des = float(late[-1]["snd_des"])

    def col(name):
        out = []
        for r in late:
            try:
                x = float(r[name])
            except (KeyError, ValueError):
                continue
            if math.isfinite(x):
                out.append(x)
        return out

    applied, audit = col("applied_snd"), col("posthoc_full_snd")
    return {
        "n_iters": len(rows), "snd_des": round(des, 2),
        "applied_mean": st.mean(applied), "applied_err_rel": abs(st.mean(applied) / des - 1),
        "audit_mean": st.mean(audit), "audit_err_rel": abs(st.mean(audit) / des - 1),
        "reward_late_mean": st.mean(col("reward_mean")),
        "metric_time_ms_median": st.median(col("metric_time_ms")),
        "iter_time_ms_median": st.median(col("iter_time_ms")),
    }


def paired(diffs):
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    return {"n": len(diffs), "mean": m, "se": se, "t": m / se,
            "ci95": [m - T975_DF5 * se, m + T975_DF5 * se] if len(diffs) == 6 else None}


def main():
    paper = {(r["estimator"], round(float(r["snd_des"]), 2), int(r["seed"])): r
             for r in csv.DictReader(open(PAPER))}
    cells = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "seed*", "snd*", "*", "graph_snd_log.csv"))):
        parts = path.split(os.sep)
        seed, est = int(parts[-4][4:]), parts[-2]
        s = summarize(path)
        cells[(est, s["snd_des"], seed)] = s
    done = {k: v for k, v in cells.items() if v["n_iters"] >= FULL_ITERS}
    print("incomplete (excluded):", sorted(f"{k[0]} {k[1]} seed{k[2]}: {v['n_iters']}/{FULL_ITERS}"
                                         for k, v in cells.items() if k not in done))
    targets = sorted({k[1] for k in done if k[0] == "balanced"})
    per_target = {}
    for t in targets:
        b = [done[("balanced", t, s)] for s in (0, 1, 2)]
        rw = [x["reward_late_mean"] for x in b]
        full_new = done.get(("full", t, 0))
        mt = st.median(x["metric_time_ms_median"] for x in b)
        per_target[str(t)] = {
            "audit_err_rel_seed_avg": abs(st.mean(x["audit_mean"] for x in b) / t - 1),
            "audit_err_rel_max_seed": max(x["audit_err_rel"] for x in b),
            "reward_mean": st.mean(rw), "reward_sem": st.stdev(rw) / math.sqrt(len(rw)),
            "metric_time_ms_median": mt,
            "full_same_stack_metric_time_ms": full_new["metric_time_ms_median"] if full_new else None,
            "speedup_same_stack": full_new["metric_time_ms_median"] / mt if full_new else None,
        }
    comps = {}
    keymap = {"reward_late_mean": "reward_late_mean", "audit_err_rel": "posthoc_full_snd_err_rel",
              "applied_err_rel": "applied_err_rel"}
    for ref in ("bern", "full"):
        for metric, pcol in keymap.items():
            d = [done[("balanced", t, s)][metric] - float(paper[(ref, t, s)][pcol])
                 for t in targets for s in (0, 1, 2)]
            comps[f"balanced_minus_{ref}_{metric}"] = paired(d)
    # seed-level comparisons: the set points of one seed share initial conditions and their
    # differences are strongly correlated, so the paper treats seeds as the unit (t on 2 df)
    T975_DF2 = 4.3027
    mean_reward = st.mean(float(r["reward_late_mean"]) for r in paper.values())

    def seed_level(diff_fn, sets):
        sm = [st.mean(diff_fn(t, s) for t in sets) for s in (0, 1, 2)]
        m, se = st.mean(sm), st.stdev(sm) / math.sqrt(3)
        return {"seed_means": sm, "mean": m, "se": se, "t": m / se,
                "ci95": [m - T975_DF2 * se, m + T975_DF2 * se],
                "ci95_pct_of_mean_reward": [100 * (m - T975_DF2 * se) / mean_reward, 100 * (m + T975_DF2 * se) / mean_reward]}
    rew = lambda e, t, s: float(paper[(e, t, s)]["reward_late_mean"])
    aud = lambda e, t, s: float(paper[(e, t, s)]["posthoc_full_snd_err_rel"])
    seed_lvl = {
        "reward_bernoulli_minus_full": seed_level(lambda t, s: rew("bern", t, s) - rew("full", t, s), (0.12, 0.14, 0.15)),
        "reward_balanced_minus_full": seed_level(lambda t, s: done[("balanced", t, s)]["reward_late_mean"] - rew("full", t, s), targets),
        "reward_balanced_minus_bernoulli": seed_level(lambda t, s: done[("balanced", t, s)]["reward_late_mean"] - rew("bern", t, s), targets),
        "audit_pp_balanced_minus_full": seed_level(lambda t, s: 100 * (done[("balanced", t, s)]["audit_err_rel"] - aud("full", t, s)), targets),
        "audit_pp_balanced_minus_bernoulli": seed_level(lambda t, s: 100 * (done[("balanced", t, s)]["audit_err_rel"] - aud("bern", t, s)), targets),
    }
    stack = {str(t): {"new_full": done[("full", t, 0)],
                      "paper_full_seed0": {k: float(v) for k, v in paper[("full", t, 0)].items()
                                           if k not in ("estimator",)}}
             for t in targets if ("full", t, 0) in done}
    out = {"late_window": LATE, "cells": {f"{k[0]}/{k[1]}/seed{k[2]}": v for k, v in sorted(cells.items())},
           "per_target_balanced": per_target, "paired_vs_earlier_runs": comps, "paired_seed_level": seed_lvl,
           "stack_check": stack}
    json.dump(out, open(OUT, "w"), indent=1)
    for t, v in per_target.items():
        print(f"SND_des={t}: audit {100*v['audit_err_rel_seed_avg']:.2f}% (max seed {100*v['audit_err_rel_max_seed']:.2f}%), "
              f"reward {v['reward_mean']:.3f}+-{v['reward_sem']:.3f}, {v['metric_time_ms_median']:.1f} ms/call, "
              f"speed-up vs same-stack all pairs {v['speedup_same_stack']:.1f}x")
    for k, v in comps.items():
        print(f"{k}: {v['mean']:+.5f} +- {v['se']:.5f} (t = {v['t']:+.2f}, n = {v['n']} cells)")
    for k, v in seed_lvl.items():
        print(f"seed level {k}: {v['mean']:+.5f} +- {v['se']:.5f} (t = {v['t']:+.2f}, 2 df), "
              f"95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] = [{v['ci95_pct_of_mean_reward'][0]:+.1f}%, {v['ci95_pct_of_mean_reward'][1]:+.1f}%] of mean reward")


if __name__ == "__main__":
    main()
