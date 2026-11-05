"""Summarize DiCo n=50 runs: balanced (new) vs Bernoulli/all-pairs (paper's audited runs).

Late window = final 50 iterations, as in the paper. For each cell (seed, SND_des)
we report the applied (online) tracking error, the post-hoc all-pairs audit error,
the mean episode reward, and the median per-call metric time.

Usage: python dico_balanced_summary.py <balanced_root> <audited_root> <out.json>
  balanced_root: .../n50_balanced/seed{S}/snd{0pXX}/balanced/graph_snd_log.csv
  audited_root : .../neurips_final_n50_posthoc_full_snd/seed{S}/snd{0pXX}/{bern,full}/graph_snd_log.csv
"""
import csv
import glob
import json
import math
import os
import statistics as st
import sys

LATE = 50


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def summarize(rows):
    rows = [r for r in rows if r.get("iter", "") != ""]
    late = rows[-LATE:]
    des = float(late[-1]["snd_des"])

    def col(name):
        out = []
        for r in late:
            v = r.get(name, "")
            try:
                x = float(v)
            except ValueError:
                continue
            if math.isfinite(x):
                out.append(x)
        return out

    applied = col("applied_snd")
    posthoc = col("posthoc_full_snd")
    reward = col("reward_mean")
    mtime = col("metric_time_ms")
    res = {
        "n_iters": len(rows),
        "snd_des": des,
        "applied_err_rel": abs(st.mean(applied) - des) / des if applied else None,
        "posthoc_err_rel": abs(st.mean(posthoc) - des) / des if posthoc else None,
        "reward_late_mean": st.mean(reward) if reward else None,
        "metric_time_ms_median": st.median(mtime) if mtime else None,
    }
    return res


def main():
    bal_root, aud_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cells = {}
    for path in glob.glob(os.path.join(bal_root, "seed*", "snd*", "balanced", "graph_snd_log.csv")):
        parts = path.split(os.sep)
        seed, tag = parts[-4], parts[-3]
        cells.setdefault((seed, tag), {})["balanced"] = summarize(load(path))
    for est in ("bern", "full"):
        for path in glob.glob(os.path.join(aud_root, "seed*", "snd*", est, "graph_snd_log.csv")):
            parts = path.split(os.sep)
            seed, tag = parts[-4], parts[-3]
            if (seed, tag) in cells:
                cells[(seed, tag)][est] = summarize(load(path))
    table = {f"{s}/{t}": v for (s, t), v in sorted(cells.items())}
    # paired reward differences (balanced minus all-pairs), over complete cells
    diffs = [v["balanced"]["reward_late_mean"] - v["full"]["reward_late_mean"]
             for v in table.values() if "full" in v and v["balanced"]["n_iters"] >= 167]
    summary = {"cells": table, "n_complete_cells": len(diffs)}
    if len(diffs) >= 2:
        m = st.mean(diffs)
        se = st.stdev(diffs) / math.sqrt(len(diffs))
        summary["paired_reward_diff_bal_minus_full"] = {"mean": m, "se": se, "t": m / se if se > 0 else None}
    json.dump(summary, open(out, "w"), indent=1)
    for k, v in table.items():
        print(k, {e: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in d.items()} for e, d in v.items()})
    print({k: v for k, v in summary.items() if k != "cells"})


if __name__ == "__main__":
    main()
