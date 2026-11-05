"""Anatomy of the long VMAS runs (n = 16), Section 5.4 and Figure 4.

The main anatomy runs stop after 10-205 PPO iterations (13-minute budgets on shared GPUs).
These runs use the same run_e3.py settings with 42-49 minute budgets on free hardware:
IPPO Dispersion (seeds 0-2), IPPO Navigation (seeds 0-1), and Dispersion with a centralized
critic (--critic central, MAPPO-style; seeds 0-1).

For every saved table (trained snapshots and the final one-agent-reset / half-reset variants)
it writes one row in the format of results/jmlr/fig_marl_data.json (task, n, seed, it, kind,
mu, share_main, pred, meas) plus the training algorithm, and it prints a per-run summary.

Usage: python long_runs.py "<glob of *.npz>" out_rows.json [out_summary.json]
"""
import sys, os, glob, json, re, collections
import numpy as np

rng = np.random.default_rng(0)


def anatomy(D):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    mu = D[iu].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    np.fill_diagonal(R, 0)
    tot = ((D[iu] - mu) ** 2).mean()
    inter = (R[iu] ** 2).mean()
    return mu, tot, inter


def measured_ratio(D, k=4, reps=4000):  # identical to experiments/jmlr/fig_marl.py
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    dvec = D[iu]
    N = len(dvec)
    m = n * k // 2
    base_i = np.array([t for t in range(n) for s in range(1, k // 2 + 1)])
    base_j = np.array([(t + s) % n for t in range(n) for s in range(1, k // 2 + 1)])
    eb = np.empty(reps)
    eu = np.empty(reps)
    for r in range(reps):
        p = rng.permutation(n)
        eb[r] = D[p[base_i], p[base_j]].mean()
        eu[r] = dvec[rng.choice(N, m, replace=False)].mean()
    return eu.var() / eb.var()


rows = []
for f in sorted(glob.glob(sys.argv[1])):
    m = re.match(r"(\w+?)_n(\d+)_s(\d+)(_mappo)?_(trained|outlier1|twogroup)_it(\d+)\.npz", os.path.basename(f))
    if not m:
        continue
    z = np.load(f, allow_pickle=True)
    D = z["D"].astype(float)
    n = D.shape[0]
    mu, tot, inter = anatomy(D)
    kappa = (tot - inter) / inter
    rows.append(dict(task=str(z["task"]), n=n, seed=int(z["seed"]), it=int(z["iter"]), kind=str(z["kind"]),
                     algo="central" if m.group(4) else "independent", reward=float(z["reward"]),
                     mu=float(mu), share_main=float((tot - inter) / tot),
                     pred=float((1 + kappa) * n * (n - 3) / ((n - 2) * (n + 1))), meas=float(measured_ratio(D))))
json.dump(rows, open(sys.argv[2], "w"), indent=0)

summ = {}
runs = collections.defaultdict(list)
for r in rows:
    if r["kind"] == "trained":
        runs[(r["task"], r["n"], r["seed"], r["algo"])].append(r)
for key in sorted(runs):
    traj = sorted(runs[key], key=lambda r: r["it"])
    last = traj[-1]
    early = [r for r in traj if 5 <= r["it"] <= 10]
    peak = max(early, key=lambda r: r["share_main"]) if early else None
    late = [r for r in traj if r["it"] >= 5]
    s = dict(last_it=last["it"], env_steps_million=last["it"] * 64 * 128 / 1e6, final_share=last["share_main"],
             final_mu=last["mu"], mu_growth=last["mu"] / traj[0]["mu"], final_reward=last["reward"],
             peak_share_it5_10=peak["share_main"] if peak else None,
             drop_points=100 * (peak["share_main"] - last["share_main"]) if peak else None,
             share_range_from_it5=[min(r["share_main"] for r in late), max(r["share_main"] for r in late)] if late else None,
             rewards=[(r["it"], round(r["reward"], 5)) for r in traj])
    summ["/".join(map(str, key))] = s
    print(key, " ".join(f"{r['it']}:{100*r['share_main']:.0f}%" for r in traj))
    print("    final it", s["last_it"], f"({s['env_steps_million']:.1f}M steps) share {100*s['final_share']:.0f}%",
          f"peak(5-10) {100*s['peak_share_it5_10']:.0f}%" if peak else "", f"drop {s['drop_points']:.0f}pt" if peak else "",
          f"mu {s['final_mu']:.3f} (x{s['mu_growth']:.0f})", "rewards", s["rewards"][-4:])
ok = [r["meas"] / r["pred"] for r in rows]
print("tables", len(rows), "meas/pred median", np.median(ok), "range", min(ok), max(ok))
if len(sys.argv) > 3:
    json.dump(summ, open(sys.argv[3], "w"), indent=1)
