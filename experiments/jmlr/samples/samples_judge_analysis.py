"""E7 (generated text, neural judge): anatomy, comparison designs, closed-form laws, Vendi-2.

Input: judged_qwen.npz (all-pairs judge tables) + judged_qwen_meta.json (texts, timings).
d_ij = 1 - (s(i,j) + s(j,i))/2 with s = sigmoid(logit) of an STS cross-encoder; d_ii = 0.
Judge scores need not satisfy the triangle inequality: only the exact (any-table) laws apply.
"""
import json, os, sys, unicodedata
import numpy as np
from scipy.stats import t as tdist, spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from anatomy import anatomy, var_uniform, var_balanced, var_bernoulli_ht, efficiency_law, round_robin_rounds

npz, meta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
R = 4000
T = np.load(npz)
meta = json.load(open(meta_path))
info = {t["name"]: t for t in meta["tables"]}
R = int(__import__("os").environ.get("REPS", R))
rng = np.random.default_rng(7)
tq = {k: float(tdist.ppf(0.975, k - 1)) for k in (2, 4, 8)}


def nonlatin_frac(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 1.0
    return float(np.mean([not unicodedata.name(c, "").startswith("LATIN") for c in letters]))


def designs_pairs(n, k, rounds):
    """Yield (name, I, J, weightfun) for one draw of each design at budget m = n k / 2."""
    m = n * k // 2
    N = n * (n - 1) // 2
    perm = rng.permutation(n)
    sel = rng.choice(n - 1, size=k, replace=False)
    P = perm[rounds[sel].reshape(-1, 2)]
    yield "balanced", P[:, 0], P[:, 1], ("rounds", k)
    iu = np.triu_indices(n, 1)
    idx = rng.choice(N, size=m, replace=False)
    yield "uniform", iu[0][idx], iu[1][idx], None
    p = m / N
    mask = rng.random(N) < p
    yield "bernoulli_ht", iu[0][mask], iu[1][mask], p * N
    s = int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    S = rng.choice(n, size=s, replace=False)
    su = np.triu_indices(s, 1)
    yield "subset", S[su[0]], S[su[1]], None


out = {"judge": meta["judge"], "generator": meta.get("generator", "see per-table description"), "reps": R, "populations": []}
for name in T.files:
    D = T[name]
    n = D.shape[0]
    N = n * (n - 1) // 2
    an = anatomy(D)
    iu = np.triu_indices(n, 1)
    Kt = (1.0 - D) ** 2          # squared judge similarity, k_ij = s_ij, k_ii = 1
    np.fill_diagonal(Kt, 0.0)
    t_mu = Kt[iu].mean()
    vs2 = n / (1 + (n - 1) * t_mu)
    ant = anatomy(Kt)
    texts = meta["texts"][name]
    nl = np.array([nonlatin_frac(s) for s in texts])
    order = np.argsort(-an["a"])
    rounds = round_robin_rounds(n)
    pop = {"name": name, "n": n, "pairs": N, "judge_calls_all_pairs": info[name]["judge_calls"],
           "judge_seconds_all_pairs": info[name]["judge_seconds"],
           "judge_ms_per_call": 1e3 * info[name]["judge_seconds"] / info[name]["judge_calls"],
           "judge_asym_mean_abs": info[name]["asym_mean_abs"],
           "mu": an["mu"], "cv": float(np.sqrt(an["s2_tot"]) / an["mu"]),
           "s2_tot": an["s2_tot"], "s2_main": an["s2_main"], "s2_int": an["s2_int"],
           "distinct_share": an["s2_main"] / an["s2_tot"], "kappa": an["kappa"],
           "efficiency_pred": efficiency_law(an), "vs2": vs2,
           "nonlatin_share_texts": float(np.mean(nl > 0.2)),
           "corr_a_nonlatin": float(np.corrcoef(an["a"], nl)[0, 1]) if nl.std() > 0 else None,
           "most_distinct": [{"a": float(an["a"][i]), "text": texts[i]} for i in order[:3]],
           "least_distinct": [{"a": float(an["a"][i]), "text": texts[i]} for i in order[-3:]],
           "budgets": []}
    for k in (1, 2, 4, 8):
        m = n * k // 2
        p = m / N
        ests = {dz: ([], []) for dz in ("balanced", "uniform", "bernoulli_ht", "subset")}
        cover, width, rho = [], [], []
        for r in range(R):
            for dz, I, J, w in designs_pairs(n, k, rounds):
                d, t = D[I, J], Kt[I, J]
                if w is None or isinstance(w, tuple):
                    ests[dz][0].append(d.mean()); ests[dz][1].append(t.mean())
                    if isinstance(w, tuple) and k >= 2:
                        rm = d.reshape(k, n // 2).mean(axis=1)          # round means
                        Tm = rm.mean()
                        Vh = (1 - k / (n - 1)) * rm.var(ddof=1) / k     # Prop. rr-variance
                        half = tq[k] * np.sqrt(Vh)
                        cover.append(abs(Tm - an["mu"]) <= half); width.append(2 * half / an["mu"])
                        deg = np.bincount(np.r_[I, J], minlength=n)
                        msum = np.bincount(np.r_[I, J], weights=np.r_[d, d], minlength=n)
                        ahat = (n - 1) / (n - 2) * (msum / deg - Tm)
                        rho.append(spearmanr(ahat, an["a"])[0])
                else:
                    ests[dz][0].append(d.sum() / w); ests[dz][1].append(t.sum() / w)
        pred = {"balanced": var_balanced(an, k), "uniform": var_uniform(an, m),
                "bernoulli_ht": var_bernoulli_ht(D, p), "subset": None}
        row = {"k": k, "m": m, "judge_calls": 2 * m, "frac_pairs": m / N}
        for dz, (ed, et) in ests.items():
            ed, et = np.array(ed), np.array(et)
            vs_hat = n / (1 + (n - 1) * et)
            row[dz] = {"rel_rmse": float(np.sqrt(np.mean((ed / an["mu"] - 1) ** 2))),
                       "rel_bias": float(ed.mean() / an["mu"] - 1),
                       "rel_sd": float(ed.std(ddof=1) / an["mu"]),
                       "rel_sd_pred": None if pred[dz] is None else float(np.sqrt(pred[dz]) / an["mu"]),
                       "var_ratio_meas_pred": None if pred[dz] is None else float(ed.var(ddof=1) / pred[dz]),
                       "vs2_rel_rmse": float(np.sqrt(np.mean((vs_hat / vs2 - 1) ** 2))),
                       "vs2_rel_bias": float(vs_hat.mean() / vs2 - 1)}
        if cover:
            row["balanced"]["ci95_coverage"] = float(np.mean(cover))
            row["balanced"]["ci95_rel_width_median"] = float(np.median(width))
            row["balanced"]["spearman_ahat_a_median"] = float(np.median(rho))
        pop["budgets"].append(row)
    out["populations"].append(pop)
    b2 = pop["budgets"][2]
    print(f"{name:14s} mu={an['mu']:.4f} cv={pop['cv']:.3f} distinct={pop['distinct_share']:.3f} kappa={an['kappa']:.3f} "
          f"eff={pop['efficiency_pred']:.2f} VS2={vs2:.1f} nonlatin={pop['nonlatin_share_texts']:.2f} "
          f"corr(a,nl)={pop['corr_a_nonlatin']} | m=2n RMSE% bal {100*b2['balanced']['rel_rmse']:.2f} "
          f"(pred {100*b2['balanced']['rel_sd_pred']:.2f}) unif {100*b2['uniform']['rel_rmse']:.2f} "
          f"(pred {100*b2['uniform']['rel_sd_pred']:.2f}) HT {100*b2['bernoulli_ht']['rel_rmse']:.2f} "
          f"sub {100*b2['subset']['rel_rmse']:.2f} | VS2 bal {100*b2['balanced']['vs2_rel_rmse']:.2f} "
          f"unif {100*b2['uniform']['vs2_rel_rmse']:.2f} | CI cov k=4 {b2['balanced'].get('ci95_coverage')} rho {b2['balanced'].get('spearman_ahat_a_median')}", flush=True)
json.dump(out, open(out_path, "w"), indent=1)
print("saved", out_path)
