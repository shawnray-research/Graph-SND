"""Claude judge: retest reliability and agreement with the cross-encoder judge on the same texts."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from anatomy import anatomy
ce = np.load("judged_qwen.npz")
out = {}
for name in ("story_T1.0", "story_T0.5"):
    r = json.load(open(f"claude_{name}.json"))
    D = np.array(r["D"]); n = r["n"]
    iu = np.triu_indices(n, 1)
    Dce = ce[name][:n, :n]
    a_cl, a_ce = anatomy(D), anatomy(Dce)
    rec = {"n": n, "mu": a_cl["mu"], "distinct_share": a_cl["s2_main"] / a_cl["s2_tot"],
           "score_hist": np.bincount((10 * D[iu]).round().astype(int), minlength=11).tolist(),
           "pearson_tables_vs_crossencoder": float(pearsonr(D[iu], Dce[iu])[0]),
           "spearman_tables_vs_crossencoder": float(spearmanr(D[iu], Dce[iu])[0]),
           "spearman_a_vs_crossencoder_a": float(spearmanr(a_cl["a"], a_ce["a"])[0]),
           "crossencoder_mu_same_texts": a_ce["mu"],
           "crossencoder_distinct_share_same_texts": a_ce["s2_main"] / a_ce["s2_tot"],
           "cost_usd_all_pairs": (r["input_tokens"] * 4 + r["output_tokens"] * 20) / 1e6,
           "cost_usd_per_call": (r["input_tokens"] * 4 + r["output_tokens"] * 20) / 1e6 / r["calls"],
           "wall_seconds": r["wall_seconds"], "workers": r["workers"], "latency_s_median": r["latency_s_median"],
           "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}
    if "retest" in r:
        x = np.array([q["first_score"] for q in r["retest"]], float); y = np.array([q["score"] for q in r["retest"]], float)
        ok = ~(np.isnan(x) | np.isnan(y))
        rec["retest_pairs"] = int(ok.sum())
        rec["retest_pearson"] = float(pearsonr(x[ok], y[ok])[0])
        rec["retest_mean_abs_diff"] = float(np.mean(np.abs(x[ok] - y[ok])) / 10)
        rec["retest_exact_agree"] = float(np.mean(x[ok] == y[ok]))
        rec["retest_mean_diff"] = float(np.mean(y[ok] - x[ok]) / 10)
    order = np.argsort(-a_cl["a"])
    rec["most_distinct"] = [{"a": float(a_cl["a"][i]), "text": r["texts"][i]} for i in order[:3]]
    rec["least_distinct"] = [{"a": float(a_cl["a"][i]), "text": r["texts"][i]} for i in order[-3:]]
    out[name] = rec
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("most_distinct", "least_distinct")})
    for e in rec["most_distinct"] + rec["least_distinct"]:
        print("   %+.3f" % e["a"], e["text"][:120].replace("\n", " "))
json.dump(out, open("claude_extra.json", "w"), indent=1)
