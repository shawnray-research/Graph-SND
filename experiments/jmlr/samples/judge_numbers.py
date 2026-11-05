"""Collects every number quoted in the judge subsection into one JSON (traceability)."""
import json, sys
import numpy as np
A = {}
for f in ("analysis_qwen.json", "analysis_judge2.json", "analysis_claude.json"):
    for p in json.load(open(f))["populations"]:
        A[p["name"]] = p
X = json.load(open("claude_extra.json"))
F = json.load(open("samples_fmnist.json"))
single = [n for n in A if not n.startswith(("claude_", "mixture", "crossmodel", "contaminated"))]
qwen_single = [n for n in single if not n.startswith("llama_")]
llama_single = [n for n in single if n.startswith("llama_")]
ratios = []
for p in A.values():
    for b in p["budgets"]:
        for dz in ("balanced", "uniform", "bernoulli_ht"):
            ratios.append(b[dz]["var_ratio_meas_pred"])
ratios = np.array(ratios)
def rng(names, key, scale=1.0):
    v = [scale * A[n][key] for n in names]
    return [float(min(v)), float(max(v))]
def at(name, k, dz, key="rel_rmse"):
    b = [b for b in A[name]["budgets"] if b["k"] == k][0]
    return b[dz][key]
cov = [b["balanced"]["ci95_coverage"] for p in A.values() for b in p["budgets"] if b["k"] in (4, 8)]
rho4 = [at(n, 4, "balanced", "spearman_ahat_a_median") for n in single]
N = {
  "n_tables_cross_encoder": sum(1 for n in A if not n.startswith("claude_")),
  "n_tables_claude": sum(1 for n in A if n.startswith("claude_")),
  "n_cells_law_check": int(len(ratios)),
  "var_ratio_median": float(np.median(ratios)), "var_ratio_p5_p95": [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))],
  "mu_qwen_T0.5": rng([n for n in qwen_single if n.endswith("T0.5")], "mu"),
  "mu_qwen_T1.5": rng([n for n in qwen_single if n.endswith("T1.5")], "mu"),
  "mu_llama_T0.5": rng([n for n in llama_single if n.endswith("T0.5")], "mu"),
  "mu_llama_T1.5": rng([n for n in llama_single if n.endswith("T1.5")], "mu"),
  "distinct_share_single_prompt_qwen": rng(qwen_single, "distinct_share", 100),
  "distinct_share_single_prompt_llama": rng(llama_single, "distinct_share", 100),
  "distinct_share_single_prompt_all": rng(single, "distinct_share", 100),
  "efficiency_single_prompt_all": rng(single, "efficiency_pred"),
  "mixture": {k: A["mixture_T1.0"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred")},
  "crossmodel": {k: A["crossmodel_story_T1.0"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred")} if "crossmodel_story_T1.0" in A else None,
  "contaminated": {k: A["contaminated_story_T0.5"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred")} if "contaminated_story_T0.5" in A else None,
  "story_T0.5_qwen": {k: A["story_T0.5"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred", "vs2")},
  "story_T1.0_qwen": {k: A["story_T1.0"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred", "vs2")},
  "story_T1.5_qwen": {k: A["story_T1.5"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred", "vs2")},
  "ce_story_T1.0_m2n": {dz: [at("story_T1.0", 4, dz), at("story_T1.0", 4, dz, "rel_sd_pred")] for dz in ("balanced", "uniform", "bernoulli_ht")},
  "ce_story_T1.0_m2n_subset": at("story_T1.0", 4, "subset"),
  "claude_T1.0": {k: A["claude_story_T1.0"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred", "vs2")},
  "claude_T0.5": {k: A["claude_story_T0.5"][k] for k in ("mu", "distinct_share", "kappa", "efficiency_pred", "vs2")},
  "claude_T1.0_m2n": {dz: [at("claude_story_T1.0", 4, dz), at("claude_story_T1.0", 4, dz, "rel_sd_pred")] for dz in ("balanced", "uniform", "bernoulli_ht")},
  "claude_T1.0_m2n_subset": at("claude_story_T1.0", 4, "subset"),
  "claude_T1.0_m2n_ci_coverage": at("claude_story_T1.0", 4, "balanced", "ci95_coverage"),
  "claude_T1.0_m2n_ci_width": at("claude_story_T1.0", 4, "balanced", "ci95_rel_width_median"),
  "claude_T1.0_m2n_rho": at("claude_story_T1.0", 4, "balanced", "spearman_ahat_a_median"),
  "ci95_coverage_k4_k8_range_all_tables": [float(min(cov)), float(max(cov))],
  "spearman_ahat_a_k4_single_prompt_range": [float(min(rho4)), float(max(rho4))],
  "subset_over_balanced_rmse_m2n_single": [float(min(at(n, 4, "subset") / at(n, 4, "balanced") for n in single)), float(max(at(n, 4, "subset") / at(n, 4, "balanced") for n in single))],
  "ht_over_balanced_rmse_m2n_all": [float(min(at(n, 4, "bernoulli_ht") / at(n, 4, "balanced") for n in A)), float(max(at(n, 4, "bernoulli_ht") / at(n, 4, "balanced") for n in A))],
  "vs2_rel_rmse_m2n_balanced_vs_uniform_single": [[float(at(n, 4, "balanced", "vs2_rel_rmse")), float(at(n, 4, "uniform", "vs2_rel_rmse"))] for n in ("story_T1.0", "claude_story_T1.0")],
  "judge_seconds_per_population_ce": rng([n for n in A if not n.startswith("claude_")], "judge_seconds_all_pairs"),
  "judge_ms_per_call_ce_median": float(np.median([A[n]["judge_ms_per_call"] for n in A if not n.startswith("claude_")])),
  "judge_asym_mean_abs_ce": rng([n for n in A if not n.startswith("claude_")], "judge_asym_mean_abs"),
  "claude_extra": {k: {kk: vv for kk, vv in v.items() if kk not in ("most_distinct", "least_distinct", "score_hist")} for k, v in X.items()},
  "fmnist": [{k: p[k] for k in ("name", "n", "mu", "cv", "distinct_share", "kappa", "vs2", "exact_seconds_gpu", "balanced_k4_seconds_gpu")} |
             {"m_half": [{dz: [b[dz]["rel_rmse"], b[dz]["rel_sd_pred"], b[dz]["vs2_rel_rmse"]] for dz in ("balanced", "uniform", "bernoulli_ht", "subset")} for b in p["budgets"] if b["k"] == 1][0]}
             for p in F["populations"]],
  "fmnist_feature_net_test_acc": F["feature_net_test_acc"],
}
json.dump(N, open("judge_numbers.json", "w"), indent=1)
print(json.dumps(N, indent=1)[:6000])
