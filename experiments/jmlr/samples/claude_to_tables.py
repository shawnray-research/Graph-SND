"""Convert Claude-judge JSON files into the table format of samples_judge_analysis.py."""
import json, sys
import numpy as np
out_npz = sys.argv[1]
files = sys.argv[2:]
tables, meta_t, texts = {}, [], {}
for f in files:
    r = json.load(open(f))
    D = np.array(r["D"], dtype=float)
    assert not np.isnan(D).any(), f"missing scores in {f}"
    name = "claude_" + r["name"]
    tables[name] = D
    texts[name] = r["texts"]
    meta_t.append({"name": name, "n": r["n"], "pairs": r["pairs"], "judge_calls": r["calls"],
                   "judge_seconds": r["wall_seconds"] * r["workers"],   # summed call-seconds (approx.)
                   "wall_seconds": r["wall_seconds"], "workers": r["workers"],
                   "latency_s_median": r["latency_s_median"], "asym_mean_abs": float("nan"),
                   "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]})
np.savez_compressed(out_npz, **tables)
json.dump({"judge": "claude-opus-5-5 (effort low)", "generator": "Qwen/Qwen2.5-0.5B-Instruct",
           "tables": meta_t, "texts": texts}, open(out_npz.replace(".npz", "_meta.json"), "w"), indent=1)
print("saved", out_npz, list(tables))
