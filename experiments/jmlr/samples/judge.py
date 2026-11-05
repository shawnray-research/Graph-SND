"""E7 step 2: all-pairs judged dissimilarity with a cross-encoder (one forward pass per ordered pair).
d_ij = 1 - (s(i,j) + s(j,i)) / 2, s = sigmoid(logit) of an STS cross-encoder, in [0, 1]."""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

gen_path, out_path = sys.argv[1], sys.argv[2]
judge_id = os.environ.get("JUDGE", "cross-encoder/stsb-roberta-base")
BS = int(os.environ.get("BS", 1024))
tok = AutoTokenizer.from_pretrained(judge_id)
model = AutoModelForSequenceClassification.from_pretrained(judge_id, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def score(a_list, b_list):
    out = np.empty(len(a_list), dtype=np.float64)
    for s in range(0, len(a_list), BS):
        enc = tok(a_list[s:s + BS], b_list[s:s + BS], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to("cuda")
        out[s:s + BS] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
    return out


gen = json.load(open(gen_path))
pops = gen["populations"]
# extra population: prompt mixture (first 100 samples of each prompt at T = 1.0)
mix = [p for p in pops if p["temperature"] == 1.0]
pops = pops + [{"prompt_id": "mixture", "prompt": "three prompts", "temperature": 1.0,
                "texts": sum([p["texts"][:100] for p in mix], [])}]
tables = {}
meta = []
for k, p in enumerate(pops):
    texts = [t if t else "(empty)" for t in p["texts"]]
    n = len(texts)
    iu, ju = np.triu_indices(n, 1)
    A = [texts[i] for i in iu]
    Bt = [texts[j] for j in ju]
    torch.cuda.synchronize(); t0 = time.time()
    s1 = score(A, Bt)
    s2 = score(Bt, A)
    torch.cuda.synchronize(); el = time.time() - t0
    D = np.zeros((n, n))
    D[iu, ju] = 1.0 - 0.5 * (s1 + s2)
    D = D + D.T
    name = f"{p['prompt_id']}_T{p['temperature']}"
    tables[name] = D
    meta.append({"name": name, "prompt_id": p["prompt_id"], "temperature": p["temperature"], "n": n,
                 "pairs": int(len(iu)), "judge_calls": int(2 * len(iu)), "judge_seconds": el,
                 "asym_mean_abs": float(np.mean(np.abs(s1 - s2)))})
    print(name, n, f"{el:.1f}s", f"mu={D[iu, ju].mean():.4f}", flush=True)
np.savez_compressed(out_path, **tables)
json.dump({"judge": judge_id, "generator": gen["model"], "tables": meta,
           "texts": {f"{p['prompt_id']}_T{p['temperature']}": p["texts"] for p in pops}},
          open(out_path.replace(".npz", "_meta.json"), "w"), indent=1)
print("saved", out_path)
