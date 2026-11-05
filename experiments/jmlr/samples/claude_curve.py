"""E7 curve: Claude-judged diversity of 1,000 responses per temperature from ONE randomized
round-robin design per temperature (k = 4 rounds, m = 2n = 2,000 calls), with the exactly unbiased
design-based variance estimate (Prop. rr-variance) and a t-interval; T = 1 gets a second, independent
design as a consistency check. All pairs would need 499,500 calls per temperature."""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import anthropic

gen_path, out_path = sys.argv[1], sys.argv[2]
K = 4
WORKERS = int(os.environ.get("WORKERS", 40))
SYSTEM = ("You compare two short texts written in response to the same request. Rate how different "
          "they are, in content and in style, on an integer scale from 0 to 10: 0 means essentially "
          "the same text, 10 means completely different. Reply with the integer only.")
client = anthropic.Anthropic(max_retries=8)  # reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if set)
G = json.load(open(gen_path))
prompt = G["prompt"]


def judge(a, b):
    for _ in range(4):
        r = client.messages.create(model="claude-opus-5-5", max_tokens=2000, output_config={"effort": "low"},
                                   system=SYSTEM, messages=[{"role": "user", "content":
                                   f"Request: {prompt}\n\nText A: {a}\n\nText B: {b}"}])
        txt = "".join(getattr(c, "text", "") for c in r.content if c.type == "text").strip()
        m = re.fullmatch(r"\D*?(\d{1,2})\D*", txt)
        if m and 0 <= int(m.group(1)) <= 10:
            return int(m.group(1)), r.usage.input_tokens, r.usage.output_tokens
    return None, r.usage.input_tokens, r.usage.output_tokens


def rounds_circle(n):
    R = np.zeros((n - 1, n // 2, 2), dtype=int)
    for r in range(n - 1):
        R[r, 0] = (n - 1, r)
        for k in range(1, n // 2):
            R[r, k] = ((r + k) % (n - 1), (r - k) % (n - 1))
    return R


rng = np.random.default_rng(31)
tq3 = 3.182446305284263  # t_{0.975, 3}
jobs = []
for pop in G["populations"]:
    reps = 2 if pop["temperature"] == 1.0 else 1
    for rep in range(reps):
        texts = [t if t else "(empty)" for t in pop["texts"]]
        n = len(texts)
        perm = rng.permutation(n)
        sel = rng.choice(n - 1, size=K, replace=False)
        P = perm[rounds_circle(n)[sel]]          # (K, n/2, 2)
        flip = rng.random(P.shape[:2]) < 0.5
        jobs.append({"T": pop["temperature"], "rep": rep, "texts": texts, "P": P, "flip": flip})

t0 = time.time()
allpairs = [(j, r, c) for j, job in enumerate(jobs) for r in range(K) for c in range(job["P"].shape[1])]
print(f"{len(allpairs)} calls", flush=True)
results = {}
with ThreadPoolExecutor(WORKERS) as ex:
    def task(x):
        j, r, c = x
        job = jobs[j]
        i1, i2 = job["P"][r, c]
        if job["flip"][r, c]:
            i1, i2 = i2, i1
        return x, judge(job["texts"][i1], job["texts"][i2])
    for k, (x, res) in enumerate(ex.map(task, allpairs)):
        results[x] = res
        if (k + 1) % 1000 == 0:
            print(f"{k+1}/{len(allpairs)} in {time.time()-t0:.0f}s", flush=True)

out = {"model": "claude-opus-5-5", "effort": "low", "K": K, "n": 1000, "prompt": prompt, "wall_seconds": time.time() - t0,
       "workers": WORKERS, "designs": []}
for j, job in enumerate(jobs):
    n = len(job["texts"])
    S = np.array([[results[(j, r, c)][0] if results[(j, r, c)][0] is not None else np.nan
                   for c in range(job["P"].shape[1])] for r in range(K)], float) / 10.0
    fails = int(np.isnan(S).sum())
    rm = np.nanmean(S, axis=1)
    T = float(rm.mean())
    V = (1 - K / (n - 1)) * rm.var(ddof=1) / K
    half = tq3 * np.sqrt(V)
    # sparse distinctiveness (Prop. sparse-anatomy): a_hat_i = (n-1)/(n-2) (mean of i's comparisons - T)
    I = job["P"][..., 0].ravel(); J = job["P"][..., 1].ravel(); d = S.ravel()
    ok = ~np.isnan(d)
    deg = np.bincount(np.r_[I[ok], J[ok]], minlength=n)
    ms = np.bincount(np.r_[I[ok], J[ok]], weights=np.r_[d[ok], d[ok]], minlength=n)
    ahat = (n - 1) / (n - 2) * (ms / np.maximum(deg, 1) - T)
    order = np.argsort(-ahat)
    tin = sum(results[(j, r, c)][1] for r in range(K) for c in range(job["P"].shape[1]))
    tout = sum(results[(j, r, c)][2] for r in range(K) for c in range(job["P"].shape[1]))
    out["designs"].append({"temperature": job["T"], "rep": job["rep"], "calls": int(S.size), "failures": fails,
                           "mu_hat": T, "ci95": [T - half, T + half], "rel_halfwidth": float(half / T),
                           "round_means": rm.tolist(), "input_tokens": int(tin), "output_tokens": int(tout),
                           "cost_usd": (tin * 4 + tout * 20) / 1e6,
                           "most_distinct": [{"a_hat": float(ahat[i]), "text": job["texts"][i]} for i in order[:3]],
                           "least_distinct": [{"a_hat": float(ahat[i]), "text": job["texts"][i]} for i in order[-3:]],
                           "scores": S.tolist(), "pairs": job["P"].tolist()})
    print(f"T={job['T']} rep={job['rep']}: mu_hat={T:.4f} CI=[{T-half:.4f},{T+half:.4f}] (+-{100*half/T:.2f}%) "
          f"fails={fails} cost=${(tin*4+tout*20)/1e6:.2f}", flush=True)
json.dump(out, open(out_path, "w"))
print("saved", out_path, f"{time.time()-t0:.0f}s")
