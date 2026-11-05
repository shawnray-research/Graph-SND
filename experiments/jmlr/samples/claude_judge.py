"""E7 (frontier LLM judge): all-pairs judged dissimilarity of LM samples with Claude Opus 5.5.

Each unordered pair {i, j} gets ONE judge call with a random A/B orientation; the judge returns an
integer 0..10 ("how different, in content and style"); d_ij = score / 10, d_ii = 0.
A retest of 200 random pairs with the opposite orientation measures judge noise."""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import anthropic

gen_path, out_dir = sys.argv[1], sys.argv[2]
NPOP = int(os.environ.get("NPOP", 64))
WORKERS = int(os.environ.get("WORKERS", 24))
MODEL = "claude-opus-5-5"
SYSTEM = ("You compare two short texts written in response to the same request. Rate how different "
          "they are, in content and in style, on an integer scale from 0 to 10: 0 means essentially "
          "the same text, 10 means completely different. Reply with the integer only.")
client = anthropic.Anthropic(max_retries=6)  # reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if set)
G = json.load(open(gen_path))
P = {f"{p['prompt_id']}_T{p['temperature']}": p for p in G["populations"]}
lock = threading.Lock()


def judge(prompt, a, b):
    for attempt in range(4):
        t0 = time.time()
        r = client.messages.create(
            model=MODEL, max_tokens=2000, output_config={"effort": "low"}, system=SYSTEM,
            messages=[{"role": "user", "content": f"Request: {prompt}\n\nText A: {a}\n\nText B: {b}"}])
        txt = "".join(getattr(c, "text", "") for c in r.content if c.type == "text").strip()
        m = re.fullmatch(r"\D*?(\d{1,2})\D*", txt)
        if m and 0 <= int(m.group(1)) <= 10:
            return int(m.group(1)), time.time() - t0, r.usage.input_tokens, r.usage.output_tokens, txt
    return None, time.time() - t0, r.usage.input_tokens, r.usage.output_tokens, txt


def run(name, texts, prompt, pairs_oriented, tag):
    res = [None] * len(pairs_oriented)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(judge, prompt, texts[a], texts[b]): k for k, (a, b) in enumerate(pairs_oriented)}
        for f in as_completed(futs):
            k = futs[f]
            try:
                res[k] = f.result()
            except Exception as e:  # keep going; record the failure
                res[k] = (None, 0.0, 0, 0, f"ERROR {type(e).__name__}: {e}")
            done += 1
            if done % 200 == 0:
                print(f"{name}/{tag}: {done}/{len(pairs_oriented)} in {time.time()-t0:.0f}s", flush=True)
    return res, time.time() - t0


rng = np.random.default_rng(2026)
summary = {"model": MODEL, "effort": "low", "system": SYSTEM, "populations": []}
for name in ("story_T1.0", "story_T0.5"):
    texts = [t if t else "(empty)" for t in P[name]["texts"][:NPOP]]
    prompt = P[name]["prompt"]
    n = len(texts)
    iu, ju = np.triu_indices(n, 1)
    flip = rng.random(len(iu)) < 0.5
    oriented = [(int(j), int(i)) if f else (int(i), int(j)) for i, j, f in zip(iu, ju, flip)]
    res, wall = run(name, texts, prompt, oriented, "all-pairs")
    D = np.zeros((n, n))
    fails = 0
    for (a, b), r in zip(oriented, res):
        s = r[0]
        if s is None:
            fails += 1; s = np.nan
        D[a, b] = D[b, a] = s / 10.0
    rec = {"name": name, "n": n, "pairs": int(len(iu)), "calls": int(len(iu)), "wall_seconds": wall,
           "workers": WORKERS, "failures": fails,
           "latency_s_median": float(np.median([r[1] for r in res])),
           "input_tokens": int(sum(r[2] for r in res)), "output_tokens": int(sum(r[3] for r in res)),
           "texts": texts, "D": D.tolist(),
           "raw": [{"a": a, "b": b, "score": r[0], "text": r[4]} for (a, b), r in zip(oriented, res)]}
    if name == "story_T1.0":
        sel = rng.choice(len(oriented), size=200, replace=False)
        re_or = [(oriented[k][1], oriented[k][0]) for k in sel]
        res2, wall2 = run(name, texts, prompt, re_or, "retest")
        rec["retest"] = [{"a": a, "b": b, "score": r2[0], "first_score": res[k][0]}
                         for (a, b), r2, k in zip(re_or, res2, sel)]
        rec["retest_input_tokens"] = int(sum(r[2] for r in res2)); rec["retest_output_tokens"] = int(sum(r[3] for r in res2))
    json.dump(rec, open(os.path.join(out_dir, f"claude_{name}.json"), "w"))
    print(f"{name}: done {len(iu)} calls in {wall:.0f}s, failures {fails}, median latency {rec['latency_s_median']:.2f}s, "
          f"tokens in/out {rec['input_tokens']}/{rec['output_tokens']}", flush=True)
print("all done")
