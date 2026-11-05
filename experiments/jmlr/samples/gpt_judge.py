"""E7: the same 2,016 pairs (same A/B orientation) as the Claude judge, scored by GPT-5 through the
OpenAI Chat Completions API; integer 0..10 -> d = score/10."""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

src, out_path = sys.argv[1], sys.argv[2]
MODEL = os.environ.get("GPT_MODEL", "gpt-5")
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]
SYSTEM = ("You compare two short texts written in response to the same request. Rate how different "
          "they are, in content and in style, on an integer scale from 0 to 10: 0 means essentially "
          "the same text, 10 means completely different. Reply with the integer only.")
R = json.load(open(src))
texts, prompt = R["texts"], "Write a one-sentence story about a robot."


def call(a, b):
    body = {"model": MODEL, "reasoning_effort": "minimal",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Request: {prompt}\n\nText A: {a}\n\nText B: {b}"}]}
    for attempt in range(6):
        try:
            req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                                  "User-Agent": "Mozilla/5.0 (research script)"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.loads(r.read().decode())
            txt = j["choices"][0]["message"]["content"].strip()
            m = re.fullmatch(r"\D*?(\d{1,2})\D*", txt)
            u = j.get("usage", {})
            if m and 0 <= int(m.group(1)) <= 10:
                return int(m.group(1)), u.get("prompt_tokens", 0), u.get("completion_tokens", 0), j.get("model")
        except Exception as e:
            time.sleep(1.5 * (attempt + 1))
    return None, 0, 0, None


t0 = time.time()
with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(lambda q: call(texts[q["a"]], texts[q["b"]]), R["raw"]))
n = R["n"]
D = np.zeros((n, n)); fails = 0
for q, r in zip(R["raw"], res):
    if r[0] is None:
        fails += 1; D[q["a"], q["b"]] = D[q["b"], q["a"]] = np.nan
    else:
        D[q["a"], q["b"]] = D[q["b"], q["a"]] = r[0] / 10.0
out = {"name": R["name"], "n": n, "pairs": len(R["raw"]), "calls": len(R["raw"]), "model": MODEL,
       "model_resolved": next((r[3] for r in res if r[3]), None), "reasoning_effort": "minimal",
       "wall_seconds": time.time() - t0, "workers": 32, "failures": fails,
       "input_tokens": int(sum(r[1] for r in res)), "output_tokens": int(sum(r[2] for r in res)),
       "latency_s_median": float("nan"), "texts": texts, "D": D.tolist(),
       "raw": [{"a": q["a"], "b": q["b"], "score": r[0]} for q, r in zip(R["raw"], res)]}
json.dump(out, open(out_path, "w"))
iu = np.triu_indices(n, 1)
print(f"done {len(res)} calls in {time.time()-t0:.0f}s, failures {fails}, model {out['model_resolved']}, "
      f"mu={np.nanmean(D[iu]):.4f}, tokens {out['input_tokens']}/{out['output_tokens']}")
