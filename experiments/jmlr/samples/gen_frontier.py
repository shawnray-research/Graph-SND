"""E7: 300 responses per prompt from a frontier model (Claude Opus 5.5, default sampling; the current
API exposes no temperature), for the same three prompts as gen.py."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
out_path = sys.argv[1]
N, WORKERS = 300, 40
PROMPTS = {
    "story": "Write a one-sentence story about a robot.",
    "cafe": "Invent a name for a new coffee shop and explain it in one sentence.",
    "advice": "Give one sentence of advice to someone starting their first job.",
}
client = anthropic.Anthropic(max_retries=8)  # reads ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL if set)


def gen(prompt):
    r = client.messages.create(model="claude-opus-5-5", max_tokens=2000, output_config={"effort": "low"},
                               messages=[{"role": "user", "content": prompt}])
    txt = "".join(getattr(c, "text", "") for c in r.content if c.type == "text").strip()
    return txt, r.usage.input_tokens, r.usage.output_tokens


res = {"model": "claude-opus-5-5", "effort": "low", "populations": []}
t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    for pid, p in PROMPTS.items():
        outs = list(ex.map(gen, [p] * N))
        res["populations"].append({"prompt_id": pid, "prompt": p, "temperature": "default",
                                   "texts": [o[0] for o in outs],
                                   "input_tokens": sum(o[1] for o in outs), "output_tokens": sum(o[2] for o in outs)})
        print(pid, f"{time.time()-t0:.0f}s", repr(outs[0][0][:90]), repr(outs[1][0][:90]), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("saved")
