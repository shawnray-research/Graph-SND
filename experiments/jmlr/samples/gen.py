"""E7 step 1: sample populations of short LM responses (pure temperature sampling, no top-k/top-p)."""
import os, sys, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = sys.argv[1]
out_path = sys.argv[2]
N = int(os.environ.get("N", 300))
B = int(os.environ.get("B", 150))
MAXTOK = int(os.environ.get("MAXTOK", 40))
PROMPTS = {
    "story": "Write a one-sentence story about a robot.",
    "cafe": "Invent a name for a new coffee shop and explain it in one sentence.",
    "advice": "Give one sentence of advice to someone starting their first job.",
}
TEMPS = [0.5, 1.0, 1.5]

tok = AutoTokenizer.from_pretrained(model_id)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
res = {"model": model_id, "max_new_tokens": MAXTOK, "populations": []}
for pi, (pname, ptext) in enumerate(PROMPTS.items()):
    text = tok.apply_chat_template([{"role": "user", "content": ptext}], tokenize=False,
                                   add_generation_prompt=True)
    enc = tok([text] * B, return_tensors="pt", padding=True).to("cuda")
    for ti, T in enumerate(TEMPS):
        torch.manual_seed(1000 * pi + ti)
        outs, t0 = [], time.time()
        while len(outs) < N:
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0,
                                   max_new_tokens=MAXTOK, pad_token_id=tok.pad_token_id)
            outs += [s.strip() for s in tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                                          skip_special_tokens=True)]
        outs = outs[:N]
        res["populations"].append({"prompt_id": pname, "prompt": ptext, "temperature": T,
                                   "texts": outs, "gen_seconds": time.time() - t0})
        print(pname, T, f"{time.time()-t0:.1f}s", repr(outs[0][:80]), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("saved", out_path)
