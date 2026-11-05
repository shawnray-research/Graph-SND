"""E7 curve: 1,000 story responses per temperature (Qwen2.5-0.5B-Instruct, pure temperature sampling)."""
import json, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id, out_path = sys.argv[1], sys.argv[2]
N, B, MAXTOK = 1000, 250, 40
TEMPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
prompt = "Write a one-sentence story about a robot."
tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).cuda().eval()
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
enc = tok([text] * B, return_tensors="pt", padding=True).to("cuda")
res = {"model": model_id, "prompt": prompt, "max_new_tokens": MAXTOK, "populations": []}
for ti, T in enumerate(TEMPS):
    torch.manual_seed(5000 + ti); outs = []; t0 = time.time()
    while len(outs) < N:
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0, max_new_tokens=MAXTOK,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += [s.strip() for s in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    res["populations"].append({"temperature": T, "texts": outs[:N]})
    print(T, f"{time.time()-t0:.1f}s", repr(outs[0][:70]), flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("saved")
