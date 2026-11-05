"""Build the second batch of text populations: Llama populations, a cross-model population,
and a contaminated population (2% degenerate high-temperature samples among coherent ones)."""
import json, sys
q = json.load(open("gen_qwen.json"))["populations"]
l = json.load(open("gen_llama.json"))["populations"]
Q = {f"{p['prompt_id']}_T{p['temperature']}": p["texts"] for p in q}
L = {f"{p['prompt_id']}_T{p['temperature']}": p["texts"] for p in l}
pops = [{"name": f"llama_{k}", "texts": v, "description": "Llama-3.2-1B-Instruct"} for k, v in L.items()]
pops.append({"name": "crossmodel_story_T1.0", "texts": Q["story_T1.0"][:150] + L["story_T1.0"][:150],
             "description": "150 Qwen + 150 Llama story samples at T=1.0"})
pops.append({"name": "contaminated_story_T0.5", "texts": Q["story_T0.5"][:294] + Q["story_T1.5"][:6],
             "description": "294 Qwen story samples at T=0.5 + 6 at T=1.5 (2% degenerate)"})
json.dump(pops, open(sys.argv[1], "w"), indent=1)
print(len(pops), [p["name"] for p in pops])
