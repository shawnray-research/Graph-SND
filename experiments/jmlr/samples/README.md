# E7: generated samples, neural judges and sample sets at scale

Scripts behind `sec:exp-judge` (generated text scored by a neural judge) and the Fashion-MNIST
sample-set paragraph. Estimator code and exact laws are those of `../core/anatomy.py`.

| File | Purpose |
|---|---|
| `gen.py` | Samples 300 responses per (prompt, temperature) from an instruction-tuned LM (pure temperature sampling, no top-k/top-p, 40 new tokens) |
| `judge.py`, `judge2.py` | All-pairs judge tables with the STS cross-encoder `cross-encoder/stsb-roberta-base`: one forward pass per ordered pair, `d_ij = 1 - (s(i,j) + s(j,i))/2`, `s = sigmoid(logit)` |
| `gen_frontier.py` | 300 responses per prompt from Claude Opus 5.5 (effort `low`, default sampling); judged with `judge2.py`, analysed into `analysis_opus.json` |
| `build_pops.py` | Second batch: Llama populations, a cross-model population, a 2%-degenerate population |
| `claude_judge.py` | All 2,016 pairs of 64 responses scored by Claude Opus 5.5 (effort `low`), integer 0-10, `d = score/10`, one call per pair with random A/B order; 200-pair order-swapped retest |
| `gpt_judge.py` | The same 2,016 Claude-scored pairs (same order) scored by GPT-5 (`gpt-5-2025-08-07`, reasoning effort minimal) via the OpenAI Chat Completions API |
| `claude_to_tables.py`, `claude_extra.py` | Converts the Claude scores to tables; retest reliability, agreement with the cross-encoder, API cost |
| `samples_judge_analysis.py` | Anatomy, four designs at `m = n/2, n, 2n, 4n` (4,000 draws each), measured vs closed-form s.d., round-robin 95% intervals, distinctiveness recovery, order-2 Vendi score with the judge kernel |
| `samples_fmnist.py` | Fashion-MNIST sample sets (n = 10,000 and 60,000) in a learned 128-d feature space: exact all-pairs statistics on GPU, designs (1,000 draws), order-2 Vendi score with a Gaussian kernel |
| `gen_curve.py`, `claude_curve.py` | Temperature curve: 1,000 story responses at each of six temperatures; one randomized round-robin design (k = 4 rounds, 2,000 Claude calls) per temperature, with the design-based 95% interval of Prop. rr-variance and sparse distinctiveness estimates; T = 1 repeated with an independent design |
| `judge_numbers.py` | Collects every number quoted in Sec. 5.6 into `judge_numbers.json` |
| `fig_judge.py` (`fig_curve.py`: standalone version of panel (c)) | `jmlr/figures/fig_judge.pdf` |

## Reproduce

```bash
export HF_HOME=$PWD/hf
CUDA_VISIBLE_DEVICES=0 python gen.py Qwen/Qwen2.5-0.5B-Instruct gen_qwen.json
CUDA_VISIBLE_DEVICES=0 python gen.py unsloth/Llama-3.2-1B-Instruct gen_llama.json
python judge.py gen_qwen.json judged_qwen.npz                    # ~11 GPU-min (10 populations)
python build_pops.py pops2.json && python judge2.py pops2.json judged2.npz
python claude_judge.py gen_qwen.json claude                       # needs ANTHROPIC_API_KEY
python claude_to_tables.py judged_claude.npz claude/claude_story_T1.0.json claude/claude_story_T0.5.json
REPS=4000 python samples_judge_analysis.py judged_qwen.npz judged_qwen_meta.json analysis_qwen.json
REPS=4000 python samples_judge_analysis.py judged2.npz judged2_meta.json analysis_judge2.json
REPS=4000 python samples_judge_analysis.py judged_claude.npz judged_claude_meta.json analysis_claude.json
python claude_extra.py                                            # run where claude_*.json and judged_qwen.npz are
REPS=1000 python samples_fmnist.py samples_fmnist.json /path/to/fashion-mnist   # raw idx .gz files
python judge_numbers.py                                           # run where the analysis JSONs are
python gen_curve.py Qwen/Qwen2.5-0.5B-Instruct gen_curve.json
python claude_curve.py gen_curve.json claude_curve.json           # 14,000 calls, about $29
python fig_judge.py ../../../jmlr/figures/fig_judge analysis_qwen.json analysis_judge2.json analysis_claude.json --curve claude_curve.json   # panel (c) = temperature curve
python fig_curve.py claude_curve.json fig_curve analysis_qwen.json   # optional standalone curve (not in the paper)
```

Outputs are in `results/jmlr/samples/`. Software: Python 3.11, torch 2.8.0+cu128, transformers 4.57.1,
NumPy, SciPy 1.17; GPU: one NVIDIA Quadro RTX 5000 (16 GB) per job.
