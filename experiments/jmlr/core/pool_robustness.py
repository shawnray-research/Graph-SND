"""Robustness of the anatomy to the observation sample: recompute D for saved final
policies from independent rollouts/pools and compare level and shares."""
import sys, glob, os, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "repo")); sys.path.insert(0, str(HERE))
import training.train_navigation_batched as T
from graphsnd.batched_policies import BatchedGaussianMLPPolicy, BatchedValueMLP
from graphsnd.policies import PolicyConfig
from run_e3 import behavioral_D, pool_from_buffers
from anatomy import anatomy
dev = torch.device("cuda:0")
out = []
for pt in sys.argv[1:]:
    name = os.path.basename(pt).replace("_policies.pt", "")
    task, ns, ss = name.split("_"); n = int(ns[1:]); seed = int(ss[1:])
    ck = torch.load(pt, map_location=dev)
    pcfg = PolicyConfig(**{k: (tuple(v) if k == "hidden_sizes" else v) for k, v in ck["pcfg"].items()})
    pol = BatchedGaussianMLPPolicy(n, pcfg).to(dev); pol.load_state_dict(ck["policy"])
    res = []
    for rep in range(3):
        cfg = T.PPOConfig(rollout_steps=64, num_envs=128)
        env = T.build_env(task, cfg.num_envs, n, cfg.max_steps, "cuda:0", 1000 + 17 * rep + seed, T.scenario_kwargs_for(n))
        val = BatchedValueMLP(n, pcfg.obs_dim).to(dev)
        buf = T.allocate_buffers(cfg.rollout_steps, n, cfg.num_envs, pcfg.obs_dim, pcfg.act_dim, dev)
        T.collect_rollout_batched(env, pol, val, buf, cfg.rollout_steps, n, dev)
        D = behavioral_D(pol, pool_from_buffers(buf, 4096, torch.Generator().manual_seed(rep)))
        an = anatomy(D)
        res.append((an["mu"], an["s2_main"] / an["s2_tot"], an["kappa"]))
    r = np.array(res)
    out.append(dict(run=name, mu=r[:, 0].tolist(), distinct=r[:, 1].tolist()))
    print(f"{name}: mu {r[:,0].mean():.4f} (rel spread {100*(r[:,0].max()-r[:,0].min())/r[:,0].mean():.2f}%) | distinct share {r[:,1].mean():.3f} [{r[:,1].min():.3f}, {r[:,1].max():.3f}]", flush=True)
json.dump(out, open(HERE / "pool_robustness.json", "w"), indent=1)
