"""E3: train heterogeneous IPPO teams on VMAS and dump full behavioral distance
matrices D (SND definition: all policies evaluated on a shared observation pool)
at a schedule of iterations. Also dumps semi-synthetic variants built from real
policies: (a) one-outlier team (one trained agent reset to its init weights),
(b) two-group team (half trained, half init).

--critic central replaces the independent per-agent critics by a centralized critic
(MAPPO-style: each agent's value head sees the concatenated observations of all agents;
actors stay decentralized and are trained by the same PPO update)."""
import argparse, sys, time, json, copy
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent / "repo"
sys.path.insert(0, str(REPO))
import training.train_navigation_batched as T
from graphsnd.batched_policies import BatchedGaussianMLPPolicy, BatchedValueMLP
from graphsnd.policies import PolicyConfig


class CentralValue(torch.nn.Module):
    """Centralized critic: (n, B, d) -> (n, B); agent i's head sees all n observations."""

    def __init__(self, n_agents, obs_dim):
        super().__init__()
        self.n = n_agents
        self.inner = BatchedValueMLP(n_agents, n_agents * obs_dim)

    def forward(self, obs):
        n, B, d = obs.shape
        joint = obs.permute(1, 0, 2).reshape(1, B, n * d).expand(n, B, n * d)
        return self.inner(joint)


@torch.no_grad()
def behavioral_D(policy, obs_pool):
    n = policy.n_agents
    obs = obs_pool.unsqueeze(0).expand(n, -1, -1).contiguous()
    means, stds = policy(obs)  # (n, P, a)
    D = torch.zeros(n, n, dtype=torch.float64, device=means.device)
    for i in range(n):
        dm = means[i:i + 1] - means
        ds = stds[i:i + 1] - stds
        w = torch.sqrt((dm ** 2).sum(-1) + (ds ** 2).sum(-1))  # (n, P)
        D[i] = w.double().mean(-1)
    D = 0.5 * (D + D.T)
    D.fill_diagonal_(0.0)
    return D.cpu().numpy()


def pool_from_buffers(buffers, size, gen):
    flat = buffers.obs.reshape(-1, buffers.obs.shape[-1])
    idx = torch.randint(0, flat.shape[0], (size,), generator=gen, device="cpu").to(flat.device)
    return flat[idx].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", type=str, default="navigation")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--save-at", type=str, default="0,10,25,50,100,150,200,250,300")
    ap.add_argument("--out", type=str, default="matrices")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-minutes", type=float, default=14.0)
    ap.add_argument("--critic", type=str, default="independent", choices=["independent", "central"])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_at = sorted(set(int(x) for x in args.save_at.split(",")))
    device = torch.device(args.device)
    scen = T.scenario_kwargs_for(args.n_agents)
    cfg = T.PPOConfig(rollout_steps=args.rollout_steps, num_envs=args.num_envs)
    T.set_seeds(args.seed)
    env = T.build_env(args.scenario, cfg.num_envs, args.n_agents, cfg.max_steps,
                      args.device, args.seed, scen)
    ex = env.reset()
    obs_dim = int(ex[0].shape[-1]); act_dim = int(env.get_agent_action_size(env.agents[0]))
    u_range = float(env.agents[0].u_range)
    pcfg = PolicyConfig(obs_dim=obs_dim, act_dim=act_dim, u_range=u_range)
    policy = BatchedGaussianMLPPolicy(args.n_agents, pcfg, seed_base=args.seed * 1000).to(device)
    init_policy = copy.deepcopy(policy)
    value = (CentralValue(args.n_agents, obs_dim) if args.critic == "central"
             else BatchedValueMLP(args.n_agents, obs_dim)).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=cfg.lr)
    buffers = T.allocate_buffers(cfg.rollout_steps, args.n_agents, cfg.num_envs, obs_dim, act_dim, device)
    gen = torch.Generator().manual_seed(12345 + args.seed)
    tag = f"{args.scenario}_n{args.n_agents}_s{args.seed}" + ("_mappo" if args.critic == "central" else "")
    log = []
    t0 = time.time()

    def dump(it, rew, pol=policy, kind="trained"):
        pool = pool_from_buffers(buffers, args.pool, gen)
        D = behavioral_D(pol, pool)
        fn = out / f"{tag}_{kind}_it{it}.npz"
        np.savez(fn, D=D, n=args.n_agents, task=args.scenario, iter=it, seed=args.seed,
                 reward=rew, kind=kind)
        iu = np.triu_indices(args.n_agents, 1)
        print(f"[{tag}] it={it} kind={kind} SND={D[iu].mean():.5f} reward={rew:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return D

    last_it = 0
    for it in range(0, args.iters + 1):
        boot, rew, neps = T.collect_rollout_batched(env, policy, value, buffers, cfg.rollout_steps,
                                                    args.n_agents, device)
        if it in save_at:
            dump(it, rew)
            log.append({"iter": it, "reward": rew, "elapsed": time.time() - t0})
        if it == args.iters or (time.time() - t0) / 60 > args.max_minutes:
            last_it = it
            if it not in save_at:
                dump(it, rew)
                log.append({"iter": it, "reward": rew, "elapsed": time.time() - t0})
            break
        T.ppo_update_batched(policy, value, opt, buffers, boot, cfg)
        last_it = it

    # Semi-synthetic variants built from real policies (same final obs pool regime)
    n = args.n_agents
    # (a) one outlier: agent 0 reset to its initial weights
    outl = copy.deepcopy(policy)
    with torch.no_grad():
        for name, p in outl.named_parameters():
            p0 = dict(init_policy.named_parameters())[name]
            p[0].copy_(p0[0])
    dump(last_it, rew, outl, kind="outlier1")
    # (b) two groups: first half trained, second half reset to init
    grp = copy.deepcopy(policy)
    with torch.no_grad():
        for name, p in grp.named_parameters():
            p0 = dict(init_policy.named_parameters())[name]
            p[n // 2:].copy_(p0[n // 2:])
    dump(last_it, rew, grp, kind="twogroup")
    torch.save({"policy": policy.state_dict(), "init": init_policy.state_dict(), "pcfg": pcfg.__dict__},
               out / f"{tag}_policies.pt")
    (out / f"{tag}_log.json").write_text(json.dumps(log, indent=1))
    print(f"[{tag}] DONE last_it={last_it} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
