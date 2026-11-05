"""Wall-clock cost of all-pairs SND vs balanced designs, n up to 10^4.

Policies: n independent 2-hidden-layer tanh MLPs (64 units) with Gaussian
heads (state-independent std), evaluated on a shared batch of B
observations (as in SND). Behavioral distance: mean over observations of
W2 between diagonal Gaussians. All-pairs cost is measured exactly up to
n_full_max and extrapolated beyond with a fitted quadratic only for the
annotation (never plotted as measured).
Compatible with python3.6 / torch 1.10.
"""
import json
import math
import sys
import time

import torch

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
out = sys.argv[2] if len(sys.argv) > 2 else "timing_scale.json"
torch.manual_seed(0)
OBS, ACT, H, B = 18, 2, 64, 1024


def make_policies(n):
    W1 = torch.randn(n, OBS, H, device=dev) / math.sqrt(OBS)
    W2 = torch.randn(n, H, H, device=dev) / math.sqrt(H)
    W3 = torch.randn(n, H, ACT, device=dev) / math.sqrt(H)
    ls = 0.1 * torch.randn(n, 1, ACT, device=dev) - 0.5
    return W1, W2, W3, ls


def forward(pol, obs):
    W1, W2, W3, ls = pol
    h = torch.tanh(torch.einsum("bo,noh->nbh", obs, W1))
    h = torch.tanh(torch.bmm(h, W2))
    mu = torch.bmm(h, W3)                      # n,B,A
    std = ls.exp().expand_as(mu)
    return mu, std


def pair_dist(mu, std, I, J, chunk=4096):
    out = torch.empty(I.numel(), device=dev)
    for s in range(0, I.numel(), chunk):
        i, j = I[s:s + chunk], J[s:s + chunk]
        dm = (mu[i] - mu[j]).pow(2).sum(-1)
        ds = (std[i] - std[j]).pow(2).sum(-1)
        out[s:s + chunk] = (dm + ds).sqrt().mean(-1)
    return out


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def time_call(fn, reps):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


rows = []
obs = torch.randn(B, OBS, device=dev)
for n in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    pol = make_policies(n)

    def full():
        mu, std = forward(pol, obs)
        I, J = torch.triu_indices(n, n, 1, device=dev)
        return pair_dist(mu, std, I, J).mean()

    def balanced(k):
        def f():
            mu, std = forward(pol, obs)
            perm = torch.randperm(n, device=dev)
            t = torch.arange(n, device=dev)
            I = torch.cat([perm[t] for s in range(1, k // 2 + 1)])
            J = torch.cat([perm[(t + s) % n] for s in range(1, k // 2 + 1)])
            return pair_dist(mu, std, I, J).mean()
        return f

    row = {"n": n, "pairs_full": n * (n - 1) // 2}
    if n <= 5000:
        row["t_full_s"] = time_call(full, 3 if n >= 2000 else 7)
        row["snd_full"] = float(full())
    for k in [4, 8, 16]:
        row["t_bal_k%d_s" % k] = time_call(balanced(k), 7)
        vals = torch.stack([balanced(k)() for _ in range(50)])
        row["bal_k%d_mean" % k] = float(vals.mean())
        row["bal_k%d_sd" % k] = float(vals.std())
    rows.append(row)
    print(json.dumps(row), flush=True)
    del pol
    torch.cuda.empty_cache()
json.dump(rows, open(out, "w"), indent=1)
