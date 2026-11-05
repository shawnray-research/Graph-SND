"""Accuracy of the balanced design at scale: measured vs closed-form sd.

Same policy population as timing_scale.py (seed 0, B=1024 observations).
For each n, computes the full distance table on the GPU, its anatomy, the
closed-form relative sd of Theorem 5, and the measured relative sd over
`reps` independent permuted-circulant designs.
"""
import json
import math
import sys

import torch

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
out = sys.argv[2] if len(sys.argv) > 2 else "accuracy_scale.json"
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
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
    mu = torch.bmm(h, W3)
    return mu, ls.exp().expand_as(mu)


obs = torch.randn(B, OBS, device=dev)
rows = []
for n in [50, 100, 200, 500, 1000, 2000, 5000]:
    pol = make_policies(n)
    mu_, std_ = forward(pol, obs)
    D = torch.zeros(n, n, device=dev, dtype=torch.float64)
    ch = max(1, int(2.5e8 // (n * B * ACT * 4)))
    for i0 in range(0, n, ch):
        i1 = min(n, i0 + ch)
        dm = (mu_[i0:i1, None] - mu_[None]).pow(2).sum(-1)
        ds = (std_[i0:i1, None] - std_[None]).pow(2).sum(-1)
        D[i0:i1] = (dm + ds).sqrt().mean(-1).double()
    D.fill_diagonal_(0)
    N = n * (n - 1) / 2
    iu = torch.triu_indices(n, n, 1, device=dev)
    mu = D[iu[0], iu[1]].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]
    R.fill_diagonal_(0)
    s_int = (R[iu[0], iu[1]] ** 2).mean()
    row = {"n": n, "mu": float(mu), "sigma_int_over_mu": float(s_int.sqrt() / mu)}
    t = torch.arange(n, device=dev)
    for k in [4, 8, 16]:
        m = n * k / 2
        pred = math.sqrt(float(s_int) / m * (n - 1 - k) / (n - 3)) / float(mu)
        ests = torch.empty(reps, dtype=torch.float64, device=dev)
        for r in range(reps):
            p = torch.randperm(n, device=dev)
            I = torch.cat([p[t] for s in range(1, k // 2 + 1)])
            J = torch.cat([p[(t + s) % n] for s in range(1, k // 2 + 1)])
            ests[r] = D[I, J].mean()
        rel = ests / mu - 1
        sd = float(rel.std())
        row["k%d_pred" % k] = pred
        row["k%d_meas" % k] = sd
        row["k%d_meas_se" % k] = sd / math.sqrt(2 * (reps - 1))
        # sqrt(v_{n,k}) with the tight bound sigma2_int <= (n-2)/n mu^2 (Theorem thm:metric)
        row["k%d_bound" % k] = math.sqrt(2 * (n - 2) * (n - 1 - k) / (n * n * (n - 3) * k))
    rows.append(row)
    print(json.dumps(row), flush=True)
    del D, R
    torch.cuda.empty_cache()
json.dump(rows, open(out, "w"), indent=1)
