"""E7 (sample sets at scale): anatomy, comparison designs and the order-2 Vendi score on
Fashion-MNIST sample sets in a learned feature space.

Population = a set of images; d_ij = Euclidean distance between penultimate-layer features
of a small MLP classifier trained on the training split. Exact all-pairs quantities are
computed in GPU chunks; every design is simulated by Monte Carlo on the fixed table.
Order-2 Vendi score / RKE with a Gaussian kernel: VS2 = n / (1 + (n-1) * mean_{i<j} k_ij^2).
"""
import gzip, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

out_path = sys.argv[1]
root = sys.argv[2]
R = int(os.environ.get("REPS", 1000))
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False


def idx(fn, off):
    with gzip.open(os.path.join(root, fn)) as f:
        return np.frombuffer(f.read(), np.uint8, offset=off)


Xtr = idx("train-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255.0
ytr = idx("train-labels-idx1-ubyte.gz", 8).astype(np.int64)
Xte = idx("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784).astype(np.float32) / 255.0
yte = idx("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)

# ---- feature extractor: 784-256-128-10 MLP, 5 epochs, Adam
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128),
                          torch.nn.ReLU(), torch.nn.Linear(128, 10)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
for ep in range(5):
    perm = torch.randperm(len(Xt), device=dev)
    for s in range(0, len(Xt), 256):
        b = perm[s:s + 256]
        loss = F.cross_entropy(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
torch.set_grad_enabled(False)
net.eval()
acc = (net(torch.tensor(Xte, device=dev)).argmax(1).cpu().numpy() == yte).mean()
feat = lambda X: net[:4](torch.tensor(X, device=dev)).double().detach()
print(f"feature net test acc {acc:.4f}", flush=True)

rng = np.random.default_rng(0)
populations = {}
populations["test_n10000"] = Xte
populations["train_n60000"] = Xtr
# 1% corrupted samples (uniform-noise images): a generator that occasionally emits garbage
Xo = Xte.copy(); bad = rng.choice(len(Xo), size=len(Xo) // 100, replace=False)
Xo[bad] = rng.random((len(bad), 784)).astype(np.float32)
populations["test_1pct_noise"] = Xo
# partial mode collapse: 30% of samples are near-copies of 10 images
Xc = Xte.copy(); col = rng.choice(len(Xc), size=3 * len(Xc) // 10, replace=False)
proto = Xte[rng.choice(len(Xte), size=10, replace=False)]
Xc[col] = np.clip(proto[rng.integers(0, 10, len(col))] + 0.05 * rng.standard_normal((len(col), 784)), 0, 1).astype(np.float32)
populations["test_30pct_collapse"] = Xc


def pair_dist(A, B):
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return sq.clamp_min(0).sqrt()


def exact_stats(Fz, h):
    """Exact all-pairs statistics of d = ||f_i - f_j|| and t = exp(-d^2/h^2) (= k^2, k Gaussian, bw h)."""
    n = Fz.shape[0]
    N = n * (n - 1) / 2
    rs_d = torch.zeros(n, dtype=torch.float64, device=dev)
    rs_t = torch.zeros(n, dtype=torch.float64, device=dev)
    s2_d = 0.0; s2_t = 0.0
    CH = 2048
    for s in range(0, n, CH):
        Dd = pair_dist(Fz[s:s + CH], Fz)
        Dd[torch.arange(Dd.shape[0]), torch.arange(s, s + Dd.shape[0])] = 0.0
        Tt = torch.exp(-(Dd ** 2) / h ** 2)
        Tt[torch.arange(Tt.shape[0]), torch.arange(s, s + Tt.shape[0])] = 0.0
        rs_d[s:s + CH] = Dd.sum(1); rs_t[s:s + CH] = Tt.sum(1)
        s2_d += (Dd ** 2).sum().item(); s2_t += (Tt ** 2).sum().item()
    out = {}
    for key, rs, s2 in (("d", rs_d, s2_d), ("t", rs_t, s2_t)):
        mu = rs.sum().item() / (2 * N)
        a = (n - 1) * (rs / (n - 1) - mu) / (n - 2)
        s2_tot = s2 / (2 * N) - mu ** 2
        s2_main = (n - 2) * (a ** 2).sum().item() / N
        out[key] = dict(mu=mu, s2_tot=s2_tot, s2_main=s2_main, s2_int=s2_tot - s2_main,
                        sum_sq=s2 / 2, a=a)
    return out


def circ_edges(n, k):
    E = []
    for o in range(1, k // 2 + 1):
        i = np.arange(n); E.append(np.c_[i, (i + o) % n])
    if k % 2 == 1:
        i = np.arange(n // 2); E.append(np.c_[i, i + n // 2])
    return torch.tensor(np.concatenate(E), device=dev)


def evaluate(Fz, I, J, h):
    d = (Fz[I] - Fz[J]).norm(dim=1)
    return d, torch.exp(-(d ** 2) / h ** 2)


def simulate(Fz, h, k, n, N, g):
    """R draws of each design at budget m = n k / 2; returns dict design -> (est_d, est_t) arrays."""
    m = n * k // 2
    E = circ_edges(n, k)
    res = {key: ([], []) for key in ("balanced", "uniform", "bernoulli_ht", "subset")}
    s = int(round((1 + np.sqrt(1 + 8 * m)) / 2))
    p = m / N
    for r in range(R):
        perm = torch.randperm(n, device=dev, generator=g)
        P = perm[E]
        d, t = evaluate(Fz, P[:, 0], P[:, 1], h)
        res["balanced"][0].append(d.mean().item()); res["balanced"][1].append(t.mean().item())
        I = torch.randint(0, n, (m,), device=dev, generator=g)
        J = (I + torch.randint(1, n, (m,), device=dev, generator=g)) % n
        d, t = evaluate(Fz, I, J, h)
        res["uniform"][0].append(d.mean().item()); res["uniform"][1].append(t.mean().item())
        M = int(np.random.default_rng(r).binomial(int(N), p))
        I = torch.randint(0, n, (M,), device=dev, generator=g)
        J = (I + torch.randint(1, n, (M,), device=dev, generator=g)) % n
        d, t = evaluate(Fz, I, J, h)
        res["bernoulli_ht"][0].append(d.sum().item() / (p * N)); res["bernoulli_ht"][1].append(t.sum().item() / (p * N))
        S = torch.randperm(n, device=dev, generator=g)[:s]
        iu = torch.triu_indices(s, s, 1, device=dev)
        d, t = evaluate(Fz, S[iu[0]], S[iu[1]], h)
        res["subset"][0].append(d.mean().item()); res["subset"][1].append(t.mean().item())
    return {kk: (np.array(v[0]), np.array(v[1])) for kk, v in res.items()}, m, s * (s - 1) // 2


results = {"feature_net_test_acc": float(acc), "reps": R, "populations": []}
g = torch.Generator(device=dev); g.manual_seed(0)
for name, X in populations.items():
    t0 = time.time()
    Fz = feat(X)
    n = Fz.shape[0]; N = n * (n - 1) / 2
    # bandwidth: median pairwise distance over 200k random pairs (fixed once per population)
    I = torch.randint(0, n, (200000,), device=dev, generator=g)
    J = (I + torch.randint(1, n, (200000,), device=dev, generator=g)) % n
    h = (Fz[I] - Fz[J]).norm(dim=1).median().item()
    torch.cuda.synchronize(); te = time.time()
    ex = exact_stats(Fz, h)
    torch.cuda.synchronize(); t_exact = time.time() - te
    vs2 = n / (1 + (n - 1) * ex["t"]["mu"])
    a = ex["d"]["a"].cpu().numpy()
    pop = {"name": name, "n": n, "pairs": N, "bandwidth": h, "exact_seconds_gpu": t_exact,
           "mu": ex["d"]["mu"], "cv": float(np.sqrt(ex["d"]["s2_tot"]) / ex["d"]["mu"]),
           "s2_tot": ex["d"]["s2_tot"], "s2_main": ex["d"]["s2_main"], "s2_int": ex["d"]["s2_int"],
           "distinct_share": ex["d"]["s2_main"] / ex["d"]["s2_tot"],
           "kappa": ex["d"]["s2_main"] / ex["d"]["s2_int"],
           "vs2": vs2, "t_mu": ex["t"]["mu"], "t_s2_tot": ex["t"]["s2_tot"], "t_s2_int": ex["t"]["s2_int"],
           "t_distinct_share": ex["t"]["s2_main"] / ex["t"]["s2_tot"],
           "a_top_share_1pct": float(np.sort(a ** 2)[::-1][: max(1, n // 100)].sum() / (a ** 2).sum()),
           "budgets": []}
    if name == "test_1pct_noise":
        ai = np.argsort(-a); pop["noise_in_top1pct_by_a"] = float(np.isin(ai[: n // 100], bad).mean())
    if name == "test_30pct_collapse":
        pop["collapsed_mean_a"] = float(a[col].mean()); pop["other_mean_a"] = float(np.delete(a, col).mean())
    for k in (1, 2, 4, 8):
        sims, m, m_sub = simulate(Fz, h, k, n, N, g)
        p = m / N
        pred = {"balanced": ex["d"]["s2_int"] / m * (n - 1 - k) / (n - 3),
                "uniform": ex["d"]["s2_tot"] / m,  # with-replacement pairs (N >> m)
                "bernoulli_ht": (1 - p) / p * ex["d"]["sum_sq"] / N ** 2,
                "subset": None}
        row = {"k": k, "m": m, "m_subset": m_sub, "frac_pairs": m / N}
        for dz, (est_d, est_t) in sims.items():
            mu, tmu = ex["d"]["mu"], ex["t"]["mu"]
            vs_hat = n / (1 + (n - 1) * est_t)
            row[dz] = {"rel_rmse": float(np.sqrt(np.mean((est_d / mu - 1) ** 2))),
                       "rel_bias": float(np.mean(est_d) / mu - 1),
                       "rel_sd": float(np.std(est_d, ddof=1) / mu),
                       "rel_sd_pred": None if pred[dz] is None else float(np.sqrt(pred[dz]) / mu),
                       "vs2_rel_rmse": float(np.sqrt(np.mean((vs_hat / vs2 - 1) ** 2))),
                       "vs2_rel_bias": float(np.mean(vs_hat) / vs2 - 1)}
        # delta-method prediction for the VS2 estimate under the balanced design
        tv = ex["t"]["s2_int"] / m * (n - 1 - k) / (n - 3)
        row["balanced"]["vs2_rel_sd_pred"] = float((n - 1) * np.sqrt(tv) / (1 + (n - 1) * ex["t"]["mu"]))
        tu = ex["t"]["s2_tot"] / m
        row["uniform"]["vs2_rel_sd_pred"] = float((n - 1) * np.sqrt(tu) / (1 + (n - 1) * ex["t"]["mu"]))
        pop["budgets"].append(row)
        print(name, k, {dz: round(100 * row[dz]["rel_rmse"], 3) for dz in sims},
              "pred bal %.3f unif %.3f" % (100 * row["balanced"]["rel_sd_pred"], 100 * row["uniform"]["rel_sd_pred"]),
              "VS2 bal %.3f unif %.3f" % (100 * row["balanced"]["vs2_rel_rmse"], 100 * row["uniform"]["vs2_rel_rmse"]), flush=True)
    # timing: one balanced k=4 evaluation vs one exact all-pairs pass
    E4 = circ_edges(n, 4); torch.cuda.synchronize(); tb = time.time()
    for _ in range(20):
        P = torch.randperm(n, device=dev, generator=g)[E4]; d, t = evaluate(Fz, P[:, 0], P[:, 1], h); d.mean().item()
    torch.cuda.synchronize(); pop["balanced_k4_seconds_gpu"] = (time.time() - tb) / 20
    pop["total_seconds"] = time.time() - t0
    print(name, "n", n, "mu %.4f cv %.3f distinct %.3f kappa %.3f VS2 %.2f exact %.2fs bal %.4fs" % (
        pop["mu"], pop["cv"], pop["distinct_share"], pop["kappa"], vs2, t_exact, pop["balanced_k4_seconds_gpu"]), flush=True)
    results["populations"].append(pop)
    json.dump(results, open(out_path, "w"), indent=1)
print("saved", out_path)
