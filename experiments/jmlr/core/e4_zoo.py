"""E4: model-zoo diversity. 128 MLP classifiers on Fashion-MNIST in 4 training recipes
(32 seeds each). Dissimilarity = mean over test inputs of the Jensen-Shannon distance
(sqrt of base-2 JSD) between predictive distributions (non-separable)."""
import gzip, os, sys, time, urllib.request
import numpy as np
import torch, torch.nn.functional as F

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fmnist")
os.makedirs(root, exist_ok=True)
BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
for fn in FILES:
    p = os.path.join(root, fn)
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + fn, p)

def idx(fn, off):
    with gzip.open(os.path.join(root, fn)) as f:
        return np.frombuffer(f.read(), np.uint8, offset=off)

Xtr = idx(FILES[0], 16).reshape(-1, 784).astype(np.float32) / 255.0
ytr = idx(FILES[1], 8).astype(np.int64)
Xte = idx(FILES[2], 16).reshape(-1, 784).astype(np.float32) / 255.0
yte = idx(FILES[3], 8).astype(np.int64)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Xtr_t, ytr_t = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
Xte_t = torch.tensor(Xte[:2000], device=dev); yte_t = torch.tensor(yte[:2000], device=dev)

RECIPES = {  # name: (width, epochs, lr, label_noise, input_noise, subset)
    "small_1ep":   (32, 1, 1e-3, 0.0, 0.0, 60000),
    "wide_3ep":    (256, 3, 1e-3, 0.0, 0.0, 60000),
    "labelnoise":  (256, 3, 1e-3, 0.3, 0.0, 60000),
    "noisy_input": (128, 3, 1e-3, 0.0, 0.3, 60000),
}
S = int(os.environ.get("SEEDS", 32))


class BMLP(torch.nn.Module):
    def __init__(self, B, w):
        super().__init__()
        def lin(i, o):
            W = torch.empty(B, i, o); [torch.nn.init.kaiming_uniform_(W[b].T, a=5 ** 0.5) for b in range(B)]
            return torch.nn.Parameter(W), torch.nn.Parameter(torch.zeros(B, 1, o))
        self.W1, self.b1 = lin(784, w); self.W2, self.b2 = lin(w, w); self.W3, self.b3 = lin(w, 10)
    def forward(self, x):  # x: (B, bs, 784)
        h = F.relu(torch.bmm(x, self.W1) + self.b1)
        h = F.relu(torch.bmm(h, self.W2) + self.b2)
        return torch.bmm(h, self.W3) + self.b3


probs, labels, accs = [], [], []
t0 = time.time()
for gi, (name, (w, ep, lr, ln, inn, sub)) in enumerate(RECIPES.items()):
    torch.manual_seed(100 + gi)
    model = BMLP(S, w).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(gi)
    y_noisy = ytr_t.clone().unsqueeze(0).repeat(S, 1)
    if ln > 0:  # independent symmetric label noise per model
        flip = torch.rand(S, len(ytr), generator=g).to(dev) < ln
        y_noisy[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g).to(dev)
    bs = 256
    for e in range(ep):
        perms = torch.stack([torch.randperm(len(ytr), generator=g) for _ in range(S)]).to(dev)
        for s in range(0, len(ytr) - bs + 1, bs):
            ib = perms[:, s:s + bs]
            xb = Xtr_t[ib]
            if inn > 0:
                xb = xb + inn * torch.randn_like(xb)
            yb = torch.gather(y_noisy, 1, ib)
            loss = F.cross_entropy(model(xb).reshape(-1, 10), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.softmax(model(Xte_t.unsqueeze(0).expand(S, -1, -1)), -1)
    probs.append(P); labels += [name] * S
    accs += (P.argmax(-1) == yte_t).float().mean(1).tolist()
    print(f"{name}: acc {np.mean(accs[-S:]):.4f} ({time.time()-t0:.0f}s)", flush=True)

P = torch.cat(probs).double()  # (n, T, 10)
n = P.shape[0]
D = torch.zeros(n, n, dtype=torch.float64, device=dev)
eps = 1e-12
for i in range(n):
    M = 0.5 * (P[i:i + 1] + P)
    kl1 = (P[i:i + 1] * (torch.log2(P[i:i + 1] + eps) - torch.log2(M + eps))).sum(-1)
    kl2 = (P * (torch.log2(P + eps) - torch.log2(M + eps))).sum(-1)
    D[i] = torch.sqrt(torch.clamp(0.5 * (kl1 + kl2), min=0)).mean(-1)
D = 0.5 * (D + D.T); D.fill_diagonal_(0)
out = sys.argv[1] if len(sys.argv) > 1 else "zoo_fmnist.npz"
np.savez(out, D=D.cpu().numpy(), n=n, task="fmnist_zoo", iter=0, seed=0, reward=float(np.mean(accs)),
         kind="zoo", recipe=np.array(labels), acc=np.array(accs))
print("saved", out, "n", n, f"total {time.time()-t0:.0f}s")
