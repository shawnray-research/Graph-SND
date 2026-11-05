import glob, os, re, sys, json
import numpy as np
root = sys.argv[1]
def anatomy(D):
    n = D.shape[0]; iu = np.triu_indices(n, 1); mu = D[iu].mean()
    a = (D.sum(1) - (n - 1) * mu) / (n - 2)
    R = D - mu - a[:, None] - a[None, :]; np.fill_diagonal(R, 0)
    tot = ((D[iu] - mu) ** 2).mean(); inter = (R[iu] ** 2).mean(); main = tot - inter
    ev = np.linalg.eigvalsh(R); top = np.sort(np.abs(ev))[::-1]
    return dict(n=n, mu=mu, cv_tot=np.sqrt(tot)/mu, share_main=main/tot, share_int=inter/tot, kappa=main/inter,
                eff=(1+main/inter)*n*(n-3)/((n-2)*(n+1)), top_eig_frac=float(top[0]**2/(top**2).sum()),
                maxa_over_mu=float(a.max()/mu), rel_sd_bal_k4=float(np.sqrt(inter/(n*2)*(n-5)/(n-3))/mu))
rows = []
for f in sorted(glob.glob(os.path.join(root, '*.npz'))):
    z = np.load(f, allow_pickle=True)
    D = z['D'].astype(float)
    r = anatomy(D); r['file'] = os.path.basename(f)
    rows.append(r)
for r in rows:
    print(f"{r['file']:45s} n={r['n']:3d} mu={r['mu']:.4f} cv={r['cv_tot']:.3f} main={r['share_main']:.2f} int={r['share_int']:.2f} eff={r['eff']:6.2f} topEig={r['top_eig_frac']:.2f} maxa/mu={r['maxa_over_mu']:.2f} sdk4={100*r['rel_sd_bal_k4']:.3f}%")
json.dump(rows, open(os.path.join(root, '..', 'anatomy_real_summary.json'), 'w'), default=float)
