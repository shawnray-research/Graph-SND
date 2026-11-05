import numpy as np, sys
sys.path.insert(0, '.')
from anatomy import *
rng = np.random.default_rng(0)
for fam in FAMILIES:
    D = synthetic(fam, 24, rng)
    an = anatomy(D)
    # orthogonality check
    assert abs(an['s2_tot'] - an['s2_main'] - an['s2_int']) < 1e-10 * max(1, an['s2_tot']), (an['s2_tot'], an['s2_main'], an['s2_int'])
    R = an['R']; assert np.allclose(R.sum(1), 0, atol=1e-9); assert abs(an['a'].sum()) < 1e-9
    des = Designs(D, rng)
    line = [fam, f"kappa={an['kappa']:.3g}"]
    for k in [1, 3, 6]:
        est, m = des.round_robin(k, 20000)
        vr = var_balanced(an, k)
        est2, _ = des.shuffled_circulant(k, 20000)
        estu, _ = des.uniform(m, 20000)
        vu = var_uniform(an, m)
        estb, _ = des.bernoulli_ht(m, 20000)
        vb = var_bernoulli_ht(D, m / an['N'])
        line.append(f"k={k}: RR {np.var(est)/vr:.3f} bias {abs(est.mean()-an['mu'])/np.sqrt(vr/20000):.1f}se | circ {np.var(est2)/vr:.3f} | U {np.var(estu)/vu:.3f} | HT {np.var(estb)/vb:.3f}")
    print(" ".join(line))
