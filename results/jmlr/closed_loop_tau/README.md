# Soft-update (tau) sweep of the simulated DiCo loop (supplement table `tab:tau`)

`python experiments/jmlr/closed_loop_tau.py results/jmlr/closed_loop_tau/tau_n50.json 10 4 50` (seconds) and
`python experiments/jmlr/closed_loop_tau.py results/jmlr/closed_loop_tau/tau_n500.json 10 4 500` (minutes).
Same drifting populations as `experiments/jmlr/closed_loop.py`; the controller rescales with the running average
s_t = (1 - tau) s_{t-1} + tau * estimate_t, tau in {1, 0.3, 0.1, 0.03, 0.01}. Each row: n, setting (equal or unequal
scales), design (all pairs / uniform / balanced round-robin), tau, mean and standard error over 10 populations of the
root-mean-square tracking error over steps 100-300, and the per-population values.
