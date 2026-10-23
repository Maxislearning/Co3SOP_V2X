"""任务34b: how many runs / training seeds are needed (statistical-power skill).
Model per run r for contrast A-B:  d_r = Delta + u_r + (eA_r - eB_r),
  u_r ~ N(0, tau^2) scenario x treatment interaction, eX_r ~ N(0, sigma_s^2 / k) seed-mean noise.
Planned analysis: paired t-test over runs on seed-mean accuracies (as in per_run_seed.py).
Closed form (t_paired, dz = Delta / sqrt(tau^2 + 2 sigma_s^2 / k)) + Monte Carlo check with the skill harness."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, '/home/admin0/.claude/skills/statistical-power/scripts')
from power import sample_size, power as pw  # noqa: E402
from simulate_power import simulate_power  # noqa: E402

OUT = sys.argv[1]
SIGMA_S = 0.05
rows = []
for alpha, lab in ((0.05, 'alpha=.05'), (0.05 / 3, 'alpha=.0167 (Bonferroni over 3 key contrasts)')):
    for delta in (0.05, 0.10):
        for tau in (0.065, 0.10, 0.14):
            for k in (1, 3, 5):
                dz = delta / np.sqrt(tau ** 2 + 2 * SIGMA_S ** 2 / k)
                n = sample_size(test='t_paired', effect_size=dz, power=0.80, alpha=alpha)
                p6 = pw(test='t_paired', effect_size=dz, nobs1=6, alpha=alpha)
                rows.append(dict(alpha=lab, delta_pp=delta * 100, tau=tau, seeds=k, dz=round(dz, 3),
                                 runs_needed_80pct=int(np.ceil(n)), power_with_6_runs=round(p6, 3)))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/data/power_plan.csv', index=False)


# Monte Carlo check of one cell with the skill harness (normal model, same planned test)
def gen(delta, tau, k, alpha=0.05):
    def f(n, rng):
        d = delta + rng.normal(0, tau, n) + rng.normal(0, SIGMA_S / np.sqrt(k), n) - rng.normal(0, SIGMA_S / np.sqrt(k), n)
        return stats.ttest_1samp(d, 0).pvalue < alpha
    return f


chk = []
for delta, tau, k, n in ((0.0, 0.10, 3, 20), (0.05, 0.10, 3, 36), (0.10, 0.14, 3, 18)):
    est = simulate_power(gen(delta, tau, k), n=n, n_sims=4000, alpha=0.05, seed=1)
    chk.append(dict(delta=delta, tau=tau, seeds=k, runs=n, sim=str(est)))
pd.DataFrame(chk).to_csv(f'{OUT}/data/power_mc_check.csv', index=False)
pd.set_option('display.width', 220)
print(T.pivot_table(index=['alpha', 'delta_pp', 'tau'], columns='seeds', values='runs_needed_80pct').to_string())
print(T[T.alpha == 'alpha=.05'].pivot_table(index=['delta_pp', 'tau'], columns='seeds', values='power_with_6_runs').to_string())
for c in chk: print(c)
