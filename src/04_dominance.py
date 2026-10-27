"""Stage 4: exact dominance analysis (Budescu, 1993; Azen & Budescu, 2003).

Fits OLS on every one of the 2^9 = 512 subsets of the 9 features and records each subset's R2.
  conditional dominance: a feature's average R2 gain when added to subsets of size k (k = 0..8)
  general dominance:     the average of its 9 conditional values; these sum exactly to the full-model R2
  complete dominance:    feature A completely dominates B if A adds more R2 than B to every subset
                         that contains neither

Outputs
  results/dominance_general.csv
  results/dominance_conditional.csv
  results/dominance_complete_pairs.csv
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

import config as C

df = pd.read_csv(C.MODEL_DATA)
y = df[C.TARGET].to_numpy()
feats = C.X_COLS
p = len(feats)


def r2(subset):
    if not subset:
        return 0.0
    X = np.column_stack([np.ones(len(y))] + [df[f].to_numpy() for f in subset])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


R2 = {frozenset(s): r2(list(s)) for k in range(p + 1) for s in combinations(feats, k)}
print(f"Sub-models fitted: {len(R2)}")
full = R2[frozenset(feats)]

cond = pd.DataFrame(index=feats, columns=range(p), dtype=float)
for f in feats:
    others = [o for o in feats if o != f]
    for k in range(p):
        gains = [R2[frozenset(s) | {f}] - R2[frozenset(s)] for s in combinations(others, k)]
        assert len(gains) == comb(p - 1, k)
        cond.loc[f, k] = np.mean(gains)

gen = cond.mean(axis=1)
assert abs(gen.sum() - full) < 1e-12, "general dominance must sum to the full-model R2"

out = pd.DataFrame({"stat": [C.LABELS[f] for f in feats], "general_dominance": gen.values,
                    "share_of_r2_pct": 100 * gen.values / full, "share_of_all_variance_pct": 100 * gen.values})
out = out.sort_values("general_dominance", ascending=False).reset_index(drop=True)
out.insert(0, "rank", range(1, p + 1))
out.to_csv(C.RESULTS / "dominance_general.csv", index=False)

cond.index = [C.LABELS[f] for f in feats]
cond.columns = [f"added_to_{k}_others" for k in range(p)]
cond.to_csv(C.RESULTS / "dominance_conditional.csv")

pairs = []
for a, b in combinations(feats, 2):
    rest = [o for o in feats if o not in (a, b)]
    diffs = [(R2[frozenset(s) | {a}] - R2[frozenset(s)]) - (R2[frozenset(s) | {b}] - R2[frozenset(s)])
             for k in range(p - 1) for s in combinations(rest, k)]
    cmean = [cond.loc[C.LABELS[a]].iloc[k] - cond.loc[C.LABELS[b]].iloc[k] for k in range(p)]
    winner = C.LABELS[a] if gen[a] > gen[b] else C.LABELS[b]
    loser = C.LABELS[b] if winner == C.LABELS[a] else C.LABELS[a]
    complete = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    conditional = all(d > 0 for d in cmean) or all(d < 0 for d in cmean)
    pairs.append((winner, loser, complete, conditional))
pairs = pd.DataFrame(pairs, columns=["general_winner", "general_loser", "complete_dominance", "conditional_dominance"])
pairs.to_csv(C.RESULTS / "dominance_complete_pairs.csv", index=False)

print(f"Full-model R2: {full:.4f}")
print(out.round(4).to_string(index=False))
print(f"Pairs with complete dominance: {int(pairs.complete_dominance.sum())} of {len(pairs)}; "
      f"with conditional dominance: {int(pairs.conditional_dominance.sum())} of {len(pairs)}")
