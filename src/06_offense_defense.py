"""Stage 6: the same 9 features fit to offensive and defensive 5-year RAPM.

Output
  results/offense_defense.csv   in-sample and leave-one-out R2 for total, offensive and defensive RAPM
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

import config as C

df = pd.read_csv(C.MODEL_DATA)
X = df[C.X_COLS].to_numpy()
rows = []
for name, col in [("Offensive RAPM", "orapm_5y"), ("Defensive RAPM", "drapm_5y"), ("Total RAPM", C.TARGET)]:
    y = df[col].to_numpy()
    ins = LinearRegression().fit(X, y).score(X, y)
    loo = r2_score(y, cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut()))
    rows.append((name, ins, loo))
out = pd.DataFrame(rows, columns=["target", "in_sample_r2", "loo_r2"])
out.to_csv(C.RESULTS / "offense_defense.csv", index=False)
print(out.round(4).to_string(index=False))
