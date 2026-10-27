"""Stage 5: predictive validity.

  80-20 train-test split (seed 42): 193 training players, 49 test players
  Leave-one-out cross-validation (242 folds)
  10-fold cross-validation, repeated 200 times (numpy default_rng(0): each repeat shuffles the
  players and cuts them into 10 folds)
  1,000 random 80-20 splits, to show how much a single split can vary

Output
  results/validation.csv
  results/split_variability.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split

import config as C

df = pd.read_csv(C.MODEL_DATA)
X = df[C.X_COLS].to_numpy()
y = df[C.TARGET].to_numpy()
ols = LinearRegression()


def scores(actual, pred):
    return r2_score(actual, pred), mean_squared_error(actual, pred) ** .5, mean_absolute_error(actual, pred)


rows = []
pred = ols.fit(X, y).predict(X)
rows.append(("In-sample (all players)", len(y), *scores(y, pred)))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.SPLIT_SEED)
ols.fit(Xtr, ytr)
rows.append(("Training set (80%)", len(ytr), *scores(ytr, ols.predict(Xtr))))
rows.append(("Test set (20%)", len(yte), *scores(yte, ols.predict(Xte))))

loo = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
rows.append(("Leave-one-out CV", len(y), *scores(y, loo)))

rng = np.random.default_rng(0)
kf = []
for _ in range(C.KFOLD_REPEATS):
    order = rng.permutation(len(y))
    pred = np.empty(len(y))
    for test_idx in np.array_split(order, 10):
        train_idx = np.setdiff1d(order, test_idx)
        pred[test_idx] = LinearRegression().fit(X[train_idx], y[train_idx]).predict(X[test_idx])
    kf.append(scores(y, pred))
kf = np.array(kf)
rows.append((f"10-fold CV (repeated {C.KFOLD_REPEATS}x, mean)", len(y), *kf.mean(axis=0)))

res = pd.DataFrame(rows, columns=["method", "n", "r2", "rmse", "mae"])
res.to_csv(C.RESULTS / "validation.csv", index=False)
print(res.round(4).to_string(index=False))
print(f"10-fold R2 across repeats: 5th pct {np.percentile(kf[:, 0], 5):.4f}, "
      f"95th pct {np.percentile(kf[:, 0], 95):.4f}")

split = []
for s in range(1000):
    a, b, c, d = train_test_split(X, y, test_size=C.TEST_SIZE, random_state=s)
    split.append(r2_score(d, LinearRegression().fit(a, c).predict(b)))
split = np.array(split)
var = pd.DataFrame({"statistic": ["splits", "mean_test_r2", "p05", "p95"],
                    "value": [len(split), split.mean(), np.percentile(split, 5), np.percentile(split, 95)]})
var.to_csv(C.RESULTS / "split_variability.csv", index=False)
print(var.round(4).to_string(index=False))
