"""Stage 7: proof. Checks every number in README.md against a fresh run of the pipeline.

For each check, the computed value is rounded the way the README shows it, rebuilt into the exact
README text (for example "| AST% | 3.58 |"), and that text must appear in README.md. It also
checks that the Excel workbook's data and headline results match this pipeline.

Writes results/verification_report.txt and exits with an error if any check fails.
"""
import sys

import numpy as np
import openpyxl
import pandas as pd

import config as C

readme = (C.ROOT / "README.md").read_text(encoding="utf-8")
readme = readme.replace("\\-", "-").replace("\\#", "#").replace("\\+", "+").replace("\\*", "*")
R = lambda name: pd.read_csv(C.RESULTS / name)  # noqa: E731
log, failures = [], 0


def check(description, expected_text, ok=None):
    global failures
    passed = (expected_text in readme) if ok is None else bool(ok)
    failures += not passed
    log.append(f"{'PASS' if passed else 'FAIL'}  {description}: {expected_text}")


def r2dp(v):
    return f"{v:.2f}".replace("0.", ".", 1) if v >= 0 else "-" + f"{-v:.2f}".replace("0.", ".", 1)


df = pd.read_csv(C.MODEL_DATA)

# ---- sample ----
log.append("== Sample")
check("Modeled players", f"{len(df)} players total")
check("Cutoffs", f"{C.MIN_MINUTES}+ total minutes and {C.MIN_POSSESSIONS:,}+ RAPM possessions",
      ok=df.minutes.min() >= C.MIN_MINUTES and df.rapm_possessions.min() >= C.MIN_POSSESSIONS
      and "2500+ total minutes and 5,000+ RAPM possessions" in readme)
check("Features listed", "TS%, 3PAr, FTr, TRB%, AST%, STL%, BLK%, TOV%, and USG%")
check("No missing values or duplicate players", "no missing / no duplicates",
      ok=df.isna().sum().sum() == 0 and not df.nba_id.duplicated().any())

# ---- correlation matrix: all 100 cells ----
log.append("== Correlation matrix (every cell)")
corr = R("correlation_matrix.csv").set_index(R("correlation_matrix.csv").columns[0])
for row in corr.index:
    cells = ["1" if row == col else r2dp(corr.loc[row, col]) for col in corr.columns]
    check(f"Row {row}", f"| **{row}** | " + " | ".join(cells) + " |")

# ---- box plot ----
log.append("== Box plot")
box = R("rapm_boxplot_values.csv").set_index("location")
v = box["value"].astype(float)
for label, key, fmt in [("Min (Lower Whisker)", "Min", "{:.3f}"), ("Q1 (25th Percentile)", "Q1", "{:.4f}"),
                        ("Median (50th Percentile)", "Median", "{:.4f}"), ("Q3 (75th Percentile)", "Q3", "{:.3f}"),
                        ("Upper Whisker", "Upper whisker", "{:.3f}"), ("Outlier Cutoff", "Upper outlier cutoff", "{:.3f}"),
                        ("Outlier #1", "Outlier #1", "{:.3f}"), ("Outlier #2", "Outlier #2", "{:.3f}"),
                        ("Max (Outlier #3)", "Max", "{:.3f}")]:
    check(f"Box plot {key}", f"| {label} | {fmt.format(v[key])} |")
check("Min is also the lower whisker", "Min == Lower whisker", ok=v["Min"] == v["Lower whisker"])
check("Exactly 3 outliers, none low", "3 high, 0 low",
      ok=sum(box.index.str.startswith("Outlier #")) == 3 and v["Outliers below lower cutoff"] == 0)
check("Max outlier is the third outlier", "Max == Outlier #3", ok=v["Max"] == v["Outlier #3"])
names = box["player"]
check("Outlier names", f"Outlier 1: {names['Outlier #1']}; Outlier 2: {names['Outlier #2']}; "
      f"Max Outlier: {names['Max']}")

# ---- VIFs ----
log.append("== VIFs")
vif = R("vif.csv")
for _, r in vif.iterrows():
    check(f"VIF {r.stat}", f"| {r.stat} | {r.vif:.2f} |")
check("All VIFs below 5", "max VIF < 5", ok=vif.vif.max() < 5)

# ---- overall fit ----
log.append("== Overall results")
fit = R("overall_fit.csv").set_index("statistic")["value"]
check("Sample size", f"| Sample Size | {int(fit['n'])} |")
check("R2", f"| Coefficient of Determination | {fit['r2']:.2f} |")
check("RMSE", f"| Root Mean Square Error | {fit['rmse']:.2f} |")
check("MAE", f"| Mean Absolute Error | {fit['mae']:.2f} |")
check("Unexplained share", f"leaves {round(100 * (1 - fit['r2']))}% of the variance unexplained")
check("Model is statistically significant (F test)", f"F p = {fit['f_p']:.1e}", ok=fit["f_p"] < .001)

# ---- dominance analysis ----
log.append("== Dominance analysis")
dom = R("dominance_general.csv")
coef = R("ols_coefficients.csv").set_index("term")
for _, r in dom.iterrows():
    star = "*" if r.stat == "TOV%" else ""
    check(f"Rank {r['rank']}", f"| {r['rank']} | {star}{r.stat} | {r.share_of_r2_pct:.1f}% | "
          f"{r.share_of_all_variance_pct:.1f}% |")
check("Dominance total", f"**{dom.share_of_all_variance_pct.sum():.1f}% (R² {dom.general_dominance.sum():.3f})**")
check("Dominance weights sum to R2", "sum == R2", ok=abs(dom.general_dominance.sum() - fit["r2"]) < 1e-12)
negatives = [t for t in coef.index if t != "(Intercept)" and coef.loc[t, "estimate"] < 0]
check("TOV% is the only negative coefficient (footnote)", f"negative: {negatives}, p = {coef.loc['TOV%', 'p_value']:.1e}",
      ok=negatives == ["TOV%"] and coef.loc["TOV%", "p_value"] < .001
      and "*TOV% has a negative correlation to RAPM directionally when controlling for the other stats" in readme)

# ---- validation ----
log.append("== Training, test and cross-validation")
val = R("validation.csv").set_index("method")
tr, te = val.loc["Training set (80%)"], val.loc["Test set (20%)"]
check("Training n", f"| Sample Size | {int(tr.n)} (80% of overall dataset) |")
check("Training R2", f"| Coefficient of Determination | {tr.r2:.3f} |")
check("Training RMSE", f"| Root Mean Square Error | {tr.rmse:.2f} |")
check("Training MAE", f"| Mean Absolute Error | {tr.mae:.2f} |")
check("Test set n (prose)", f"test set of {int(te.n)} players")
check("Test R2", f"| Test Set ({int(te.n)} players) | {te.r2:.3f} |")
check("LOO R2", f"| Leave-One-Out Cross Validation | {val.loc['Leave-one-out CV'].r2:.3f} |")
kname = [m for m in val.index if m.startswith("10-fold")][0]
check("10-fold R2", f"| 10-fold Cross Validation | {val.loc[kname].r2:.3f} |")
check("Prose: about 44% in training", "about 44% of explained variance in the training set", ok=round(tr.r2 * 100) == 44
      and "about 44% of explained variance in the training set" in readme)
oos = [te.r2, val.loc["Leave-one-out CV"].r2, val.loc[kname].r2]
check("Prose: around 40% out of sample", "around 40% among the test set and Cross-Validation results",
      ok=all(round(x * 100) in (39, 40) for x in oos) and "around 40% among the test set" in readme)

# ---- offense / defense ----
log.append("== Offense and defense")
od = R("offense_defense.csv").set_index("target")
for t in od.index:
    check(t, f"| {t} | {od.loc[t, 'in_sample_r2']:.3f} | {od.loc[t, 'loo_r2']:.3f} |")
check("Prose: offense", f"explained {100 * od.loc['Offensive RAPM', 'in_sample_r2']:.1f}% of the variance in 5-year Offensive")
check("Prose: defense", f"only explained {100 * od.loc['Defensive RAPM', 'in_sample_r2']:.1f}% for the Defensive")

# ---- Excel workbook matches the pipeline ----
log.append("== Excel workbook")
wb = openpyxl.load_workbook(C.ROOT / "workbook" / "RAPM_BoxScore_Dominance.xlsx", data_only=True)
ws = wb["Data"]
hdr = [c.value for c in ws[1]]
wdf = pd.DataFrame(list(ws.iter_rows(min_row=2, values_only=True)), columns=hdr).dropna(subset=["nba_id"])
wdf = wdf.set_index("nba_id").loc[df.nba_id]
pairs = {"RAPM5 (target)": C.TARGET, "TS%": "ts_pct", "3PAr": "three_par", "FTr": "ftr", "TRB%": "trb_pct",
         "AST%": "ast_pct", "STL%": "stl_pct", "BLK%": "blk_pct", "TOV%": "tov_pct", "USG%": "usg_pct",
         "ORAPM5 (offense)": "orapm_5y", "DRAPM5 (defense)": "drapm_5y"}
maxdiff = max(np.abs(wdf[w].astype(float).to_numpy() - df[c].to_numpy()).max() for w, c in pairs.items())
check("Workbook data equals pipeline data (242 players x 12 columns)", f"max abs difference {maxdiff:.1e}",
      ok=len(wdf) == len(df) and maxdiff < 1e-9)
ans = {r[0]: r[1] for r in wb["Answers"].iter_rows(min_row=4, values_only=True) if r[0]}
check("Workbook R2 (live LINEST)", f"{ans['R² (in-sample)']:.10f}", ok=abs(ans["R² (in-sample)"] - fit["r2"]) < 1e-9)
check("Workbook LOO R2", f"{ans['Leave-one-out cross-validated R²']:.6f}",
      ok=abs(ans["Leave-one-out cross-validated R²"] - val.loc["Leave-one-out CV"].r2) < 1e-6)
check("Workbook 10-fold R2", f"{ans['10-fold cross-validated R² (mean of 200 repeats)']:.6f}",
      ok=abs(ans["10-fold cross-validated R² (mean of 200 repeats)"] - val.loc[kname].r2) < 1e-6)

passed = sum(line.startswith("PASS") for line in log)
log.append("")
log.append(f"{passed} checks passed, {failures} failed")
report = "\n".join(log)
(C.RESULTS / "verification_report.txt").write_text(report + "\n", encoding="utf-8")
print(report)
sys.exit(1 if failures else 0)
