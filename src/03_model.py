"""Stage 3: VIFs and the OLS model on all 242 players.

Outputs
  results/vif.csv
  results/ols_coefficients.csv   estimate, SE, t, p, standardized beta
  results/ols_summary.txt        R-style coefficient table and fit statistics
  results/overall_fit.csv        n, R2, adjusted R2, RMSE, MAE, F test
  results/diagnostics.csv        residual normality, heteroscedasticity, RESET
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as C

df = pd.read_csv(C.MODEL_DATA)
X = df[C.X_COLS]
y = df[C.TARGET]
Xc = sm.add_constant(X)

# VIFs
vif = pd.DataFrame({"stat": [C.LABELS[c] for c in C.X_COLS],
                    "vif": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(C.X_COLS))]})
vif = vif.sort_values("vif", ascending=False)
vif.round(4).to_csv(C.RESULTS / "vif.csv", index=False)
print(vif.round(2).to_string(index=False))

# OLS
m = sm.OLS(y, Xc).fit()
resid = m.resid
rmse = float(np.sqrt(np.mean(resid ** 2)))
mae = float(np.mean(np.abs(resid)))
sd = X.std(ddof=1)
coef = pd.DataFrame({
    "term": ["(Intercept)"] + [C.LABELS[c] for c in C.X_COLS],
    "estimate": m.params.values, "std_error": m.bse.values, "t_value": m.tvalues.values,
    "p_value": m.pvalues.values,
    "std_beta": [np.nan] + [m.params[c] * sd[c] / y.std(ddof=1) for c in C.X_COLS],
})
coef.to_csv(C.RESULTS / "ols_coefficients.csv", index=False)

fit = pd.DataFrame({"statistic": ["n", "r2", "adj_r2", "rmse", "mae", "residual_se", "f_stat", "f_df1", "f_df2", "f_p"],
                    "value": [len(y), m.rsquared, m.rsquared_adj, rmse, mae, np.sqrt(m.scale),
                              m.fvalue, m.df_model, m.df_resid, m.f_pvalue]})
fit.to_csv(C.RESULTS / "overall_fit.csv", index=False)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .1 else ""


def pfmt(p):
    return "< 2e-16" if p < 2e-16 else (f"{p:.2e}" if p < 1e-4 else f"{p:.6f}")


lines = ["Coefficients:", f"{'':<12}{'Estimate':>10}{'Std. Error':>11}{'t value':>9}{'Pr(>|t|)':>10}"]
for _, r in coef.iterrows():
    lines.append(f"{r.term:<12}{r.estimate:>10.5f}{r.std_error:>11.5f}{r.t_value:>9.3f}"
                 f"{pfmt(r.p_value):>10} {stars(r.p_value)}")
lines += ["---", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "",
          f"Residual standard error: {np.sqrt(m.scale):.3f} on {int(m.df_resid)} degrees of freedom",
          f"Multiple R-squared:  {m.rsquared:.4f},    Adjusted R-squared:  {m.rsquared_adj:.4f}",
          f"F-statistic: {m.fvalue:.2f} on {int(m.df_model)} and {int(m.df_resid)} DF,  p-value: {pfmt(m.f_pvalue)}",
          "", "Note: TS%, 3PAr and FTr are stored as decimals (0.607), so their estimates are per 1.00;",
          "divide by 100 for the effect of one percentage point. The other six stats are in percentage points.",
          "", f"RMSE (divides by n): {rmse:.4f}    MAE: {mae:.4f}"]
summary = "\n".join(lines)
(C.RESULTS / "ols_summary.txt").write_text(summary + "\n")
print(summary)

# diagnostics
diag = pd.DataFrame({
    "test": ["Shapiro-Wilk on residuals", "Breusch-Pagan", "Ramsey RESET (power 2)"],
    "p_value": [stats.shapiro(resid).pvalue, het_breuschpagan(resid, Xc)[1],
                float(linear_reset(m, power=2, use_f=True).pvalue)],
})
diag.to_csv(C.RESULTS / "diagnostics.csv", index=False)
print(diag.round(3).to_string(index=False))
