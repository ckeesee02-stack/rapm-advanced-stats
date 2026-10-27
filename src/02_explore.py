"""Stage 2: data understanding.

Outputs
  results/correlation_matrix.csv     Pearson r for RAPM and the 9 features
  results/rapm_boxplot_values.csv    quartiles, whiskers, outlier cutoff and outliers (with players)
  results/rapm_summary.csv           mean, SD, skew and Shapiro-Wilk for RAPM
  figures/correlation_matrix.png
  figures/rapm_boxplot.png
"""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

df = pd.read_csv(C.MODEL_DATA)
y = df[C.TARGET].to_numpy()
C.RESULTS.mkdir(exist_ok=True)
C.FIGURES.mkdir(exist_ok=True)

# ---------- correlation matrix ----------
corr = df[C.CORR_ORDER].corr()
corr.index = corr.columns = [C.label(c) for c in C.CORR_ORDER]
corr.round(4).to_csv(C.RESULTS / "correlation_matrix.csv")
t = stats.t.ppf(0.975, len(df) - 2)
r_crit = t / np.sqrt(len(df) - 2 + t * t)
print(f"Correlations of |{r_crit:.3f}| or more are significant at p < .05 (n = {len(df)})")
print(corr.round(2).to_string())

# ---------- box plot values (inclusive quartiles = Excel QUARTILE.INC = numpy 'linear') ----------
q1, med, q3 = np.percentile(y, [25, 50, 75], method="linear")
iqr = q3 - q1
lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
inside = y[(y >= lo_fence) & (y <= hi_fence)]
lo_whisk, hi_whisk = inside.min(), inside.max()
name = lambda v: df.loc[np.isclose(df[C.TARGET], v), "player"].iloc[0]  # noqa: E731
rows = [("Min", y.min(), name(y.min())), ("Lower whisker", lo_whisk, name(lo_whisk)),
        ("Q1", q1, ""), ("Median", med, ""), ("Q3", q3, ""), ("IQR", iqr, ""),
        ("Upper whisker", hi_whisk, name(hi_whisk)), ("Lower outlier cutoff", lo_fence, ""),
        ("Upper outlier cutoff", hi_fence, ""), ("Max", y.max(), name(y.max()))]
for i, v in enumerate(sorted(y[y > hi_fence]), 1):
    rows.append((f"Outlier #{i}", v, name(v)))
rows.append(("Outliers below lower cutoff", int((y < lo_fence).sum()), ""))
box = pd.DataFrame(rows, columns=["location", "value", "player"])
box.to_csv(C.RESULTS / "rapm_boxplot_values.csv", index=False)
print(box.to_string(index=False))

summary = pd.DataFrame({
    "statistic": ["n", "mean", "sd", "skew", "share_positive", "shapiro_wilk_p"],
    "value": [len(y), y.mean(), y.std(ddof=1), stats.skew(y), (y > 0).mean(), stats.shapiro(y).pvalue],
})
summary.to_csv(C.RESULTS / "rapm_summary.csv", index=False)

# ---------- figures ----------
for f in C.FONTS.glob("*.ttf"):
    fm.fontManager.addfont(str(f))
names = {f.name for f in fm.fontManager.ttflist}
BC = "Barlow Condensed" if "Barlow Condensed" in names else "DejaVu Sans"
PS = "IBM Plex Sans" if "IBM Plex Sans" in names else "DejaVu Sans"
BG, INK, MUTED, DIM, OR = "#14202E", "#F4F6F8", "#AEB8C4", "#8C99A8", "#F08A4B"


def minus(s):
    return s.replace("-", "−")


# correlation heatmap
labs = list(corr.columns)
k = len(labs)
cmap = LinearSegmentedColormap.from_list("div", ["#5B9BD5", "#22324A", OR])
fig = plt.figure(figsize=(16, 16.5), dpi=100, facecolor=BG)
ax = fig.add_axes([0.12, 0.11, 0.80, 0.725])
ax.set_facecolor(BG)
for i in range(k):
    for j in range(k):
        r = corr.iloc[i, j]
        col = "#0D1520" if i == j else cmap((r + 1) / 2)
        ax.add_patch(plt.Rectangle((j - .47, i - .47), .94, .94, facecolor=col, edgecolor="none"))
        s = "1" if i == j else ("−" if r < 0 else "") + f"{abs(r):.2f}".lstrip("0")
        tc = MUTED if i == j else (BG if abs(r) >= .55 else (INK if abs(r) >= r_crit else DIM))
        ax.text(j, i, s, ha="center", va="center", fontsize=22, fontfamily=PS, color=tc)
ax.set_xlim(-.55, k - .45)
ax.set_ylim(k - .45, -.55)
ax.set_xticks(range(k))
ax.set_xticklabels(labs, fontfamily=BC, fontsize=24, fontweight=700, color=INK)
ax.set_yticks(range(k))
ax.set_yticklabels(labs, fontfamily=BC, fontsize=24, fontweight=700, color=INK)
ax.tick_params(length=0, pad=8)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.text(.12, .935, "HOW THE STATS MOVE TOGETHER", fontfamily=BC, fontweight=700, fontsize=56, color=INK)
fig.text(.12, .905, f"Correlation matrix (Pearson r) of the model's variables · {len(df)} players · "
         "5-year window, 2021-22 to 2025-26", fontfamily=PS, fontsize=19, color=MUTED)
cax = fig.add_axes([.57, .858, .35, .016])
cax.imshow(np.linspace(-1, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=[-1, 1, 0, 1])
cax.set_yticks([])
cax.set_xticks([-1, -.5, 0, .5, 1])
cax.set_xticklabels(["−1", "−.5", "0", "+.5", "+1"], fontfamily=PS, fontsize=15, color=MUTED)
cax.tick_params(length=0, pad=6)
for sp in cax.spines.values():
    sp.set_visible(False)
fig.text(.57, .88, "Moves opposite", fontfamily=PS, fontsize=15, color="#7FB2E5")
fig.text(.92, .88, "Moves together", fontfamily=PS, fontsize=15, color=OR, ha="right")
fig.text(.12, .045, "Grouped by role:  Creation (AST, USG, TOV) · Shooting (TS, FTr, 3PAr) · "
         "Defense & size (STL, BLK, TRB)", fontfamily=PS, fontsize=15, color=MUTED)
fig.text(.12, .02, "Gray numbers are not statistically significant (p > .05). Diagonal = each stat with itself "
         "(always 1). Each pair appears twice.", fontfamily=PS, fontsize=15, color=MUTED)
fig.text(.92, .945, "@HoopKeesee", fontfamily=BC, fontweight=700, fontsize=30, color=INK, ha="right")
fig.savefig(C.FIGURES / "correlation_matrix.png", facecolor=BG)
plt.close(fig)

# box plot
fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
ax = fig.add_axes([.06, .22, .88, .52])
ax.set_facecolor(BG)
bp = ax.boxplot(y, orientation="horizontal", whis=1.5, widths=.42, patch_artist=True,
                boxprops=dict(facecolor="#3A2A22", edgecolor=OR, linewidth=2.5),
                medianprops=dict(color=INK, linewidth=3.5), whiskerprops=dict(color=MUTED, linewidth=2),
                capprops=dict(color=MUTED, linewidth=2),
                flierprops=dict(marker="o", markerfacecolor=OR, markeredgecolor=BG, markersize=13))
ax.set_ylim(.35, 1.68)
ax.set_xlim(-7.2, 11.2)
ax.set_yticks([])
ticks = range(-6, 12, 2)
ax.set_xticks(ticks)
ax.set_xticklabels([("−" if v < 0 else "+" if v > 0 else "") + str(abs(v)) for v in ticks],
                   fontfamily=PS, fontsize=15, color=MUTED)
ax.axhline(.36, color="#3E5068", lw=1.5, zorder=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0, pad=8)
ax.set_xlabel("5-year RAPM (points per 100 possessions)", fontfamily=PS, fontsize=14, color=DIM, labelpad=10)
ax.plot([hi_fence] * 2, [.76, 1.24], color=DIM, lw=1.5, ls=(0, (4, 4)))
ax.text(hi_fence, .73, f"Outlier cutoff\n{hi_fence:.3f}", ha="center", va="top", fontfamily=PS, fontsize=13, color=DIM)


def top(x, lab, val, dx=0):
    ax.plot([x, x], [1.22, 1.30], color=MUTED, lw=1.2)
    ax.text(x + dx, 1.31, lab, ha="center", va="bottom", fontfamily=BC, fontweight=700, fontsize=22, color=INK)
    ax.text(x + dx, 1.42, val, ha="center", va="bottom", fontfamily=PS, fontsize=17, color=OR)


top(q1, "Q1", minus(f"{q1:.4f}"), -.25)
top(med, "MEDIAN", f"{med:.4f}")
top(q3, "Q3", f"{q3:.3f}", .2)
ax.plot([q1, q1, q3, q3], [.75, .70, .70, .75], color=OR, lw=1.8)
ax.text((q1 + q3) / 2, .67, f"IQR {iqr:.4f}\nmiddle 50% of players", ha="center", va="top",
        fontfamily=PS, fontsize=14, color=MUTED)
ax.text(lo_whisk, 1.14, minus(f"MIN / LOWER WHISKER\n{lo_whisk:.3f}") + f"  {name(lo_whisk)}",
        ha="left", va="bottom", fontfamily=PS, fontsize=14, color=INK)
ax.text(hi_whisk + .08, .86, f"UPPER WHISKER\n{hi_whisk:.3f}  {name(hi_whisk)}", ha="right", va="top",
        fontfamily=PS, fontsize=14, color=INK)
outs = sorted(y[y > hi_fence])
for v, yy in zip(outs, [1.27, 1.47, 1.27]):
    first = v == outs[0]
    ax.plot([v, v], [1.06, yy - .01], color=DIM, lw=1)
    ax.text(v + (.12 if first else 0), yy, f"{name(v)}\n{v:.3f}" + ("  (MAX)" if v == y.max() else ""),
            ha="right" if first else "center", va="bottom", fontfamily=PS, fontsize=14, color=INK)
fig.text(.06, .9, "HOW 5-YEAR RAPM IS DISTRIBUTED", fontfamily=BC, fontweight=700, fontsize=54, color=INK)
fig.text(.06, .86, f"Box plot of 5-year RAPM · {len(df)} players · 2021-22 to 2025-26",
         fontfamily=PS, fontsize=19, color=MUTED)
fig.text(.94, .9, "@HoopKeesee", fontfamily=BC, fontweight=700, fontsize=30, color=INK, ha="right")
fig.text(.06, .065, "Box = Q1 to Q3 (middle 50%). Whiskers reach the furthest player within 1.5 x IQR of the box; "
         "dots beyond are outliers.", fontfamily=PS, fontsize=14, color=MUTED)
fig.text(.06, .035, minus(f"Quartiles: inclusive linear method (Excel QUARTILE.INC). Mean {y.mean():.3f}. "
         f"No player falls below the lower cutoff ({lo_fence:.3f})."), fontfamily=PS, fontsize=14, color=MUTED)
fig.savefig(C.FIGURES / "rapm_boxplot.png", facecolor=BG)

# proof: the drawn box plot must match the computed values
drawn = {"Q1": bp["boxes"][0].get_path().vertices[:, 0].min(), "Q3": bp["boxes"][0].get_path().vertices[:, 0].max(),
         "Median": bp["medians"][0].get_xdata()[0], "Lower whisker": min(bp["whiskers"][0].get_xdata()),
         "Upper whisker": max(bp["whiskers"][1].get_xdata()), "Outliers": len(bp["fliers"][0].get_xdata())}
expected = {"Q1": q1, "Q3": q3, "Median": med, "Lower whisker": lo_whisk, "Upper whisker": hi_whisk,
            "Outliers": len(outs)}
for key in drawn:
    assert abs(drawn[key] - expected[key]) < 1e-9, f"box plot mismatch: {key}"
print("Box plot geometry matches the computed values")
print("Wrote results/correlation_matrix.csv, results/rapm_boxplot_values.csv, results/rapm_summary.csv, "
      "figures/correlation_matrix.png, figures/rapm_boxplot.png")
