"""Stage 1: build the modeling dataset from the raw files.

Inputs
  data/raw/bbref_advanced_season_rows.csv  one Basketball-Reference Advanced row per player-season
                                           (traded players: the combined multi-team row)
  data/raw/nbarapm_5y_ending_2026.csv      nbarapm 5-year RAPM, ORAPM, DRAPM and possessions

Steps
  1. keep players with a row in all five seasons (2021-22 to 2025-26)      -> 249 players
  2. average each rate over the five seasons, weighted by minutes played
  3. join 5-year RAPM and apply 2,500+ total minutes and 5,000+ possessions -> 242 players
  4. check for missing values and duplicates
"""
import numpy as np
import pandas as pd

import config as C

seasons = pd.read_csv(C.SEASON_ROWS)
rapm = pd.read_csv(C.RAPM_FILE)

assert not seasons.duplicated(["season", "nba_id"]).any(), "duplicate player-season rows"
assert set(seasons["season"]) == set(C.SEASONS)

n_seasons = seasons.groupby("nba_id")["season"].nunique()
population_ids = n_seasons[n_seasons == len(C.SEASONS)].index
pop = seasons[seasons["nba_id"].isin(population_ids)]
print(f"Players with a row in all 5 seasons: {len(population_ids)}")


def weighted(group):
    out = {"games": group["br_G"].sum(), "minutes": group["br_MP"].sum()}
    for src, (dst, _) in C.FEATURES.items():
        out[dst] = np.average(group[src], weights=group["br_MP"])
    return pd.Series(out)


agg = pop.groupby("nba_id").apply(weighted, include_groups=False).reset_index()
df = rapm.merge(agg, on="nba_id", how="inner", validate="one_to_one")
assert len(df) == len(population_ids), "every population player must have a 5-year RAPM"

gate = (df["minutes"] >= C.MIN_MINUTES) & (df["rapm_possessions"] >= C.MIN_POSSESSIONS)
removed = df.loc[~gate, ["player", "minutes", "rapm_possessions"]]
print(f"Removed by the minutes/possessions cutoffs: {len(removed)}")
print(removed.to_string(index=False))

model = df[gate].copy()
cols = ["nba_id", "player", "games", "minutes", "rapm_possessions",
        C.TARGET, "orapm_5y", "drapm_5y"] + C.X_COLS
# rows are ordered by nba_id; the 80-20 split and the CV shuffles depend on this order
model = model[cols].sort_values("nba_id").reset_index(drop=True)
model["games"] = model["games"].astype(int)
model["minutes"] = model["minutes"].astype(int)

missing = int(model.isna().sum().sum())
dupes = int(model["nba_id"].duplicated().sum())
zero_3par = int((model["three_par"] == 0).sum())
print(f"Modeled players: {len(model)} | missing values: {missing} | duplicate players: {dupes}"
      f" | players with 3PAr = 0: {zero_3par}")
assert missing == 0 and dupes == 0

C.PROCESSED.mkdir(parents=True, exist_ok=True)
model.to_csv(C.MODEL_DATA, index=False)
print(f"Wrote {C.MODEL_DATA.relative_to(C.ROOT)}")
