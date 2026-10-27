# Data Dictionary

## data/raw/bbref_advanced_season_rows.csv

One row per player-season, 2021-22 through 2025-26 (2,867 rows). Rates are from Basketball-Reference's league Advanced pages. For players traded mid-season, the combined multi-team row is used, so each player has at most one row per season. Every row was matched to an NBA Stats player-season by name and checked in both directions.

| Column | Meaning |
| :---- | :---- |
| season | NBA season, for example 2021-22 |
| nba_id | NBA Stats player ID (join key) |
| nba_name / br_name | Player name on NBA Stats / Basketball-Reference |
| bbref_id | Basketball-Reference player ID |
| br_G / nba_GP | Games played (Basketball-Reference / NBA Stats) |
| br_MP / nba_MIN | Minutes played (Basketball-Reference / NBA Stats); br_MP is the weight for the 5-year averages |
| ts_pct | True shooting percentage, as a decimal (0.595 = 59.5%) |
| fg3a_per_fga_pct | 3-point attempt rate (3PA / FGA), as a decimal |
| fta_per_fga_pct | Free throw rate (FTA / FGA), as a decimal |
| orb_pct / drb_pct / trb_pct | Offensive / defensive / total rebound percentage, in percentage points |
| ast_pct | Assist percentage, in percentage points |
| stl_pct | Steal percentage, in percentage points |
| blk_pct | Block percentage, in percentage points |
| tov_pct | Turnover percentage, in percentage points |
| usg_pct | Usage percentage, in percentage points |

A few cameo seasons (7 minutes or less, no shot attempts) have blank shooting or turnover rates. None of them belong to players in the modeled sample.

## data/raw/nbarapm_5y_ending_2026.csv

nbarapm.com Pure 5-year RAPM for the window ending 2026 (2021-22 to 2025-26), for the 249 players with a row in all five seasons.

| Column | Meaning |
| :---- | :---- |
| nba_id | NBA Stats player ID (join key) |
| player | Player name |
| rapm_possessions | Possessions behind the player's 5-year RAPM estimate |
| rapm_5y | 5-year RAPM: points per 100 possessions added to team point differential |
| orapm_5y / drapm_5y | Offensive / defensive 5-year RAPM |

## data/processed/model_data.csv

Built by `src/01_build_dataset.py`. One row per modeled player (242 rows), ordered by nba_id.

| Column | Meaning |
| :---- | :---- |
| nba_id, player | Identifiers |
| games, minutes | Totals over the five seasons (Basketball-Reference) |
| rapm_possessions, rapm_5y, orapm_5y, drapm_5y | From the nbarapm file |
| ts_pct, three_par, ftr | TS%, 3PAr, FTr: minutes-weighted 5-year averages, as decimals |
| trb_pct, ast_pct, stl_pct, blk_pct, tov_pct, usg_pct | TRB%, AST%, STL%, BLK%, TOV%, USG%: minutes-weighted 5-year averages, in percentage points |

Inclusion: a row in all five seasons (249 players), then 2,500+ total minutes and 5,000+ RAPM possessions (242 players). The 7 players removed by the cutoffs are DeAndre Jordan, Taj Gibson, Garrett Temple, Charles Bassey, Lindy Waters III, Luka Garza and Micah Potter.

## Sources

- Sports Reference LLC. (n.d.). *Basketball-Reference.com*. https://www.basketball-reference.com
- *nbarapm*. (n.d.). https://nbarapm.com

Both sources' terms of use apply to this data.
