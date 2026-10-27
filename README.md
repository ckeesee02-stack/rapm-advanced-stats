Identification of Research Goal: Determining the percentage variation of 5-Year Regularized Adjusted Plus Minus that can be explained by a model comprising various advanced stats. RAPM measures a player's impact on a team's point differential per 100 possessions. It uses Ridge Regression to account for the quality of teammates and opponents on the floor (Sill, 2010).

Analytical Approach: RAPM values are continuous, and various advanced stats would be used as feature variables. Therefore, the selected analytical approach for this research is the Ordinary Least Squares Multiple Linear Regression approach. This approach is used to determine the coefficient of determination, along with how each variable relates to RAPM when controlling for other variables. A Dominance Analysis is also used to see how much each variable contributes to the overall coefficient of determination.

Data Requirements: The feature variables would consist of advanced stats so that they’re independent of minutes played. This provides a clearer understanding of the knowledge, skillsets, and abilities required to be a positive contributor on the basketball court. 

Data Collection: The advanced stat feature variables were collected from Basketball Reference (Sports Reference LLC, n.d.). They consisted of: TS%, 3PAr, FTr, TRB%, AST%, STL%, BLK%, TOV%, and USG%. The 5-Year RAPM target variable for each player was from the official NBARAPM webpage (*nbarapm*, n.d.). Given that this research consisted of 5-Year RAPM for the target variable, only players who played in the last five completed seasons were included (As of October 2026), with additional cutoffs of 2500+ total minutes and 5,000+ RAPM possessions. This includes the 2025-26 season to the 2021-22 season. 242 players total. The feature variables were also minutes-weighted in the averages over their last 5 seasons.

Data Understanding: Various visualizations were utilized to obtain a better understanding of the data that would be used for linear modeling:

**Correlation Matrix**

| Stat | RAPM | AST% | USG% | TOV% | TS% | FTr | 3PAr | STL% | BLK% | TRB% |
| ----- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAPM** | 1 | .35 | .33 | \-.04 | .23 | .20 | \-.08 | .24 | .04 | .13 |
| **AST%** | .35 | 1 | .63 | .41 | \-.25 | .08 | \-.06 | .24 | \-.36 | \-.18 |
| **USG%** | .33 | .63 | 1 | \-.04 | \-.16 | .25 | \-.10 | \-.08 | \-.20 | \-.05 |
| **TOV%** | \-.04 | .41 | \-.04 | 1 | .01 | .27 | \-.36 | .22 | .14 | .28 |
| **TS%** | .23 | \-.25 | \-.16 | .01 | 1 | .49 | \-.38 | \-.31 | .43 | .44 |
| **FTr** | .20 | .08 | .25 | .27 | .49 | 1 | \-.67 | \-.14 | .38 | .51 |
| **3PAr** | \-.08 | \-.06 | \-.10 | \-.36 | \-.38 | \-.67 | 1 | .04 | \-.53 | \-.69 |
| **STL%** | .24 | .24 | \-.08 | .22 | \-.31 | \-.14 | .04 | 1 | \-.07 | \-.13 |
| **BLK%** | .04 | \-.36 | \-.20 | .14 | .43 | .38 | \-.53 | \-.07 | 1 | .66 |
| **TRB%** | .13 | \-.18 | \-.05 | .28 | .44 | .51 | \-.69 | \-.13 | .66 | 1 |

**Box & Whisker Data**

| Location | Value |
| :---: | :---: |
| Min (Lower Whisker) | \-6.068 |
| Q1 (25th Percentile) | \-1.3025 |
| Median (50th Percentile) | 0.2295 |
| Q3 (75th Percentile) | 2.075 |
| Upper Whisker | 6.192 |
| Outlier Cutoff | 7.141 |
| Outlier \#1 | 7.541 |
| Outlier \#2 | 8.395 |
| Max (Outlier \#3) | 10.312 |

Note: Outlier 1: Giannis Antetokounmpo; Outlier 2: Shai Gilgeous-Alexander; Max Outlier: Nikola Jokić

Data Preparation: There were no values that had to be added to a variable, and there were no invalid rows that had to be removed. 

Modeling: Variance Inflation Factors had to be checked to ensure that there was no hidden collinearity. Afterwards, the overall model coefficient of determination was collected and a Dominance Analysis was performed to see how much each variable contributes to the overall coefficient. An 80-20 train-test split was then conducted to split the dataset into a dataset for training the model and a smaller dataset for testing the model.

**Variance Inflation Factors**

| Stat | VIF |
| :---: | :---: |
| AST% | 3.58 |
| USG% | 2.82 |
| 3PAr | 2.79 |
| TRB% | 2.71 |
| FTr | 2.43 |
| BLK% | 2.18 |
| TOV% | 2.06 |
| TS% | 1.79 |
| STL% | 1.32 |

Note: All VIFs Less Than 5 Which Indicates That No Significant Collinearity is Present

**Overall Results**

| Concept | Values |
| :---: | :---: |
| Sample Size | 242 |
| Coefficient of Determination | 0.45 |
| Root Mean Square Error | 1.86 |
| Mean Absolute Error | 1.54 |

**Dominance Analysis**

| Rank | Stat | Share of R² | Share of all RAPM variance |
| ----- | :---: | :---: | :---: |
| 1 | AST% | 25.3% | 11.4% |
| 2 | STL% | 19.1% | 8.6% |
| 3 | TS% | 18.0% | 8.1% |
| 4 | USG% | 14.8% | 6.7% |
| 5 | \*TOV% | 8.4% | 3.8% |
| 6 | TRB% | 5.8% | 2.6% |
| 7 | FTr | 3.9% | 1.7% |
| 8 | 3PAr | 3.1% | 1.4% |
| 9 | BLK% | 1.6% | 0.7% |
|  | **Total** | **100%** | **45.0% (R² 0.450)** |

\*TOV% has a negative correlation to RAPM directionally when controlling for the other stats

**Training Results** 

| Concept | Values |
| :---: | :---: |
| Sample Size | 193 (80% of overall dataset) |
| Coefficient of Determination | 0.443 |
| Root Mean Square Error | 1.88 |
| Mean Absolute Error | 1.57 |

Evaluation: After obtaining the training results, the predictive validity of the model had to be determined. This was done through evaluating the test set of 49 players, performing the Leave-One-Out Cross Validation procedure, and performing the 10-fold Cross Validation procedure. The results go from about 44% of explained variance in the training set to around 40% among the test set and Cross-Validation results, which isn’t a practically significant drop-off. 

**Testing & Cross-Validation Results** 

| Method |  R² |
| :---: | :---: |
| Test Set (49 players) | 0.401 |
| Leave-One-Out Cross Validation | 0.396 |
| 10-fold Cross Validation | 0.393 |

Takeaways & Limitations: Although these advanced statistical features account for 45% of the overall variance in 5-Year Regularized Adjusted Plus Minus, that leaves 55% of the variance unexplained. A large reason for that is due to the nuance of the defensive side of the floor. The Offensive & Defensive RAPM variables were collected for each player as well, despite them not being included in the model. The same feature variables explained 61.5% of the variance in 5-year Offensive RAPM, whereas it only explained 30.7% for the Defensive equivalent. A hypothesis for these results is that steals and blocks are relatively rare occurrences compared to points, assists, and rebounds. The advanced stats do not account for rim deterrence, executing rotations correctly, navigating through screens, keeping a ball handler in front of you, etc. 

| Target | In-sample R² | Out-of-sample R² (leave-one-out) |
| :---- | :---: | :---: |
| Offensive RAPM | 0.615 | 0.578 |
| Defensive RAPM | 0.307 | 0.239 |
| Total RAPM | 0.450 | 0.396 |

There are also intangibles valuable on both sides of the floor, such as communication that the stat sheets cannot capture. Another component of the remaining variance being unexplained could simply be due to additional noise within the 5-Year RAPM. 45% of variance being explained is significant, nonetheless, and provides a framework to build off when determining the “science” of a positive contributor. 

Citations

Sill, J. (2010, March 6). *Improved NBA adjusted \+/- using regularization and out-of-sample testing* \[Paper presentation\]. MIT Sloan Sports Analytics Conference, Boston, MA, United States. [https\://sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing](https://sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing)

*nbarapm*. (n.d.). Retrieved October 27, 2026, from [https\://nbarapm.com](https://nbarapm.com)

Sports Reference LLC. (n.d.). *Basketball-Reference.com*. Retrieved October 27, 2026, from [https\://www\.basketball-reference.com](https://www.basketball-reference.com)
Repository Structure

```
rapm-advanced-stats/
├── README.md
├── run_all.py                     runs every step below in order
├── requirements.txt
├── data/
│   ├── DATA_DICTIONARY.md         every column, its units and its source
│   ├── raw/
│   │   ├── bbref_advanced_season_rows.csv   2,867 player-seasons (Basketball Reference)
│   │   └── nbarapm_5y_ending_2026.csv       5-Year RAPM, ORAPM, DRAPM for 249 players
│   └── processed/
│       └── model_data.csv         the 242 modeled players (built by step 1)
├── src/
│   ├── config.py                  paths, features, cutoffs and random seeds
│   ├── 01_build_dataset.py        5-season filter, minutes-weighted averages, cutoffs
│   ├── 02_explore.py              correlation matrix, box plot values, figures
│   ├── 03_model.py                VIFs, OLS coefficients, overall fit, diagnostics
│   ├── 04_dominance.py            exact Dominance Analysis (all 512 sub-models)
│   ├── 05_validation.py           80-20 split, Leave-One-Out CV, repeated 10-fold CV
│   ├── 06_offense_defense.py      the same model for Offensive and Defensive RAPM
│   └── verify.py                  checks every number in this README against a fresh run
├── results/                       CSV and text outputs, including verification_report.txt
├── figures/                       correlation matrix, box plot, X graphic
├── workbook/
│   └── RAPM_BoxScore_Dominance.xlsx   the same analysis with live Excel formulas
└── assets/fonts/                  fonts used in the figures (SIL Open Font License)
```

Reproducing the Results

Tested with Python 3.13 and the package versions in requirements.txt.

```
pip install -r requirements.txt
python run_all.py
```

The pipeline rebuilds the dataset from the raw files, reruns every analysis, redraws the figures and finishes with `src/verify.py`, which confirms that every number in this README (all 100 correlation matrix cells, the box plot values, VIFs, overall results, Dominance Analysis, training, test and cross-validation results, and the offense and defense results) matches the fresh run, and that the Excel workbook matches the same data and results. The report is saved to `results/verification_report.txt`. The 80-20 split uses random seed 42 and the repeated 10-fold Cross-Validation uses seed 0, so every run gives the same numbers.

The X graphic (`figures/x_post_graphic.png`) was designed separately and is not redrawn by the pipeline; its numbers come from the same results.

License

The code is released under the MIT License. The data belongs to its sources (Basketball Reference and nbarapm) and remains subject to their terms of use.
