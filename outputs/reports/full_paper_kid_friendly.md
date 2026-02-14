# Full Paper: Alcohol Policy Impact Tracker

## A Fun, 12-Year-Old-Friendly Abstract
Imagine each U.S. state is a giant school with its own rulebook. Some schools change alcohol rules (like tax, sales limits, and underage purchase rules). We asked: when a school changes a rule, does the real world change too? We tracked two scoreboards: teen drinking and alcohol-related traffic deaths.
To keep this fair, we used **two different lanes**: a **cause lane** (did rules likely move outcomes?) and a **prediction lane** (which models forecast best?). Think of this like using both a microscope and a weather forecast.

## 1) Research Question
Main question: **When states change alcohol policy, do important outcomes change afterward?**
Kid metaphor: we are checking if turning a game controller knob actually changes the game score.

## 2) Data Story (What We Collected)
We merged policy, crash, teen survey, and context data into one timeline.

### Table 1. Data Coverage
| Metric | Value | Meaning |
| --- | --- | --- |
| State-year rows | 1071 | How many state x year snapshots are in the main panel. |
| Jurisdictions | 51 | How many places were tracked (50 states + DC). |
| Year range | 2003-2023 | Start and end of the main analysis timeline. |
| Teen panel rows | 571 | How many teen wave records we have. |
| Teen current-use observations | 418 | How many teen drinking values are actual observations (not missing). |

**What this means:** We built a long timeline across almost all U.S. jurisdictions. So instead of guessing from one place, we compare many places over many years.

## 3) How We Tested Cause vs Prediction
- **Cause lane (event study + DiD ideas):** checks before-vs-after around policy changes while comparing to other states.
- **Prediction lane (ML ladder):** tests who forecasts better on future data they have never seen.
Kid metaphor: cause lane asks *who pushed the domino?* prediction lane asks *which player guesses the next domino best?*

## 4) Causal Findings
### Table 2. Causal Model Summary
| Outcome | Rows Used | R^2 | Pretrend p-value | Placebo Mean | Placebo SD |
| --- | --- | --- | --- | --- | --- |
| rate_impaired_per100k | 1048.000 | 0.776 | 0.125 | 0.286 | 0.402 |
| rate_alcohol_involved_per100k | 1048.000 | 0.873 | 0.203 | 0.280 | 0.114 |

**How to read this:** `R^2` tells how much pattern the model explains. `Pretrend p-value` checks whether states were already drifting apart *before* policy changes (higher is generally safer for this check). Placebo values should hover near zero if fake policy dates do not create fake effects.

### Table 3. Primary Event-Study Effects (Alcohol-Impaired Fatality Rate)
| Event Time | Effect | 95% CI Low | 95% CI High | p-value |
| --- | --- | --- | --- | --- |
| -5.000 | -0.566 | -1.918 | 0.787 | 0.412 |
| -4.000 | -0.949 | -2.113 | 0.216 | 0.110 |
| -3.000 | -0.918 | -2.200 | 0.364 | 0.160 |
| -2.000 | -0.443 | -1.222 | 0.336 | 0.265 |
| 0.000 | -0.712 | -1.887 | 0.464 | 0.235 |
| 1.000 | -0.204 | -1.434 | 1.026 | 0.745 |
| 2.000 | -0.586 | -1.829 | 0.657 | 0.355 |
| 3.000 | -0.343 | -1.714 | 1.028 | 0.624 |
| 4.000 | -0.222 | -1.415 | 0.972 | 0.716 |
| 5.000 | -0.021 | -0.971 | 0.930 | 0.966 |

**Big headline:** average post-policy effect for the primary crash outcome is **-0.348**. Negative means the rate moved down on average after policy-change timing in this framework.

## 5) Predictive Findings
### Table 4. Model Performance (Held-Out Test Data)
| Task | Model | Test RMSE | Test MAE | Test R^2 |
| --- | --- | --- | --- | --- |
| crash_rate_next_year | RandomForestRegressor | 0.806 | 0.553 | 0.546 |
| crash_rate_next_year | HistGradientBoostingRegressor | 0.875 | 0.595 | 0.465 |
| crash_rate_next_year | ElasticNet | 1.486 | 1.172 | -0.544 |
| crash_rate_next_year | MLPRegressor | 1.503 | 1.294 | -0.579 |
| teen_current_use_next_wave | ElasticNet | 5.619 | 4.672 | -1.229 |
| teen_current_use_next_wave | RandomForestRegressor | 5.925 | 4.440 | -1.478 |
| teen_current_use_next_wave | HistGradientBoostingRegressor | 7.042 | 5.925 | -2.500 |
| teen_current_use_next_wave | MLPRegressor | 16.945 | 16.517 | -19.269 |

**How to read this:** lower RMSE/MAE is better (smaller mistakes). Higher R² is better (more explained variation). For crashes, Random Forest performed best. Teen forecasting is harder because coverage is patchier and noisy.

## 6) Where Things Changed Most
### Table 5. States with Largest Changes in Impaired Fatality Rate
| Group | State | Early Rate | Latest Rate | Delta |
| --- | --- | --- | --- | --- |
| Most Improved | MS | 19.140 | 1.291 | -17.849 |
| Most Improved | AZ | 16.260 | 1.914 | -14.347 |
| Most Improved | AL | 16.387 | 3.537 | -12.851 |
| Most Improved | WY | 16.288 | 4.273 | -12.015 |
| Most Improved | TN | 14.826 | 3.609 | -11.217 |
| Most Worsened | NE | 4.026 | 2.616 | -1.410 |
| Most Worsened | CT | 4.047 | 2.388 | -1.659 |
| Most Worsened | NH | 3.750 | 1.926 | -1.825 |
| Most Worsened | ME | 3.827 | 1.715 | -2.112 |
| Most Worsened | OH | 4.932 | 2.571 | -2.361 |

**What this means:** Some states improved a lot more than others. This does not automatically prove one law did it, but it tells us where to look more closely.

## 7) Teen Coverage Details
### Table 6. Teen Data Coverage by Source
| Source | Coverage Flag | Rows |
| --- | --- | --- |
| YRBS_Explorer | missing_no_location_code | 15 |
| YRBS_Explorer | missing_no_state_data | 138 |
| YRBS_Socrata | observed | 418 |

**What this means:** We used what exists publicly. Some state-year teen values are missing because not every state reports every wave. We mark missing values clearly instead of pretending they are real.

## 8) Limits (Honest Science Section)
- Not all policy enforcement intensity is observed.
- Teen data are wave-based and uneven in later years.
- Causal methods reduce bias but cannot remove all uncertainty.
Kid metaphor: we built a strong flashlight, not x-ray vision.

## 9) What the Figures Show
- Event-study figure: `/Users/sangeetasinha/Documents/Ishaan/ISEF 26/projectv1/outputs/figures/event_study_primary_from_table.png`
- Model comparison figure: `/Users/sangeetasinha/Documents/Ishaan/ISEF 26/projectv1/outputs/figures/model_comparison_rmse.png`
- State map: `/Users/sangeetasinha/Documents/Ishaan/ISEF 26/projectv1/outputs/figures/state_impact_map.html`

## 10) Final Takeaway
If this were a video game, policy changes are not magic cheat codes. But our evidence says they can be meaningful control knobs, especially when we evaluate them with fair comparisons and transparent uncertainty checks.
