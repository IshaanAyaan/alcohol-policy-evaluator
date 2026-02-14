# Executive Project Summary

## Objective
This project built a reproducible state-level evidence system to evaluate whether changes in alcohol policy are followed by measurable changes in alcohol-related traffic fatality outcomes and teen drinking behavior. The implementation intentionally separates causal inference from forecasting so correlation and causation are not conflated.

## Data Coverage
The primary panel contains **1071 state-year rows** across **51 jurisdictions** (50 states + DC) from **2003 to 2023**.
Teen wave data contains **571 rows** with **418 observed** values for current alcohol use and explicit missingness flags for non-participating waves.
Policy features include APIS Beer tax (topic 30), Sunday off-premise sales change events (topic 28), and Underage purchase change events (topic 43), assembled with an APIS-first parser and fallback pathways.

## Causal Findings
Event-study models were estimated with state and year fixed effects, covariate adjustment, and clustered standard errors. Primary outcome was alcohol-impaired fatality rate per 100k.
Average post-policy dynamic coefficient (event time >= 0) for the primary outcome: **-0.348**.
- rate_impaired_per100k: R²=0.776, pretrend p-value=0.125, placebo mean=0.286, placebo sd=0.402
- rate_alcohol_involved_per100k: R²=0.873, pretrend p-value=0.203, placebo mean=0.280, placebo sd=0.114
Placebo distributions were generated to check whether estimated effects are centered near zero under randomized policy timing.

## Predictive Performance
A model ladder (ElasticNet, RandomForest, HistGradientBoosting, MLP with policy text embeddings) was evaluated using out-of-time splits.
Best held-out models by task:
- crash_rate_next_year: RandomForestRegressor with test RMSE=0.806, MAE=0.553, R²=0.546
- teen_current_use_next_wave: ElasticNet with test RMSE=5.619, MAE=4.672, R²=-1.229
This predictive output is used for scenario exploration and prioritization, not causal attribution.

## Geographic Pattern Highlights
Largest estimated improvements in impaired fatality rate (latest year vs earliest year):
- MS: delta -17.849 per 100k
- AZ: delta -14.347 per 100k
- AL: delta -12.851 per 100k
- WY: delta -12.015 per 100k
- TN: delta -11.217 per 100k
Largest estimated worsenings in impaired fatality rate (latest year vs earliest year):
- NE: delta -1.410 per 100k
- CT: delta -1.659 per 100k
- NH: delta -1.825 per 100k
- ME: delta -2.112 per 100k
- OH: delta -2.361 per 100k

## Limitations
Teen outcome coverage is wave-based and state participation is uneven in later cycles. APIS access required an HTTP fallback parser due direct endpoint blocking. Some policy dimensions and enforcement intensity are not fully observed.
Accordingly, causal findings should be interpreted as quasi-experimental estimates under tested assumptions, not as definitive policy law effects in every context.

## Reproducibility
The pipeline is command-driven (`make data`, `make features`, `make causal`, `make predict`, `make text`, `make dashboard`, `make report`, `make all`) and produces versioned processed artifacts and figure outputs for poster and dashboard use.
