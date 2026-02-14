# End-to-End Implementation Plan
## Project: Mapping How State Alcohol Policies Shift Teen Drinking and Alcohol‑Impaired Fatal Crashes Over Time
**Goal:** Build a reproducible, poster-ready causal analysis **and** a prediction tool that together answer:
1) **Causal:** Do state alcohol policy changes lead to measurable changes in teen drinking and alcohol‑impaired fatal crash outcomes?
2) **Predictive:** Given a state’s policy environment + context, can we forecast next-year teen drinking and alcohol‑impaired fatal crash rates, and identify which policy features are most predictive?

> **Key principle (judge-proof):** Prediction accuracy ≠ causality. The project is explicitly **two-track**:
> - **Track A: Causal evaluation** (event-study / DiD + validity checks)
> - **Track B: Predictive AI tool** (model competition + strict out-of-sample tests)

---

## Table of contents
1. Scope decisions and success criteria  
2. MVP dataset definition (finish-first)  
3. Expansion datasets (build the “big dataset” safely)  
4. Data engineering pipeline (ETL)  
5. Feature engineering and transformations  
6. Causal track (policy evaluation)  
7. Predictive track (AI model ladder)  
8. Deep learning module: policy text representation learning  
9. Robustness, falsification, sensitivity, and QA  
10. Dashboard + visuals (poster-ready)  
11. Reproducibility, documentation, and packaging  
12. Timeline (MVP → expansions)  
13. Risk register + fallbacks  
14. Final poster/report structure + judging strategy  
15. Checklist (done/ready)

---

# 1) Scope decisions and success criteria

## 1.1 Unit of analysis
- **Primary unit:** State-year panel (50 states; optionally include DC)  
- **Secondary unit:** State-survey-wave panel for YRBSS teen outcomes (often biennial)

## 1.2 Outcomes (you will report at least one “MVP outcome”)
- **MVP outcome (annual):** Alcohol‑impaired driving fatality measure (e.g., fatalities involving driver BAC ≥ 0.08 per 100k).  
- **Expanded outcome:** Teen drinking indicators (YRBSS state estimates).

## 1.3 Policy exposure definition
You will implement **both** of these (phased):
- **MVP policy exposure:** 1–2 APIS policy topics with clear effective dates (cleanest to assemble and validate).
- **Expanded policy exposure:** Composite policy environment (APS score) or multi-topic APIS merged panel.

## 1.4 Covariates
You will start with a minimal confounder set, then expand:
- **MVP covariates (must have):** income, unemployment, population, VMT (for crash models), basic demographics.
- **Expanded covariates:** political lean, urbanization, education, poverty/inequality, enforcement proxies, etc.

## 1.5 Deliverables (end products)
**Poster-ready outputs**
- Event-study plots (before/after with confidence intervals)
- “No pre-trend” diagnostic plots
- Map (choropleth) of estimated policy impact or predicted risk
- Main results table + robustness table

**Shareable tool**
- A simple dashboard (local or web) that:
  - Lets users choose a state & policy change scenario (or compare states)
  - Shows predicted outcome changes and uncertainty
  - Explains causal evidence vs predictive forecasting clearly

**Reproducibility package**
- Cleaned dataset(s) + data dictionary
- Scripts/notebooks that rebuild everything from raw sources
- Run instructions (1 command to rebuild figures)

## 1.6 Success criteria
You “win” if:
- MVP dataset is complete and reproducible
- Causal results show credible identification checks (pretrends flat; placebos null)
- Predictive models show honest out-of-sample performance
- Deep-learning module provides either:
  - improved forecast performance, OR
  - meaningful policy-text similarity insights (even if performance gains are small)

---

# 2) MVP dataset definition (finish-first)

## 2.1 MVP research question
**Do changes in a small set of alcohol policy features predict changes in alcohol‑impaired fatal crash outcomes after controlling for major confounders?**

## 2.2 MVP outcome data (annual, clean)
### Source: NHTSA FARS
- NHTSA FARS overview: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars  
- FARS contains a census of fatal traffic crashes with BAC information for involved drivers.
- **MVP outcome options (pick 1 primary + 1 secondary):**
  1. **Alcohol-impaired driving fatalities per 100k**
  2. **Alcohol-involved fatalities per 100k**
  3. **Alcohol-impaired crashes per 100k** (if you build crash-level aggregates)
- **Primary definition to prefer for clarity:** driver BAC ≥ 0.08 involved fatalities per 100k.

**Implementation approach**
- Decide the year range (example: 2000–2023).
- Either:
  - Use NHTSA downloadable FARS annual data files and aggregate to state-year, OR
  - Use a reputable government portal distribution (ensure consistent variables across years).

## 2.3 MVP policy data (effective dates + state variation)
### Source: NIAAA APIS (Alcohol Policy Information System)
- APIS home: https://alcoholpolicy.niaaa.nih.gov/
- About APIS: https://alcoholpolicy.niaaa.nih.gov/about-apis
- Download data: https://alcoholpolicy.niaaa.nih.gov/policy-topics/download-policy-topic-data

**Pick MVP policy topics (choose 1–2)**
Select policies with:
- widespread adoption/variation
- clear effective dates
- plausible mechanism for alcohol-impaired driving outcomes

Examples (final choice should be pragmatic based on APIS availability):
- Alcohol excise tax (beer/wine/spirits) changes
- Underage access restriction changes (e.g., penalties / compliance checks)
- Hours/days of sale restrictions
- Retail availability (e.g., off-premise restrictions)

**MVP policy encoding**
- Convert policy to numeric time-varying features:
  - binary indicator: in effect (0/1)
  - intensity: tax rate, allowed hours, etc. if available
  - event-time “adoption year” if binary

## 2.4 MVP covariates (minimum confounder set)
You can assemble these quickly and they defend causal plausibility:
- **Income / personal income per capita** (BEA)  
  https://www.bea.gov/data/income-saving/personal-income-by-state
- **Unemployment rate** (BLS LAUS)  
  https://www.bls.gov/lau/
- **Vehicle miles traveled (VMT)** (FHWA Highway Statistics tables)  
  Example table: https://www.fhwa.dot.gov/policyinformation/statistics/2023/vm2.cfm
- **Population** (Census/ACS or other consistent source)

## 2.5 MVP dataset schema (state-year)
Create a single “gold” panel table with one row per **state-year**.

**Core identifiers**
- `state_fips` (2-digit)
- `state_abbrev`
- `year`

**Outcome**
- `fars_alcohol_impaired_fatalities_per_100k` (primary)
- optional: `fars_alcohol_involved_fatalities_per_100k`

**Policy**
- `policy_tax_beer` (or selected policy numeric)
- `policy_hours_of_sale` (if used)
- `policy_underage_access` (if used)
- `policy_change_indicator` (0/1 “changed this year”)

**Covariates**
- `population`
- `income_pc_real`
- `unemployment_rate`
- `vmt_per_capita`
- optional: `poverty_rate`, `urban_share`, etc. (later)

**Metadata**
- `source_versions` (json-like string: file versions / download dates)
- `notes` (for edge cases)

---

# 3) Expansion datasets (build the “big dataset” safely)

## 3.1 Teen drinking outcomes (YRBSS)
### Source: CDC YRBSS
- YRBSS pages: https://www.cdc.gov/yrbs/
- YRBSS results hub: https://www.cdc.gov/yrbs/results/index.html
- Data user guides (PDFs) contain definitions and coding rules.

**Reality check**
- Many YRBSS outputs are **biennial**.
- Not every state participates every wave.
- Treat teen outcomes as a separate panel keyed by `state` + `survey_year` (wave).

**Teen outcome variables (examples)**
- current alcohol use (% past 30 days)
- binge drinking (% past 30 days; definition differs by sex in some docs)
- age of first drink (if available / comparable)

**Integration strategy**
- Keep two “analysis datasets”:
  1) `panel_state_year_mvp` (annual; crash-focused)
  2) `panel_state_wave_yrbss` (teen-focused; wave-based)
- You can still align policy effective dates to waves by mapping wave year to policy status.

## 3.2 Composite policy environment (APS)
APS is a composite index of alcohol policy restrictiveness used in peer-reviewed work.
- Example APS policy paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7024811/

**Integration strategy**
- Add `aps_score_total` as a continuous treatment.
- Keep APIS policy-topic features as the “explainable breakdown.”

## 3.3 Political context
- Presidential vote share (Dem/Rep) from a consistent dataset (e.g., MIT Election Lab).  
  https://electionlab.mit.edu/data

Features:
- `dem_vote_share`
- `party_control_governor` (if you add)
- `legislature_control` (if you add)

## 3.4 Expanded socioeconomic + demographic covariates (recommended)
- poverty rate, gini, education attainment, % rural, age distribution
- healthcare capacity proxies (optional)
- enforcement proxies (harder; optional)

**Rule for expansion:** add covariates only if you can defend:
- consistent coverage across states and years
- stable definitions
- minimal missingness

---

# 4) Data engineering pipeline (ETL)

## 4.1 Repository structure (recommended)
```
project/
  README.md
  environment.yml or requirements.txt
  Makefile
  data/
    raw/               # downloaded as-is, never edited
    intermediate/      # cleaned but not merged
    processed/         # final analysis datasets
    dictionaries/      # data dictionaries + codebooks
  src/
    config.py          # paths, years, state list
    download/          # scripts to fetch raw data
    clean/             # scripts to clean each source
    merge/             # build panels
    features/          # feature engineering
    models/            # causal + predictive models
    viz/               # plots
    dashboard/         # app
  notebooks/
    01_data_audit.ipynb
    02_mvp_analysis.ipynb
    03_ml_models.ipynb
    04_text_embeddings.ipynb
  outputs/
    figures/
    tables/
    dashboard_build/
  logs/
  tests/
```
**Golden rule:** raw data stays raw; all transformations happen in scripts.

## 4.2 Data governance
- Maintain a `data_manifest.csv` with columns:
  - dataset_name, url, download_date, file_hash, license/terms, notes
- Store `SHA256` hashes for reproducibility.

## 4.3 Download & versioning
**Approach**
- For each dataset, write `src/download/get_<dataset>.py` that:
  - downloads file(s)
  - saves to `data/raw/<dataset>/`
  - writes metadata entry into manifest
  - computes file hash

## 4.4 Cleaning scripts
Write one script per data source:
- `clean_fars.py` → outputs `data/intermediate/fars_state_year.csv`
- `clean_apis.py` → outputs `data/intermediate/apis_policy_state_year.csv`
- `clean_bea.py` → outputs `data/intermediate/bea_income_state_year.csv`
- `clean_bls.py` → outputs `data/intermediate/bls_unemp_state_year.csv`
- `clean_fhwa.py` → outputs `data/intermediate/fhwa_vmt_state_year.csv`
- `clean_yrbss.py` (expansion) → outputs `data/intermediate/yrbss_state_wave.csv`

Each cleaning script must:
1) standardize state identifiers (FIPS + abbreviation)
2) standardize year
3) label units and definitions
4) log missingness summary

## 4.5 Merging
- `merge_mvp_panel.py` merges intermediate tables to create:
  - `data/processed/panel_state_year_mvp.csv`
- `merge_expanded_panel.py` merges additional features for:
  - `data/processed/panel_state_year_big.csv`
  - `data/processed/panel_state_wave_yrbss.csv`

**Merge QA**
- assert exactly 50 states × years rows (minus known exceptions)
- print missingness heatmap and summary table
- verify no duplicate state-year keys

---

# 5) Feature engineering and transformations

## 5.1 Standardize rates and scaling
- Convert outcomes to per 100k population.
- Convert VMT to per capita (or per 1,000 pop).
- Standardize continuous covariates (z-score) for ML models (not necessarily for DiD).

## 5.2 Inflation adjustment (if using tax rates)
If your policy includes dollar tax rates:
- Convert to **real dollars** using CPI (document source).

## 5.3 Policy variable encoding
For each policy topic, create:
- `policy_in_effect` (0/1)
- `policy_intensity` (numeric, if defined)
- `policy_change_year` (0/1)
- `event_time` relative to adoption year (for event study)

## 5.4 Lag features (predictive track)
Create lagged outcomes and covariates:
- `y_tminus1`, `y_tminus2`
- `policy_tminus1` etc.
**Important:** For causal track, treat lags carefully (don’t induce post-treatment bias).

## 5.5 Handling missingness
- MVP goal: minimal missingness.
- For expanded covariates:
  - prefer sources with complete coverage
  - for small gaps, consider:
    - linear interpolation (document it)
    - missing indicators
  - never silently drop many state-years without documenting impact.

---

# 6) Track A — Causal evaluation (event-study / DiD)

## 6.1 Identify causal estimand
- **Average treatment effect on treated (ATT)** for policy adoption/change.

## 6.2 Baseline model (DiD)
State and year fixed effects:
- Outcome: crash rate (or teen drinking)
- Treatment: policy status or APS score
- Covariates: major confounders (income, unemployment, VMT, etc.)
- Cluster SE by state

## 6.3 Event study
Create lead/lag indicators relative to adoption year:
- leads: -5, -4, -3, -2 (pre)
- lags: 0, +1, +2, +3, +4, +5 (post)
- reference period: -1 or -2

**Deliverable plots**
- coefficient plot with CIs
- highlight pretrend coefficients ~ 0 (validity)

## 6.4 Staggered adoption warning (must address)
Many states adopt at different times; classic TWFE can be biased with heterogeneous treatment effects.
**Plan**
- Use a staggered-adoption robust estimator (document choice and why).
- At minimum, include sensitivity checks where the estimator changes.

## 6.5 Placebos / falsification checks
- **Fake policy date placebo:** assign adoption year randomly or shift by +5 years
- **Never-treated placebo:** apply “treatment” to states that didn’t change
- **Outcome placebo:** use an outcome that should not plausibly respond (if available)

## 6.6 Sensitivity tests
- vary time window (e.g., ±3 vs ±5 years around adoption)
- add/remove covariates
- exclude one state at a time (leave-one-out)
- exclude early adopter states

## 6.7 Interpretation rules
- Causal claims are based on:
  - clean pretrends
  - stable post effects
  - placebos null
  - robust across reasonable covariate specs

---

# 7) Track B — Predictive AI modeling ladder

## 7.1 Define prediction tasks
You will have two main forecasting tasks:
1) Predict `crash_rate_{t+1}` from features at time `t`
2) Predict `teen_drinking_{t+1 or next_wave}` from features at time `t`

## 7.2 Train/test splits (avoid leakage)
**Preferred split methods**
- **Out-of-time split:** train on 2000–2016, validate 2017–2019, test 2020–2023
- **Leave-one-state-out (LOSO):** for robustness of generalization
- **Rolling-origin evaluation:** train up to year t, predict t+1 repeatedly

## 7.3 Model ladder (competition)
### Model 1: Linear regression (baseline)
- elastic net / ridge to handle multicollinearity
- interpret coefficients cautiously (predictive, not causal)

### Model 2: Random forest / gradient boosting
- handle nonlinearities + interactions
- tune with cross-validation that respects time ordering

### Model 3: Panel-aware baseline (optional)
- include state embeddings as fixed effects for prediction
- or use state-level intercepts

### Model 4: Deep learning (only after MVP works)
- small, regularized model that uses:
  - covariates
  - policy topic features
  - policy text embeddings (next section)

## 7.4 Metrics
- MAE, RMSE (core)
- MAPE (optional, if rates not near 0)
- calibration plot (predicted vs observed)

## 7.5 Feature importance (predictive)
- permutation importance
- SHAP (carefully; explain it as “what the model used for prediction”)
- stability check: importance should be similar across folds

**Critical wording**
- “Important for prediction” ≠ “policy causes outcome.”

---

# 8) Deep learning module — Policy text representation learning

## 8.1 Why text learning (justification)
Tabular panel is small; generic deep nets often overfit.
Text embeddings can add signal by learning similarity between policy regimes beyond hand-coded categories.

## 8.2 Build a policy text corpus
Sources:
- APIS policy topic descriptions / statutory summaries where available
- Optional: external legal summaries (must be consistent and documented)

**Corpus structure**
- `state`
- `year`
- `policy_topic`
- `policy_text_raw`
- `policy_text_source_url`
- `policy_text_version_date`

## 8.3 Preprocessing (fixed protocol)
- normalize whitespace
- remove boilerplate consistently
- keep citations/section numbers only if stable across states
- store both raw and cleaned text

## 8.4 Embedding strategy (unbiased)
**Preferred**: embedding models that produce deterministic vectors from text.
- Document model name/version and parameters.
- Freeze embeddings before model fitting.

**Avoid bias pitfalls**
- Do not generate “opinionated summaries” like “this is strict” unless you pre-register a fixed rubric.
- If you do summarization:
  - use a fixed prompt template
  - store summaries as a derived artifact
  - run audits on a random sample for consistency

## 8.5 Deep model design (minimal, regularized)
Inputs:
- covariates (scaled)
- policy topic numeric features
- policy text embeddings (vector)

Architecture:
- simple MLP with dropout + weight decay
- multi-task head (optional): crash + teen drinking

Training:
- early stopping
- out-of-time validation
- compare to non-text baselines

Outputs:
- performance deltas vs baseline
- embedding similarity map (states clustered by policy text similarity)

## 8.6 Explainability for judges
- show example states with “similar policy text embeddings”
- show whether embedding features improve forecast error
- interpret cautiously; do not claim embeddings “prove” strictness

---

# 9) Robustness, falsification, sensitivity, and QA (both tracks)

## 9.1 Data QA checklist
- Missingness report per variable
- Outlier scan (winsorize only if justified)
- Verify consistent state-year keys
- Compare national totals to official reports (sanity checks)

## 9.2 Statistical QA
- Pretrend tests for event study
- Placebo distribution (histogram of placebo effects)
- Sensitivity to excluding each state

## 9.3 Model QA
- leakage audit: ensure target year not included in features
- hyperparameter tuning uses only training/validation (not test)
- stable feature importance across folds

## 9.4 Reproducibility QA
- 1-command rebuild (`make all`)
- fixed random seeds
- environment lock file
- all intermediate outputs stored with checksums

---

# 10) Dashboard + visuals (poster-ready)

## 10.1 Figure set (minimum)
1) Event-study plot for crash outcome
2) Event-study plot for teen outcome (if available)
3) Choropleth map: estimated effect or predicted risk change
4) Model comparison bar chart (MAE/RMSE) for prediction ladder
5) Placebo plot (null distribution) to prove credibility

## 10.2 Dashboard requirements
- Choose state, year, policy scenario (or compare pre/post)
- Show:
  - predicted next-year crash rate + uncertainty band
  - causal effect estimate summary (from Track A)
  - “what this means” in plain English
- Provide citations and definitions in a “Method” tab

## 10.3 Implementation options
- Streamlit (fast), Dash, or static HTML + precomputed JSON.
- Build `outputs/dashboard_build/` that can be zipped and shared.

---

# 11) Reproducibility, documentation, packaging

## 11.1 Data dictionary
For each final dataset:
- variable name
- definition
- unit
- source
- transformation notes
- missingness rate

## 11.2 Method log
Maintain a `METHOD_LOG.md` with:
- every major decision (why policies chosen, year ranges)
- every exclusion
- model configurations

## 11.3 Poster and report
Create:
- `poster/` folder with figures
- `report.md` or `report.pdf` (optional)
- `appendix/` containing robustness outputs

---

# 12) Timeline (aggressive but realistic)

## Week 1 — MVP build
- Day 1–2: download + clean FARS → state-year outcomes
- Day 2–3: download + clean APIS policy topics
- Day 3–4: add BEA + BLS + FHWA covariates
- Day 5: merge MVP panel + QA
- Day 6–7: baseline DiD + event study + 3 key plots

## Week 2 — Predictive ladder + robustness
- Build linear + RF models with strict splits
- Add placebo tests, sensitivity checks
- Create model comparison visuals

## Week 3 — Expansion
- Add YRBSS teen outcomes
- Add APS composite score (if available)
- Add political features
- Re-run key analyses

## Week 4 — Deep text module + dashboard polish
- build policy text corpus
- embeddings + deep model
- dashboard + final writeup

---

# 13) Risk register + fallbacks

## Risk: Teen drinking data missing/patchy at state level
**Fallback:** keep teen outcomes wave-based; focus MVP on crash outcomes.

## Risk: Policy text not consistently available
**Fallback:** deep learning uses numeric policy-topic features + APS only; text module becomes optional appendix.

## Risk: Overfitting with deep models
**Fallback:** keep deep model small; treat embeddings primarily as visualization/cluster tool if performance gains are minimal.

## Risk: Confounding / messy adoption timing
**Fallback:** restrict to a single clear policy change; or use APS continuous changes; tighten event windows.

---

# 14) Final poster/report structure (judge strategy)

## Poster sections
1) Motivation + why policy evaluation matters  
2) Data sources + privacy/ethics  
3) MVP design (state-year panel)  
4) Causal results (event study + diagnostics)  
5) Predictive tool + model comparison  
6) Deep learning (policy text embeddings)  
7) Limitations + what’s next  
8) Reproducibility QR code/link (if allowed)

## “Judge-proof” talking points
- “We separated causal inference from prediction to avoid confusing correlation with policy impact.”
- “We validated identification with pretrends and placebos.”
- “We evaluated AI models honestly with out-of-sample forecasting.”

---

# 15) Completion checklist

## MVP complete
- [ ] FARS outcome aggregated to state-year
- [ ] APIS policy topic(s) cleaned + aligned
- [ ] BEA/BLS/FHWA covariates cleaned
- [ ] Merged MVP panel with QA report
- [ ] Event study plot + pretrend check
- [ ] Placebo test results
- [ ] Baseline predictive models + out-of-sample metrics

## Expanded complete
- [ ] YRBSS teen panel integrated (wave-based)
- [ ] APS composite added (if used)
- [ ] political + demographic covariates added
- [ ] re-run core analyses

## Deep text module complete
- [ ] policy text corpus saved + versioned
- [ ] embeddings computed + stored
- [ ] deep model trained + tested
- [ ] embedding similarity visualization

## Final delivery
- [ ] poster figures exported (PNG + PDF)
- [ ] results table exported (CSV + pretty table)
- [ ] dashboard runnable locally
- [ ] full reproducibility: `make all` works

---

## Appendix A — Concrete “MVP policy topic” selection procedure (fast + defensible)
1) In APIS, list candidate topics and verify:
   - available for your year range
   - has effective dates
   - varies across states
2) For each candidate, compute:
   - number of adoption/changes across years (more = more power)
   - missingness rate
3) Pick 1–2 topics with best coverage + plausible mechanism.
4) Pre-register: “These are the primary treatments; others are exploratory.”

---

## Appendix B — Minimal software stack
- Python 3.11+
- pandas, numpy
- statsmodels or linearmodels (fixed effects)
- scikit-learn (ML)
- matplotlib (plots)
- optional: streamlit (dashboard)
- optional: sentence-transformers (text embeddings)

---

**End of plan.**
