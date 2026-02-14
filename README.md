# Alcohol Policy Evaluator

Think of each U.S. state like a different city in one giant strategy game.
Each city can change a few rules:
- how expensive alcohol is (beer tax),
- when stores can sell (Sunday off-premise rules),
- and how strict underage purchase rules are.

This project asks a simple but important question:

**When a city changes its rules, does the scoreboard actually change?**

Our scoreboard has two big numbers:
1. Alcohol-related fatal crash rates
2. Teen alcohol use rates

## The Project in One Metaphor
This system is a **weather station for policy**.
- Policy changes are the wind and pressure changes.
- Crash and teen outcomes are the weather we observe.
- Causal models test whether a storm likely came *after* the pressure changed.
- Predictive models forecast tomorrow's weather from today's map.

So we do both:
- **Cause lane:** "Did policy shifts likely move outcomes?"
- **Prediction lane:** "Which model best predicts next outcomes?"

## What This Repo Produces
From raw public data, it builds:
- a 2003-2023 state-year panel (50 states + DC),
- causal event-study outputs,
- predictive model benchmarks,
- policy-text embeddings,
- poster-ready figures,
- dashboard artifacts,
- reports including a kid-friendly full paper.

## Quickstart
```bash
make setup
make all
```

Or use the CLI directly:
```bash
python -m src.cli all
```

## Main Commands
```bash
make setup      # install dependencies
make data       # download + assemble raw/intermediate/processed data
make features   # feature engineering
make causal     # causal analysis outputs
make predict    # predictive model training + eval tables
make text       # policy text embedding pipeline
make dashboard  # prepare dashboard artifacts
make report     # executive + full paper reports
make all        # run everything end-to-end
```

## Key Output Files
- `data/processed/panel_state_year.parquet`
- `data/processed/panel_state_wave_teen.parquet`
- `data/processed/policy_text_state_year.parquet`
- `data/processed/causal_effects_eventstudy.csv`
- `data/processed/model_eval_summary.csv`
- `outputs/figures/`
- `outputs/tables/`
- `outputs/reports/executive_summary.md`
- `outputs/reports/full_paper_kid_friendly.md`

## Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Reproducibility Notes
- APIS is accessed via `r.jina.ai` as an HTTP access helper.
- Manual APIS fallbacks can be placed in `data/raw/apis/manual/`.
- Optional R staggered DiD runs when `did` is available.

## Interpretation Guardrails
- Causal estimates are from identification assumptions and diagnostics.
- Prediction accuracy is not proof of causality.
- Teen coverage varies by state/year; missingness is explicitly tracked.

