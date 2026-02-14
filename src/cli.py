"""CLI entrypoint for the alcohol policy impact tracker."""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from src.download import apis, covariates_fred, fars, fhwa, manifest, teen
from src.features import engineer
from src.merge import build_panels
from src.models import causal, predictive, text_embeddings
from src.report import executive_summary, full_paper
from src.viz import generate as viz_generate


def cmd_data() -> None:
    print("[data] APIS")
    print(apis.run())
    print("[data] FARS")
    print(fars.run())
    print("[data] FRED covariates")
    print(covariates_fred.run())
    print("[data] FHWA VMT")
    print(fhwa.run())
    print("[data] Teen outcomes")
    print(teen.run())
    print("[data] Build panels")
    print(build_panels.run())
    print("[data] Raw data manifest")
    print(manifest.run())


def cmd_features() -> None:
    print("[features]")
    print(engineer.run())


def cmd_text() -> None:
    print("[text embeddings]")
    print(text_embeddings.run())


def cmd_causal() -> None:
    print("[causal]")
    print(causal.run())


def cmd_predict() -> None:
    print("[predictive]")
    print(predictive.run())


def cmd_dashboard() -> None:
    print("[dashboard assets]")
    print(viz_generate.run())
    print("Run dashboard with: streamlit run src/dashboard/app.py")


def cmd_report() -> None:
    print("[report figures]")
    print(viz_generate.run())
    print("[executive summary]")
    print(executive_summary.run())
    print("[full paper]")
    print(full_paper.run())


def cmd_all() -> None:
    cmd_data()
    cmd_features()
    cmd_text()
    cmd_causal()
    cmd_predict()
    cmd_dashboard()
    cmd_report()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alcohol Policy Impact Tracker CLI")
    parser.add_argument(
        "command",
        choices=["data", "features", "causal", "predict", "text", "dashboard", "report", "all"],
        help="Command to execute",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands: Dict[str, Callable[[], None]] = {
        "data": cmd_data,
        "features": cmd_features,
        "causal": cmd_causal,
        "predict": cmd_predict,
        "text": cmd_text,
        "dashboard": cmd_dashboard,
        "report": cmd_report,
        "all": cmd_all,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
