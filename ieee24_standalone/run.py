"""Run all three masters against one immutable SQLite scenario bank."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .model import load_study
from .solver import Master_CVAR, Master_DET, Master_DRO_CVAR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="Fixed SQLite scenario bank")
    parser.add_argument("--config", required=True, type=Path, help="Task, hazard and search settings JSON")
    parser.add_argument("--case", type=Path, help="MATPOWER case (defaults to packaged IEEE RTS-24)")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    args = parser.parse_args()
    study = load_study(args.database, args.config, args.case)
    det = Master_DET(study)
    cvar = Master_CVAR(study, det)
    dro = Master_DRO_CVAR(study, det)
    args.output.mkdir(parents=True, exist_ok=True)
    results = [det, cvar, dro]
    (args.output / "schedules.json").write_text(json.dumps({
        "input_database": str(args.database), "input_config": str(args.config),
        "scenario_ids": study.scenario_ids, "probabilities": study.probabilities.tolist(),
        "results": [asdict(x) for x in results],
    }, indent=2) + "\n", encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "utility", "deterministic_loss_mw", "mean_loss_mw", "var_loss_mw",
            "cvar_loss_mw", "dro_cvar_loss_mw", "max_loss_mw", "candidates_evaluated",
            "slave_calls", "runtime_s",
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({key: getattr(result, key) for key in writer.fieldnames})
    print(json.dumps({r.method: {
        "starts": r.starts, "utility": r.utility, "mean_loss_mw": r.mean_loss_mw,
        "cvar_loss_mw": r.cvar_loss_mw, "dro_cvar_loss_mw": r.dro_cvar_loss_mw,
        "slave_calls": r.slave_calls,
    } for r in results}, indent=2))


if __name__ == "__main__":
    main()
