"""Run N-1 contingency analysis on the IEEE24 example system.

Execute from the repository root:
    python demos/run_ieee24_contingency_analysis.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pra_psa.core.contingencies import build_n1_contingencies
from pra_psa.contingency_analysis import run_contingency_analysis
from pra_psa.data import load_system
from pra_psa.simulation.lodf import screen_contingencies_lodf
from pra_psa.risk import UniformContingencyModel, aggregate_risk

DATA_DIR = ROOT / "data" / "powersystems"
OUT_DIR = ROOT / "outputs" / "ieee24_contingency"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    system = load_system(DATA_DIR, "IEEE24", as_pandapower=True)
    net = system.net
    loads = system.load_time_series.iloc[:24]

    contingencies = build_n1_contingencies(net, include=("line", "trafo", "gen"))

    print(f"Loaded {system.system_name}: " 
          f"{len(system.bus)} buses, {len(system.branch)} branches"          )
    print(f"Running {len(contingencies)} N-1 contingencies "
          f"over {len(loads)} operating points")

    try:
        lodf_ranking = screen_contingencies_lodf(net, contingencies, loading_threshold=100.0, )
        lodf_ranking.to_csv(OUT_DIR / "lodf_screening.csv", index=False)
        print("\nTop LODF-screened contingencies:")
        print(lodf_ranking.head(10).to_string(index=False))
    except Exception as exc:
        print(f"\nLODF screening skipped: {exc}")

    analysis = run_contingency_analysis( net,  operating_points=loads, contingencies=contingencies,  pf_mode="ac",
                                         include_base_case=True,   loading_threshold=100.0,
                                         severity_metric="overload_excess",   progress=True, )

    probabilities = UniformContingencyModel( total_contingency_probability=0.01).get_probabilities(contingencies)

    risk = aggregate_risk(analysis.summary,
                          probabilities=probabilities,
                          severity_col="severity",
                          time_col="operating_point_id",
                          contingency_col="contingency_id",
                          base_case_id="base_case", )

    analysis.to_csv(OUT_DIR)
    risk.to_csv(OUT_DIR)

    print(f"\nSaved outputs to {OUT_DIR}")
    print("\nTop risk-ranked contingencies:")
    print(risk.risk_by_contingency.head(10).to_string(index=False))


if __name__ == "__main__":
    main()