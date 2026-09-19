"""Materialize the Swissgrid 2025 case as native pandapower JSON."""
from pathlib import Path
from case_swissgrid2025 import save_pandapower_json

if __name__ == "__main__":
    out = Path(__file__).with_name("swissgrid2025_pandapower.json")
    net = save_pandapower_json(out)
    print(f"Saved {out}")
    print(net)
