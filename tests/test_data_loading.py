from pathlib import Path

from pra_psa.data import load_system, read_matpower_m, read_matpower_xlsx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def test_read_matpower_xlsx_ieee24():
    case = read_matpower_xlsx(DATA_DIR / "IEEE24" / "System.xlsx")
    assert case["bus"].shape[0] == 24
    assert case["branch"].shape[0] == 38
    assert case["gen"].shape[0] == 33
    assert case["baseMVA"] == 100


def test_read_matpower_m_ieee24():
    case = read_matpower_m(DATA_DIR / "IEEE24" / "System.m")
    assert case["bus"].shape[0] == 24
    assert case["branch"].shape[0] == 38
    assert case["gen"].shape[0] == 33


def test_load_system_without_pandapower():
    system = load_system(DATA_DIR, "IEEE24", as_pandapower=False)
    assert system.net is None
    assert system.load_time_series.shape[1] == 24
    assert list(system.load_time_series.columns[:3]) == [1, 2, 3]
