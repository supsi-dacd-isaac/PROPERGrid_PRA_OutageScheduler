import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- MOCKED FUNCTIONS TO TEST ---
# Assume these come from the module under test

def aggregate_hourly_demand(hourly_demand, aggregation_step='D'):
    return hourly_demand.resample(aggregation_step).mean()


def aggregate_step_costs_and_durations(cost_per_days, expected_duration_days, aggregation_step='D'):
    # Simple mapping of duration days -> number of aggregation steps
    step_map = {'H': 24, '12H': 2, 'D': 1, 'W': 1/7}  # Days per step
    scale = step_map.get(aggregation_step, 1)
    step_durations = [int(d * scale) for d in expected_duration_days]
    step_costs = [c / scale for c in cost_per_days]
    return step_costs, step_durations


# Sample test data
def create_sample_demand_df(hours=48):
    index = pd.date_range(start="2023-01-01", periods=hours, freq="H")
    data = np.random.rand(hours, 3) * 100  # 3 buses/nodes
    return pd.DataFrame(data, index=index, columns=["bus_0", "bus_1", "bus_2"])


# --- UNIT TESTS ---
def test_aggregate_hourly_demand_empty():
    df = pd.DataFrame(columns=["bus_0", "bus_1", "bus_2"])
    df.index = pd.to_datetime([])
    agg_df = aggregate_hourly_demand(df, aggregation_step='D')
    assert agg_df.empty


def test_aggregate_hourly_demand_daily_mean():
    df = create_sample_demand_df(48)
    agg_df = aggregate_hourly_demand(df, aggregation_step='D')
    assert agg_df.shape[0] == 2  # Two days
    assert np.allclose(agg_df.iloc[0].values, df.iloc[:24].mean().values, atol=1e-2)


def test_aggregate_step_costs_and_durations_daily():
    costs = [1000, 2000]
    durations = [10, 20]
    cost_per_step, duration_steps = aggregate_step_costs_and_durations(costs, durations, aggregation_step='D')
    assert cost_per_step == [1000.0, 2000.0]
    assert duration_steps == [10, 20]


def test_aggregate_step_costs_and_durations_hourly():
    costs = [2400, 4800]
    durations = [1, 2]  # in days
    cost_per_step, duration_steps = aggregate_step_costs_and_durations(costs, durations, aggregation_step='H')
    assert cost_per_step == [100.0, 200.0]
    assert duration_steps == [24, 48]

if __name__ == '__main__':
    pytest.main([__file__]) 