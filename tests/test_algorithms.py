import numpy as np
import pytest
import pandas as pd
from epimodel import algorithms


def test_estimate_pop(regions):
    # No population left in Europe
    regions.data.loc["VA", "Population"] = np.nan
    algorithms.estimate_missing_populations(regions, root="W-EU")
    assert regions["VA"].Population == 0.0

    # Some population left, distributed evenly
    regions.data["Population"] = np.maximum(regions.data["Population"], 11000.0)
    regions.data.loc["CZ", "Population"] = np.nan
    regions.data.loc["SK", "Population"] = np.nan
    chpop = [r.Population for r in regions["W-EU"].children]
    tot_chpop = np.sum(chpop, where=np.isfinite(chpop))
    regions.data.loc["W-EU", "Population"] = tot_chpop + 20000
    algorithms.estimate_missing_populations(regions, root="W-EU")
    assert regions["CZ"].Population == pytest.approx(10000.0, rel=0.1)
    assert regions["SK"].Population == pytest.approx(10000.0, rel=0.1)


def test_distribute_down(regions_gleam):
    s = pd.Series([1e6, 9e5], index=["W", "W-NA"])
    algorithms.distribute_down_with_population(s, regions_gleam)
    s = s.sort_index()
    assert s["W"] == 1e6
    assert s["W-AS"] == pytest.approx(65000, rel=0.05)
    assert s["W-EU"] == pytest.approx(10500, rel=0.05)
    assert s["CZ"] == pytest.approx(150, rel=0.05)
    assert s["G-PRG"] == pytest.approx(80, rel=0.1)
    assert s["W-NA"] == 9e5
