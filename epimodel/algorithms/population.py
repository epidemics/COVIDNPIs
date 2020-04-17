import numpy as np
import pandas as pd

from ..regions import RegionDataset


def estimate_missing_populations(rds: RegionDataset, root="W"):
    """
    Traverse the given rds tree and fill in any missing populations.

    Useful only for lower hierarchy levels, e.g. GLEAM basins.
    Redistributes at mos the unallocated population of the parent
    between the children.
    Conservatively assumes that any missing populations are smaller than
    any known sybling population.
    """

    def rec(r):
        assert np.isfinite(r.Population)
        child_pops = [cr.Population for cr in r.children if np.isfinite(cr.Population)]
        min_child_size = min(child_pops) if child_pops else r.Population
        rem_pop = max(0, r.Population - sum(child_pops))
        for cr in r.children:
            if not np.isfinite(cr.Population):
                p = min(min_child_size, rem_pop / (len(r.children) - len(child_pops)))
                rds.data.at[cr.Code, "Population"] = p
            rec(cr)

    rec(rds[root])


def distribute_down_with_population(
    s: pd.Series, rds: RegionDataset, at_most_pop=False, root="W"
):
    """
    Distribute numbers in the Series down all the way to the leaves.

    Redistributes according to population. With `at_most_pop=True` caps the estimates
    at population (anything beyond the cap is lost).
    If a children already has value, it is preserved and substracted from the distributed
    amount (but only on the direct children levels).
    If a region does not have a known Population, it is skipped, including
    its subtree.
    """

    def rec(r):
        val = s.get(r.Code, np.nan)
        if np.isfinite(val):
            child_vals = np.array([s.get(cr.Code, np.nan) for cr in r.children])
            val = max(0.0, val - np.sum(child_vals, where=np.isfinite(child_vals)))
            empty_children = [
                cr
                for cr in r.children
                if not np.isfinite(s.get(cr.Code, np.nan))
                and np.isfinite(cr.Population)
            ]
            empty_pop_sum = np.sum([cr.Population for cr in empty_children])

            for cr in empty_children:
                v = val * cr.Population / max(empty_pop_sum, 1)
                if at_most_pop:
                    v = min(v, cr.Population)
                s[cr.Code] = v
        for cr in r.children:
            rec(cr)

    assert isinstance(s, pd.Series)
    rec(rds[root])
