### Initial imports

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ... import RegionDataset, read_csv

log = logging.getLogger(__name__)


class Loader:
    def __init__(self, start, end, regions, features, data_dir=None):
        if data_dir is None:
            data_dir = Path(__file__).parents[3] / "data"
        self.data_dir = data_dir

        # Days
        self.Ds = pd.date_range(start=start, end=end, tz="utc")

        # Features
        self.CMs = list(features)

        # Countries / regions
        self.Rs = list(regions)

        self.rds = RegionDataset.load(self.data_dir / "regions.csv")

        # Raw data, never modified
        self.johns_hopkins = read_csv(self.data_dir / "johns-hopkins.csv")
        self.features_0to1 = read_csv(self.data_dir / "countermeasures-model-0to1.csv")

        # Selected features:
        self.features = self.features_0to1

        self.TheanoType = "float64"

        self.Confirmed = None
        self.ConfirmedCutoff = 10.0
        self.Deaths = None
        self.DeathsCutoff = 10.0
        self.Active = None
        self.ActiveCutoff = 10.0
        self.Recovered = None
        self.RecoveredCutoff = 10.0

        self.ActiveCMs = None

        self.update()

    def split_0to1_features(self, exclusive=False):
        """
        Split joined features in model-0to1 into separate bool features.

        Resulting DF is stored in `self.features_split` and returned.
        """
        fs = {}
        f01 = self.features_0to1

        fs["Masks over 60"] = f01["Mask wearing"] >= 60

        fs["Asymptomatic contact isolation"] = f01["Asymptomatic contact isolation"]

        fs["Gatherings limited to 10"] = f01["Gatherings limited to"] > 0
        fs["Gatherings limited to 100"] = f01["Gatherings limited to"] > 0
        fs["Gatherings limited to 1000"] = f01["Gatherings limited to"] > 0

        fs["Business suspended - some"] = f01["Business suspended"] > 0.1
        fs["Business suspended - many"] = f01["Business suspended"] > 0.6

        fs["Schools and universities closed"] = f01["Schools and universities closed"]

        fs["Distancing and hygiene over 0.2"] = (
            f01["Minor distancing and hygiene measures"] > 0.2
        )

        fs["General curfew - permissive"] = f01["General curfew"] > 0.1
        fs["General curfew - strict"] = f01["General curfew"] > 0.6

        fs["Healthcare specialisation over 0.2"] = (
            f01["Healthcare specialisation"] > 0.2
        )

        fs["Phone line"] = f01["Phone line"]

        return pd.DataFrame(fs).astype("f4")

    def update(self):
        """(Re)compute the values used in the model after any parameter/region/etc changes."""

        def prep(name, cutoff=None):
            # Confirmed cases, masking values smaller than 10
            v = (
                self.johns_hopkins[name]
                .loc[(tuple(self.Rs), self.Ds)]
                .unstack(1)
                .values
            )
            assert v.shape == (len(self.Rs), len(self.Ds))
            if cutoff is not None:
                v[v < cutoff] = np.nan
            # [country, day]
            return np.ma.masked_invalid(v.astype(self.TheanoType))

        self.Confirmed = prep("Confirmed", self.ConfirmedCutoff)
        self.Deaths = prep("Deaths", self.DeathsCutoff)
        self.Recovered = prep("Recovered", self.RecoveredCutoff)
        self.Active = prep("Active", self.ActiveCutoff)

        self.ActiveCMs = self.get_ActiveCMs(self.Ds[0], self.Ds[-1])

    def get_ActiveCMs(self, start, end):
        local_Ds = pd.date_range(start=start, end=end, tz="utc")
        self.sel_features = self.features.loc[self.Rs, self.CMs]
        if "Mask wearing" in self.sel_features.columns:
            self.sel_features["Mask wearing"] *= 0.01
        ActiveCMs = np.stack(
            [self.sel_features.loc[rc].loc[local_Ds].T for rc in self.Rs]
        )
        assert ActiveCMs.shape == (len(self.Rs), len(self.CMs), len(local_Ds))
        # [region, CM, day] Which CMs are active, and to what extent
        return ActiveCMs.astype(self.TheanoType)

    def print_stats(self):
        """Print data stats, plot graphs, ..."""

        print("\nCountermeasures                            min   .. mean  .. max")
        for i, cm in enumerate(self.CMs):
            vals = np.array(self.sel_features[cm])
            print(
                f"{i:2} {cm:42} {vals.min():.3f} .. {vals.mean():.3f}"
                f" .. {vals.max():.3f}"
                f"  {set(vals) if len(set(vals)) <= 4 else ''}"
            )

        # TODO: add more

    def filter_regions(
        self, regions, min_feature_sum=1.0, min_final_jh=400, jh_col="Confirmed"
    ):
        """Filter and return list of region codes."""
        res = []
        for rc in regions:
            r = self.rds[rc]
            if rc in self.johns_hopkins.index and rc in self.features_0to1.index:
                if self.johns_hopkins.loc[(rc, self.Ds[-1]), jh_col] < min_final_jh:
                    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                    continue
                # TODO: filter by features?
                # if self.active_features.loc[(rc, self.Ds)] ...
                #    print(f"Region {r} has <{min_final_jh} final JH col {jh_col}")
                #    continue
                res.append(rc)
        return res


def statstr(d):
    return f"{d.mean():.3g} ({np.quantile(d, 0.05):.3g} .. {np.quantile(d, 0.95):.3g})"
