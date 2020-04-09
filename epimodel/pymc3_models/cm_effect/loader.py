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
        self.Rs = list(set(regions))

        self.rds = RegionDataset.load(self.data_dir / "regions.csv")

        # Raw data, never modified
        self.johns_hopkins = read_csv(self.data_dir / "johns-hopkins.csv")
        self.features_0to1 = read_csv(self.data_dir / "countermeasures-model-0to1.csv")

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

        self.Rs = self.filter_regions(regions, min_final_jh=101)


        self.update()

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

            if cutoff is not None:
                v[v < cutoff] = np.nan
            # [country, day]
            return np.ma.masked_invalid(v.astype(self.TheanoType))

        self.Confirmed = prep("Confirmed", self.ConfirmedCutoff)
        self.Deaths = prep("Deaths", self.DeathsCutoff)
        self.Recovered = prep("Recovered", self.RecoveredCutoff)
        self.Active = prep("Active", self.ActiveCutoff)

        self.sel_features = self.features_0to1.loc[self.Rs, self.CMs]
        if "Mask wearing" in self.sel_features.columns:
            self.sel_features["Mask wearing"] *= 0.01
        ActiveCMs = np.stack(
            [self.sel_features.loc[rc].loc[self.Ds].T for rc in self.Rs]
        )
        assert ActiveCMs.shape == (len(self.Rs), len(self.CMs), len(self.Ds))
        # [region, CM, day] Which CMs are active, and to what extent
        self.ActiveCMs = ActiveCMs.astype(self.TheanoType)

    def stats(self):
        """Print data stats, plot graphs, ..."""

        print("\nCountermeasures                            min   .. mean  .. max")
        for i, cm in enumerate(self.CMs):
            vals = np.array(self.sel_features[cm])
            print(
                f"{i:2} {cm:42} {vals.min():.3f} .. {vals.mean():.3f} .. {vals.max():.3f}"
            )
            if len(set(vals)) < 10:
                print(f"{'':46}{set(vals)}")

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
