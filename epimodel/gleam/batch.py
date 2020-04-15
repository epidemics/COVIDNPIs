import datetime
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tables
from scipy.stats import lognorm, norm

from .. import algorithms
from ..regions import Level, RegionDataset
from .definition import GleamDefinition

log = logging.getLogger(__name__)

SIMULATION_COLUMNS = ["Name", "Group", "Key", "StartDate", "DefinitionXML"]

LEVEL_TO_GTYPE = {
    Level.country: "country",
    Level.continent: "continent",
    Level.gleam_basin: "city",
}
COMPARTMENTS = {2: "Infected", 3: "Recovered"}

GLEAM_ID_SUFFIX = "574"  # Magic? Or arbtrary?


class Batch:
    """
    Represents one batch of GLEAM simulations.

    Batch metadata is held in a HDF5 file with pandas tables:
    * simulations - each row is a simulation definition
    * new_fraction - fraction of the population that transitioned
        to given compartment (columns) in the period ending on the date
        since the last (e.g. daily new fracion, weekly new fraction, ...).
    """

    def __init__(self, hdf_file, path, *, _direct=True):
        assert not _direct, "Use .new or .load"
        self.hdf = hdf_file
        self.path = path

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.hdf.filename}>"

    def stats(self):
        def get_table(n):
            return self.hdf.get(n) if n in self.hdf else None

        s = []
        t = get_table("simulations")
        if t is not None:
            s.append(f"{len(t)} simulations")
        t = get_table("initial_compartments")
        if t is not None:
            s.append(f"{len(t)} initial compartments {list(t.columns)}")
        t = get_table("new_fraction")
        if t is not None:
            s.append(
                f"{len(t)} rows of results {list(t.columns)} "
                f"({len(t.index.levels[-1])} dates, {len(t.index.levels[-2])} regions)"
            )
        return ", ".join(s)

    @classmethod
    def open(cls, path):
        path = Path(path)
        assert path.exists()
        hdf = pd.HDFStore(path, "a")
        return cls(hdf, path, _direct=False)

    @classmethod
    def new(cls, *, path=None, dir=None, comment=None):
        """
        Create new batch HDF5 file.
        
        Either `path` should be a (non-existing) file, or a `dir` should
        be given - name is then auto-generated (with optional comment suffix).
        """
        if path is None:
            dir = Path(dir)
            assert dir.is_dir()
            now = datetime.datetime.now().astimezone(datetime.timezone.utc)
            name = f"batch-{now.isoformat()}" + (f"-{comment}" if comment else "")
            path = dir / f"{name}.hdf5"
        path = Path(path)

        assert not path.exists()
        hdf = pd.HDFStore(path, "w")
        return cls(hdf, path, _direct=False)

    def set_initial_compartments(self, initial_df):
        """
        `initial_df` must be indexed by `Code` and columns should be
        compartment names.
        """
        self.hdf.put(
            "initial_compartments",
            initial_df.astype("f2"),
            format="table",
            complib="bzip2",
            complevel=9,
        )

    def close(self):
        """
        Close the HDF5 file. The Batch object is then unusable.
        """
        self.hdf.close()

    def set_simulations(self, sims_def_name_group_key):
        """
        Write simulation records.

        Each element of `sims_def_group_key` is `(definition, name, group, key)`.
        
        In addiMakes the ID unique, adds the initial compartments clearing any old seeds,
        records the simulation and seeds in the HDF file.

        TODO: Potentially have a nicer interface than list-of-tuples
        """
        last_id = 0
        rows = []
        for definition, name, group, key in sims_def_name_group_key:
            d = definition.copy()
            last_id = max(int(time.time() * 1000), last_id + 1)
            gv2id = f"{last_id}.{GLEAM_ID_SUFFIX}"
            d.set_id(gv2id)
            s = io.BytesIO()
            d.save(s)
            rows.append(
                {
                    "SimulationID": gv2id,
                    "Name": name,
                    "Group": group,
                    "Key": key,
                    "StartDate": d.get_start_date().isoformat(),
                    "DefinitionXML": s.getvalue().decode("ascii"),
                }
            )
        data = pd.DataFrame(rows).set_index("SimulationID", verify_integrity=True)
        self.hdf.put("simulations", data, format="table", complib="bzip2", complevel=9)

    def export_definitions_to_gleam(
        self, sims_dir, sim_ids=None, overwrite=False, info_level=logging.DEBUG,
    ):
        sims_df = self.hdf["simulations"]
        sims_dir = Path(sims_dir)
        if sim_ids is None:
            sim_ids = sims_df.index
        for sid in sim_ids:
            row = sims_df.loc[sid]
            p = sims_dir / f"{sid}.gvh5"
            log.log(info_level, f"Exporting sim to {p} ...")
            p.mkdir(exist_ok=overwrite)
            (p / "definition.xml").write_bytes(row.DefinitionXML.encode("ascii"))

    def import_results_from_gleam(
        self,
        sims_dir,
        regions,
        *,
        allow_unfinished=False,
        resample=None,
        overwrite=False,
        info_level=logging.DEBUG,
    ):
        """
        Import simulation result data from GLEAMViz data/sims dir into the HDF5 file.
        """
        if "new_fraction" in self.hdf and not overwrite:
            raise Exception(f"Would overwrite existing `new_fraction` in {self}!")
        sims_df = self.hdf["simulations"]
        sims_dir = Path(sims_dir)
        for sid, sim in sims_df.iterrows():
            path = sims_dir / f"{sid}.gvh5" / "results.h5"
            if not path.exists() and not allow_unfinished:
                raise Exception(f"No gleam result found for {sid} {sim.Name!r}")
        dfs = []
        skipped = set()
        for sid, sim in sims_df.iterrows():
            path = sims_dir / f"{sid}.gvh5" / "results.h5"
            if not path.exists() and allow_unfinished:
                log.log(info_level, "Skipping missing result file {} ..".format(path))
                continue

            log.log(info_level, "Loading results from {} ..".format(path))
            with tables.File(path) as f:
                for r in regions:
                    if pd.isnull(r.GleamID):
                        skipped.add(r.DisplayName)
                        continue
                    gtype = LEVEL_TO_GTYPE[r.Level]
                    node = f.get_node(f"/population/new/{gtype}/median/dset")
                    days = pd.date_range(sim.StartDate, periods=node.shape[3], tz="utc")
                    dcols = {}
                    for ci, cn in COMPARTMENTS.items():
                        new_fraction = node[ci, 0, int(r.GleamID), :]
                        new_fraction = np.expand_dims(new_fraction, 0)
                        idx = pd.MultiIndex.from_tuples(
                            [(sid, r.Code)], names=["SimulationID", "Code"]
                        )
                        dcols[cn] = pd.DataFrame(
                            new_fraction.astype("f2"),
                            index=idx,
                            columns=pd.Index(days, name="Date"),
                        ).stack()
                    dfs.append(pd.DataFrame(dcols).sort_index())
        if skipped:
            log.info(f"Skipped {len(skipped)} regions without GleamID: {skipped!r}")
        if not dfs:
            raise Exception("No GLEAM records loaded!")
        dfall = pd.concat(dfs)
        len0 = len(dfall)
        if resample is not None:
            dfall = dfall.groupby(
                [
                    pd.Grouper(level=0),
                    pd.Grouper(level=1),
                    pd.Grouper(freq=resample, level=2),
                ]
            ).mean()

        self.hdf.put(
            "new_fraction", dfall, format="table", complib="bzip2", complevel=9
        )
        log.info(f"Loaded {len0} GLEAM result rows into {self} (resampling {resample})")

    def get_cummulative_active_df(self):
        """
        Get a dataframe with cummulative 'Infected' and 'Recovered', and
        with 'Active' infections. All are fractions of population (i.e. per 1 person).
        
        Both cummulative Infected and Active are offsetted by the original Infectious
        compartment (or to make Active always positive, whatever is larger).
        """
        df = self.hdf["new_fraction"].sort_index()
        df = df.groupby(level=[0, 1]).cumsum()

        df_active = df["Infected"] - df["Recovered"]
        df_region_min = df_active.groupby(level=1).min()
        df["Infected"] += np.maximum(
            df_region_min, self.hdf["initial_compartments"]["Infectious"]
        )

        df["Active"] = df["Infected"] - df["Recovered"]
        return df

    def generate_sim_stats(self, cummulative_active_df, region, sim_ids):
        cdf = cummulative_active_df
        tot_infected = cdf.loc[
            ([s.SimulationID for s in sim_ids], region.Code, -1), "Infected"
        ]
        actives = cdf.loc[
            ([s.SimulationID for s in sim_ids], region.Code, None), "Active"
        ]
        max_active_infected = actives.groupby(level=0).max()
        print(tot_infected, max_active_infected)
        stats = {}
        for data, name in [
            (tot_infected, "TotalInfected"),
            (max_active_infected, "MaxActiveInfected"),
        ]:
            m, v = norm.fit(data)
            v = max(v, 3e-5)
            dist = norm(m, v)
            stats[f"{name}_fraction_mean"] = dist.mean()
            stats[f"{name}_fraction_q05"] = max(dist.ppf(0.05), 0.0)
            stats[f"{name}_fraction_q95"] = min(dist.ppf(0.95), 1.0)
        return stats


def generate_simulations(
    batch: Batch,
    definition: GleamDefinition,
    sizes: pd.Series,
    rds: RegionDataset,
    config: dict,
    start_date: datetime.datetime,
    top: int = None,
):
    # Estimate infections in subregions
    s = sizes.copy()
    algorithms.distribute_down_with_population(s, rds)

    # Create compartment sizes
    mults = config["compartment_multipliers"]
    mult_sum = sum(mults.values())
    s = np.minimum(
        s, rds.data.Population * config["compartments_max_fraction"] / mult_sum
    )
    s = s[pd.notnull(s)]
    est = pd.DataFrame({n: s * m for n, m in mults.items()})

    # Update definition and batch
    d = definition.copy()
    d.set_start_date(start_date)
    d.clear_seeds()
    d.add_seeds(rds, est, top=top)
    batch.set_initial_compartments(est)

    # Generate simulations
    sims = []
    for mit in config["groups"]:
        for sce in config["scenarios"]:
            par = dict(mit)
            par.update(sce)
            d2 = d.copy()

            d2.set_seasonality(par["param_seasonalityAlphaMin"])
            d2.set_traffic_occupancy(par["param_occupancyRate"])
            d2.set_variable("beta", par["param_beta"])
            # TODO: other params or variables?
            d2.set_default_name()

            sims.append((d2, par["name"], par["group"], par["key"]))
    batch.set_simulations(sims)
