import datetime
import logging
from pathlib import Path
from typing import List

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
    Level.gleam_basin: "basin",
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
        `initial_df` should be indexed by `Code` and columns should be
        compartment names.

        Alternately, it can have a `Region` column containing Region
        objects, which will then be used to obtain the `Code` index.
        """
        if "Region" in initial_df:
            codes = initial_df["Region"].apply(lambda reg: reg.Code)
            initialdf = initial_df.drop(columns=["Region"]).set_index(codes)

        self.hdf.put(
            "initial_compartments",
            initial_df.astype("float32"),
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
        rows = [
            {
                "SimulationID": definition.get_id_str(),
                "Name": name,
                "Group": group,
                "Key": key,
                "StartDate": definition.get_start_date_str(),
                "DefinitionXML": definition.to_xml_string(),
            }
            for definition, name, group, key in sims_def_name_group_key
        ]
        data = pd.DataFrame(rows).set_index("SimulationID", verify_integrity=True)
        self.hdf.put("simulations", data, format="table", complib="bzip2", complevel=9)

    def export_definitions_to_gleam(
        self, sims_dir, sim_ids=None, overwrite=False, info_level=logging.DEBUG
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
                            new_fraction.astype("float32"),
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
        df = self.hdf["new_fraction"].astype("float32").sort_index()
        df = df.groupby(level=[0, 1]).cumsum()

        df_active = df["Infected"] - df["Recovered"]
        df_region_min = df_active.groupby(level=1).min()
        df["Infected"] -= df_region_min
        # TODO: consider using self.hdf["initial_compartments"]["Infectious"]
        # But needs adjustment for Population!

        df["Active"] = df["Infected"] - df["Recovered"]
        return df

    @staticmethod
    def generate_sim_stats(cdf: pd.DataFrame, sim_ids: List[str]) -> dict:
        # get the end date of the simulations
        end_date = cdf.index.get_level_values("Date").max()
        # get the infected in the end date for the latest date per simulation
        tot_infected = cdf.loc[(sim_ids, end_date), "Infected"]

        # get the maximum number of infected per simulation
        max_active_infected = (
            cdf.loc[(sim_ids,), "Active"].groupby(level="SimulationID").max()
        )

        stats = {}
        for data, name in [
            (tot_infected, "TotalInfected"),
            (max_active_infected, "MaxActiveInfected"),
        ]:
            m, v = norm.fit(data)
            v = max(v, 3e-5)
            dist = norm(m, v)
            stats[name] = {
                "mean": dist.mean(),
                "q05": max(dist.ppf(0.05), 0.0),
                "q20": dist.ppf(0.20),
                "q40": dist.ppf(0.40),
                "q60": dist.ppf(0.60),
                "q80": dist.ppf(0.80),
                "q95": min(dist.ppf(0.95), 1.0),
            }
        return stats


def generate_simulations(
    batch: Batch,
    definition: GleamDefinition,
    data: pd.DataFrame,
    rds: RegionDataset,
    config: dict,
    start_date: datetime.datetime,
    top: int = None,
    size_column="Infectious_mean",
):
    # Estimate infections in subregions
    if size_column not in data.columns:
        raise Exception(f"Column {size_column} not found in {list(data.columns)}")
    s = data[size_column].copy().astype("float32")
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

            beta_mult = par.get("param_beta_multiplier", 1.0)

            d2.clear_exceptions()
            next_day = d2.get_start_date()
            for days, exc in par.get("param_beta_exceptions", ()):
                for rc, row in data.iterrows():
                    end_day = next_day + pd.DateOffset(days)
                    if isinstance(exc, float):
                        beta = float(exc)
                    elif isinstance(exc, str):
                        if exc not in row:
                            raise ValueError(f"Beta-value column {exc!r} not present")
                        beta = row[exc]
                        if not np.isfinite(beta):
                            raise ValueError(
                                f"Beta {exc!r} is NaN or Inf for region {rc!r}"
                            )
                    else:
                        raise TypeError(
                            f"Unsupportted type for beta in 'param_beta_exceptions': {type(exc)}"
                        )
                    d2.add_exception(
                        [rds[rc]],
                        {"beta": beta * beta_mult},
                        start=next_day,
                        end=end_day,
                    )
                next_day = end_day

            d2.set_seasonality(par["param_seasonalityAlphaMin"])
            d2.set_traffic_occupancy(par["param_occupancyRate"])
            d2.set_variable("beta", par["param_beta"] * beta_mult)
            # TODO: other params or variables?
            d2.set_name(
                f"{batch.path.name} "
                f"{d2.get_start_date().date().isoformat()}-{d2.get_end_date().date().isoformat()} "
                f"{par['group']} {par['key']} beta_mult={beta_mult:.3g}"
            )

            sims.append((d2, par["name"], par["group"], par["key"]))
    batch.set_simulations(sims)
