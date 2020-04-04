import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tables

log = logging.getLogger(__name__)


class Batch:
    """
    Represents one batch of GLEAM simulations.

    Batch metadata is held in a HDF5 file with pandas tables:
    * simulations - each row is a simulation definition
    * new_fraction - fraction of the population that transitioned
        to given compartment (columns) in the period ending on the date
        since the last (e.g. daily new fracion, weekly new fraction, ...).
    """

    SIMULATION_COLUMNS = ["Name", "Group", "Key", "Params", "DefinitionXML"]
    LEVEL_TO_GTYPE = {"country": "country", "continent": "continent"}
    COMPARTMENTS = {2: "Infected", 3: "Recovered"}

    def __init__(self, hdf_file, *, _direct=True):
        assert not _direct, "Use .new or .load"
        self.hdf = hdf_file
        self.simulations = self.hdf["simulations"]
        self.new_fraction = self.hdf["new_fraction"]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.hdf.filename} {len(self.simulations)} sims, {len(self.new_fraction)} rows>"

    @classmethod
    def open(cls, path):
        path = Path(path)
        assert path.exists()
        hdf = pd.HDFStore(path, "a")
        return cls(hdf, _direct=False)

    @classmethod
    def new(cls, dir_, name=None, suffix=None):
        """
        Create new batch, auto-naming it.
        """
        if name is None:
            now = datetime.datetime.now().astimezone()
            name = f"batch-{now.isoformat()}" + (f"-{suffix}" if suffix else "")
        path = Path(dir_) / f"{name}.hdf5"
        assert not path.exists()
        hdf = pd.HDFStore(path, "w")
        hdf["simulations"] = pd.DataFrame(
            index=pd.Index([], name="SimulationID"), columns=cls.SIMULATION_COLUMNS
        )
        hdf["new_fraction"] = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Code", "Date", "SimulationID"]),
            columns=cls.COMPARTMENTS.values(),
        )
        return cls(hdf, _direct=False)

    def import_sims(self, sims_dir, regions, *, allow_unfinished=False, resample=None):
        """
        Import simulation result data from GLEAMViz data/sims dir into the HDF5 file.
        """
        sims_dir = Path(sims_dir)
        for sid, sim in self.simulations.iterrows():
            path = sims_dir / f"{sid}.gvh5" / "results.h5"
            if not path.exists() and not allow_unfinished:
                raise Exception(f"No gleam result found for {sid} {sim.Name!r}")
        dfs = []
        skipped = set()
        for sid, sim in self.simulations.iterrows():
            path = sims_dir / f"{sid}.gvh5" / "results.h5"
            log.debug("Loading Gleam simulation from {} ..".format(path))
            with tables.File(path) as f:
                for r in regions:
                    if pd.isnull(r.GleamID):
                        skipped.add(r.DisplayName)
                        continue
                    gtype = self.LEVEL_TO_GTYPE[r.Level]
                    node = f.get_node(f"/population/new/{gtype}/median/dset")
                    ################### HACK: date TODO: get from Batch header
                    days = pd.date_range("2020-04-02", periods=node.shape[3], tz="utc")
                    dcols = {}
                    for ci, cn in self.COMPARTMENTS.items():
                        new_per_1000 = node[ci, 0, int(r.GleamID), :]
                        new_fraction = np.expand_dims(new_per_1000 / 1000.0, 0)
                        idx = pd.MultiIndex.from_tuples(
                            [(r.Code, sid)], names=["Code", "SimulationID"]
                        )
                        dcols[cn] = (
                            pd.DataFrame(
                                new_fraction.astype("f2"),
                                index=idx,
                                columns=pd.Index(days, name="Date"),
                            )
                            .stack()
                            .swaplevel(1, 2)
                        )
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
                    pd.Grouper(freq=resample, level=1),
                    pd.Grouper(level=2),
                ]
            ).mean()
        self.hdf.put("new_fraction", dfall, format="table")
        self.new_fraction = self.hdf["new_fraction"]
        log.info(f"Loaded {len0} GLEAM result rows into {self} (resampling {resample})")
