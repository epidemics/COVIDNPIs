import luigi
from luigi.util import requires, inherits
from pathlib import Path
from logging import getLogger
import epimodel
from epimodel import Level, RegionDataset, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, batch
from datetime import date

log = getLogger(__name__)

class Configuration(luigi.Config):
    data_directory = luigi.Parameter()

CONFIGURATION = Configuration()

def join_path(filename: str, dir_name: str = CONFIGURATION.data_directory) -> Path:
    return Path(dir_name) / Path(filename)


class RegionsFile(luigi.Task):
    region_dataset = luigi.Parameter(default="data/regions.csv")
    gleam_regions = luigi.Parameter(default="data/regions-gleam.csv")

    def load_region_dataset(self):
        rds = epimodel.RegionDataset.load(
            self.region_dataset, self.gleam_regions
        )
        epimodel.algorithms.estimate_missing_populations(rds)
        return rds


class JohnsHopkins(RegionsFile):
    output_file: str = luigi.Parameter()

    def run(self):
        log.info("Downloading and parsing CSSE ...")
        rds = self.load_region_dataset()
        csse = epimodel.imports.import_johns_hopkins(rds)
        dest = self.output_file
        csse.to_csv(dest)
        log.info(
            f"Saved CSSE to {dest}, last day is {csse.index.get_level_values(1).max()}"
        )

    def output(self):
        return luigi.LocalTarget(join_path(self.output_file))

class UpdateForetold(RegionsFile):
    output_file: str = luigi.Parameter()
    foretold_channel: str = luigi.Parameter()

    def run(self):
        log.info("Downloading and parsing foretold")
        rds = self.load_region_dataset()
        foretold = epimodel.imports.import_foretold(
            rds, self.foretold_channel
        )
        dest = self.output_file
        foretold.to_csv(dest, float_format="%.7g")
        log.info(f"Saved Foretold to {dest}")

    def output(self):
        return luigi.LocalTarget(join_path(self.output_file))

class BaseDefinition(luigi.ExternalTask):
    base_def: str = luigi.Parameter()

    def output(self):
        return luigi.Target(f"data/{self.base_def}")


class BaseDefinition(luigi.ExternalTask):
    base_def: str = luigi.Parameter()

    def output(self):
        return luigi.Target(f"data/{self.base_def}")

class CountryEstimates(luigi.ExternalTask):
    country_estimates: str = luigi.Parameter()

    def output(self):
        return luigi.Target(f"data/{self.country_estimates}")

@inherits(BaseDefinition, CountryEstimates)
class GenerateGleamBatch(RegionsFile):
    comment: str = luigi.Parameter(default=None)
    output_dir: str = luigi.Parameter()
    start_date = luigi.DateParameter(default=date.today())
    top = luigi.IntParameter(default=2000)

    def run(self):
        b = Batch.new(dir=self.output_dir, comment=self.comment)
        log.info(f"New batch file {b.path}")
        log.info(f"Reading base GLEAM definition {self.base_def} ...")
        d = epimodel.gleam.GleamDefinition(self.base_def)
        # TODO: This should be somewhat more versatile
        log.info(f"Reading estimates from CSV {self.country_estimates} ...")
        est = read_csv_smart(self.country_estimates, ctx.obj["RDS"], levels=Level.country)
        start_date = utils.utc_date(self.start_date) if self.start_date else d.get_start_date()
        log.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
        rds = self.load_region_dataset()
        batch.generate_simulations(
            b,
            d,
            est,
            rds=rds,
            config={},  # TODO: this needs to be changed
            start_date=start_date,
            top=self.top,
        )
        log.info(f"Generated batch {b.path!r}:\n  {b.stats()}")
        b.close()

        # todo: not sure about this
        #if "invoked_by_subcommand" in ctx.parent.__dict__:
        #    ctx.parent.batch_file = b.path

class ExportGleamBatch(RegionsFile):
    batch_file = luigi.Parameter()
    gleamviz_sims_dir = luigi.Parameter()
    out_dir = luigi.Parameter(default=None)
    overwrite = luigi.BoolParameter(default=False)

    def run(self):
        batch = Batch.open(self.batch_file)
        gdir = self.gleamviz_sims_dir
        if self.out_dir is not None:
            gdir = self.out_dir
        log.info(f"Creating GLEAM XML definitions for batch {self.batch_file} in dir {gdir} ...")
        batch.export_definitions_to_gleam(
            Path(gdir).expanduser(), overwrite=self.overwrite, info_level="INFO"
        )

    # TODO: maybe? `def requires(self): return GenerateGleamBatch()`

class ImportGleamBatch(RegionsFile):
    """TODO: Not implemented"""
    pass


@requires(ExportGleamBatch, JohnsHopkins)
class WebExport(luigi.Task):
    """TODO: Not implemented
    """
    pass

class WebUpload(luigi.Task):
    output_dir = luigi.Parameter()
    output_latest = luigi.Parameter()
    channel: str = luigi.Parameter()

    def run(self):
        dir_ = Path(self.output_dir) / self.output_latest
        c = "config" # TODO
        upload_export(dir_, c, channel=self.channel)

@requires(JohnsHopkins, GenerateGleamBatch, ExportGleamBatch)
class WorkflowPrepareGleam(luigi.WrapperTask):
    pass
