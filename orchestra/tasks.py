import luigi
from luigi.util import requires, inherits
from pathlib import Path
from logging import getLogger
import epimodel
from epimodel import Level, RegionDataset, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, batch
from datetime import date
import dill
import yaml
import datetime


log = getLogger(__name__)


class RegionsFile(luigi.ExternalTask):
    regions = luigi.Parameter(default="data/regions.csv")

    def output(self):
        return luigi.LocalTarget(self.regions)


class GleamRegions(luigi.ExternalTask):
    gleams = luigi.Parameter(default="data/regions-gleam.csv")

    def output(self):
        return luigi.LocalTarget(self.gleams)


@inherits(RegionsFile, GleamRegions)
class RegionsDataset(luigi.Task):
    region_dataset = luigi.Parameter(default="data/rds.pk")

    def run(self):
        regions = self.input()["region_file"].path
        gleams = self.input()["gleam_regions"].path
        rds = epimodel.RegionDataset.load(regions, gleams)
        epimodel.algorithms.estimate_missing_populations(rds)
        with open(self.region_dataset, "wb") as ofile:
            dill.dump(rds, ofile)

    def output(self):
        return luigi.LocalTarget(self.region_dataset)

    def requires(self):
        return {
            "region_file": self.clone(RegionsFile),
            "gleam_regions": self.clone(GleamRegions),
        }

    @staticmethod
    def load_dilled_rds(path: str):
        with open(path, "rb") as ifile:
            return dill.load(ifile)


@requires(RegionsDataset)
class JohnsHopkins(luigi.Task):
    output_file: str = luigi.Parameter()

    def run(self):
        log.info("Downloading and parsing CSSE ...")
        rds = RegionsDataset.load_dilled_rds(self.input().path)
        csse = epimodel.imports.import_johns_hopkins(rds)
        dest = self.output_file
        csse.to_csv(dest)
        log.info(
            f"Saved CSSE to {dest}, last day is {csse.index.get_level_values(1).max()}"
        )

    def output(self):
        return luigi.LocalTarget(self.output_file)


@requires(RegionsDataset)
class UpdateForetold(luigi.Task):
    output_file: str = luigi.Parameter()
    foretold_channel: str = luigi.Parameter()

    def run(self):
        log.info("Downloading and parsing foretold")
        rds = RegionsDataset.load_dilled_rds(self.input().path)
        foretold = epimodel.imports.import_foretold(rds, self.foretold_channel)
        dest = self.output_file
        foretold.to_csv(dest, float_format="%.7g")
        log.info(f"Saved Foretold to {dest}")

    def output(self):
        return luigi.LocalTarget(self.output_file)


class BaseDefinition(luigi.ExternalTask):
    base_def: str = luigi.Parameter(default="data/config.xml")

    def output(self):
        return luigi.LocalTarget(self.base_def)


class CountryEstimates(luigi.ExternalTask):
    country_estimates: str = luigi.Parameter(default="data/country_estimates.csv")

    def output(self):
        return luigi.LocalTarget(self.country_estimates)


class ConfigYaml(luigi.ExternalTask):
    config_yaml: str = luigi.Parameter(default="config.yaml")

    @staticmethod
    def load(path):

        with open(path, "rt") as f:
            return yaml.safe_load(f)

    def output(self):
        return luigi.LocalTarget(self.config_yaml)


@inherits(BaseDefinition, CountryEstimates, RegionsDataset, ConfigYaml)
class GenerateGleamBatch(luigi.Task):
    comment: str = luigi.Parameter(default=None)
    output_suffix: str = luigi.DateSecondParameter(default=datetime.datetime.utcnow())
    output_filename_prefix: str = luigi.Parameter(default="batch-")
    output_directory: str = luigi.Parameter(default="data")
    start_date = luigi.DateParameter(default=date.today())
    top = luigi.IntParameter(default=2000)

    def requires(self):
        return {
            "base_def": self.clone(BaseDefinition),
            "country_estimates": self.clone(CountryEstimates),
            "regions_dataset": self.clone(RegionsDataset),
            "config_yaml": self.clone(ConfigYaml),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_path = Path(self.output_directory).joinpath(
            f"{self.output_filename_prefix}-{self.output_suffix}.hdf5"
        )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        b = Batch.new(path=self.output_path, dir=self.output_dir, comment=self.comment)
        log.info(f"New batch file {b.path}")

        base_def = self.input()["base_def"].path
        log.info(f"Reading base GLEAM definition {base_def} ...")
        d = epimodel.gleam.GleamDefinition(base_def)

        # TODO: This should be somewhat more versatile
        country_estimates = self.input()["country_estimates"].path
        rds = RegionsDataset.load_dilled_rds(self.input()["regions_dataset"].path)
        log.info(f"Reading estimates from CSV {country_estimates} ...")
        est = read_csv_smart(self.country_estimates, rds, levels=Level.country)
        start_date = (
            utils.utc_date(self.start_date) if self.start_date else d.get_start_date()
        )
        log.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
        batch.generate_simulations(
            b,
            d,
            est,
            rds=rds,
            config=ConfigYaml.load(self.input()["config_yaml"].path),
            start_date=start_date,
            top=self.top,
        )
        log.info(f"Generated batch {b.path!r}:\n  {b.stats()}")
        b.close()

        # todo: not sure about this
        # if "invoked_by_subcommand" in ctx.parent.__dict__:
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
        log.info(
            f"Creating GLEAM XML definitions for batch {self.batch_file} in dir {gdir} ..."
        )
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
        c = "config"  # TODO
        upload_export(dir_, c, channel=self.channel)


@requires(JohnsHopkins, GenerateGleamBatch, ExportGleamBatch)
class WorkflowPrepareGleam(luigi.WrapperTask):
    pass
