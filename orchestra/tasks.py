from datetime import datetime
from logging import getLogger
from pathlib import Path

import dill
import luigi
import yaml
from luigi.util import inherits, requires

from epimodel import Level, RegionDataset, algorithms, imports, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, GleamDefinition, batch as batch_module

log = getLogger(__name__)


def default_from_config(task_name: str, param_name: str) -> dict:
    return dict(section=task_name, name=param_name)


class RegionsFile(luigi.ExternalTask):
    # TODO: we can use config_path to fetch defaults from the luigi.cfg
    # TODO: or use the default in parameter. Now we use both for different tasks
    regions = luigi.Parameter(default="data/regions.csv")

    def output(self):
        return luigi.LocalTarget(self.regions)


class GleamRegions(luigi.ExternalTask):
    gleams = luigi.Parameter(default="data/regions-gleam.csv")

    def output(self):
        return luigi.LocalTarget(self.gleams)


@inherits(RegionsFile, GleamRegions)
class RegionsDatasetTask(luigi.Task):
    region_dataset = luigi.Parameter(
        config_path=default_from_config("RegionsDatasetTask", "region_dataset")
    )

    def run(self):
        regions = self.input()["region_file"].path
        gleams = self.input()["gleam_regions"].path
        rds = RegionDataset.load(regions, gleams)
        algorithms.estimate_missing_populations(rds)
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


@requires(RegionsDatasetTask)
class JohnsHopkins(luigi.Task):
    hopkins_output: str = luigi.Parameter(
        config_path=default_from_config("JohnsHopkins", "hopkins_output")
    )

    def run(self):
        log.info("Downloading and parsing CSSE ...")
        rds = RegionsDatasetTask.load_dilled_rds(self.input().path)
        csse = imports.import_johns_hopkins(rds)
        dest = self.hopkins_output
        csse.to_csv(dest)
        log.info(
            f"Saved CSSE to {dest}, last day is {csse.index.get_level_values(1).max()}"
        )

    def output(self):
        return luigi.LocalTarget(self.hopkins_output)


@requires(RegionsDatasetTask)
class UpdateForetold(luigi.Task):
    foretold_output: str = luigi.Parameter(
        config_path=default_from_config("UpdateForetold", "foretold_output")
    )
    foretold_channel: str = luigi.Parameter(
        config_path=default_from_config("UpdateForetold", "foretold_channel")
    )

    def run(self):
        log.info("Downloading and parsing foretold")
        rds = RegionsDatasetTask.load_dilled_rds(self.input().path)
        foretold = imports.import_foretold(rds, self.foretold_channel)
        dest = self.foretold_output
        foretold.to_csv(dest, float_format="%.7g")
        log.info(f"Saved Foretold to {dest}")

    def output(self):
        return luigi.LocalTarget(self.foretold_output)


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


@inherits(BaseDefinition, CountryEstimates, RegionsDatasetTask, ConfigYaml)
class GenerateGleamBatch(luigi.Task):
    comment: str = luigi.Parameter(default=None)
    output_suffix: str = luigi.DateSecondParameter(default=datetime.utcnow())
    output_filename_prefix: str = luigi.Parameter(default="batch-")
    output_directory: str = luigi.Parameter(default="data")
    start_date = luigi.DateParameter(default=datetime.utcnow())
    top = luigi.IntParameter(default=2000)

    def requires(self):
        return {
            "base_def": self.clone(BaseDefinition),
            "country_estimates": self.clone(CountryEstimates),
            "regions_dataset": self.clone(RegionsDatasetTask),
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
        b = Batch.new(path=self.output_path)
        log.info(f"New batch file {b.path}")

        base_def = self.input()["base_def"].path
        log.info(f"Reading base GLEAM definition {base_def} ...")
        d = GleamDefinition(base_def)

        # TODO: This should be somewhat more versatile
        country_estimates = self.input()["country_estimates"].path
        rds = RegionsDatasetTask.load_dilled_rds(self.input()["regions_dataset"].path)
        log.info(f"Reading estimates from CSV {country_estimates} ...")
        est = read_csv_smart(self.country_estimates, rds, levels=Level.country)
        start_date = (
            utils.utc_date(self.start_date) if self.start_date else d.get_start_date()
        )
        log.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
        batch_module.generate_simulations(
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


@inherits(RegionsDatasetTask, GenerateGleamBatch)
class ExportGleamBatch(luigi.Task):
    exports_dir = luigi.Parameter(default="~/GLEAMviz/data/sims/")
    overwrite = luigi.BoolParameter(default=False)

    def run(self):
        batch_file = self.inpu()["batch_file"]
        batch = Batch.open(batch_file)
        gdir = Path(self.exports_dir)
        log.info(
            f"Creating GLEAM XML definitions for batch {self.batch_file} in dir {gdir} ..."
        )
        batch.export_definitions_to_gleam(
            gdir.expanduser(), overwrite=self.overwrite, info_level="INFO"
        )

    def requires(self):
        return {
            "batch_file": self.clone(GenerateGleamBatch),
            "region_dataset": self.clone(RegionsDatasetTask),
        }


class GleamvizResults(luigi.ExternalTask):
    """This is done manually by a user via Gleam software"""

    gleamviz_result = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.gleamviz_result)


@inherits(GleamvizResults, RegionsDatasetTask, ConfigYaml)
class ImportGleamBatch(luigi.Task):
    # TODO: I wasn't sure about the flow here, not complete
    exports_dir = luigi.Parameter(default="~/GLEAMviz/data/sims/")
    overwrite = luigi.BoolParameter(default=True)
    allow_missing = luigi.BoolParameter(default=True)

    def requires(self):
        return {
            "batch_file": self.clone(GleamvizResults),
            "region_dataset": self.clone(RegionsDatasetTask),
            "config_yaml": self.clone(ConfigYaml),
        }

    def run(self):
        batch_file = self.input()["batch_file"].path
        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetTask.load_dilled_rds(
            self.input()["regions_dataset"].path
        )

        b = Batch.open(batch_file)
        d = regions_dataset.data

        regions = set(
            d.loc[
                ((d.Level == Level.country) | (d.Level == Level.continent))
                & (d.GleamID != "")
            ].Region.values
        )
        # Add all configured regions
        for rc in config_yaml["export_regions"]:
            r = regions_dataset[rc]
            if r.GleamID != "":
                regions.add(r)

        log.info(
            f"Importing results for {len(regions)} from GLEAM into {batch_file} ..."
        )
        b.import_results_from_gleam(
            Path(self.exports_dir).expanduser(),
            regions,
            resample=config_yaml["gleam_resample"],
            allow_unfinished=self.allow_missing,
            overwrite=self.overwrite,
            info_level="INFO",
        )

    # TODO: finish here
    def output(self):
        pass


class Rates(luigi.ExternalTask):
    rates: str = luigi.Parameter(default="data/rates.csv")

    def output(self):
        return luigi.LocalTarget(self.rates)


class Timezones(luigi.ExternalTask):
    timezones: str = luigi.Parameter(default="data/timezones.csv")

    def output(self):
        return luigi.LocalTarget(self.timezones)


class AgeDistributions(luigi.ExternalTask):
    age_distributions: str = luigi.Parameter(default="data/various/age_dist_un.csv")

    def output(self):
        return luigi.LocalTarget(self.age_distributions)


WEB_EXPORT_REQUIRED_TASKS = {
    "batch_file": ImportGleamBatch,
    "hopkins": JohnsHopkins,
    "foretold": UpdateForetold,
    "rates": Rates,
    "timezone": Timezones,
    "age_distribution": AgeDistributions,
    "config_yaml": ConfigYaml,
    "country_estimates": CountryEstimates,  # "estimates" in click - is the same?
}


@requires(*WEB_EXPORT_REQUIRED_TASKS.values())
class WebExport(luigi.Task):
    comment: str = luigi.Parameter(default="")
    pretty_print: bool = luigi.BoolParameter(default=False)
    web_export_directory: str = luigi.Parameter(default="web-exports")
    gs_datafile_name: str = luigi.Parameter(default="data-v4.json")

    # full_export_path: str = luigi.Parameter()  # something which can be used in the output

    def run(self):
        batch_file = self.input()["batch_file"].path
        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetTask.load_dilled_rds(
            self.input()["regions_dataset"].path
        )
        estimates = self.input()["country_estimates"].path

        ex = process_export(
            config_yaml,
            regions_dataset,
            False,
            self.comment,
            batch_file,
            estimates,
            self.pretty_print,
        )
        ex.write(
            self.web_export_directory,
            self.gs_datafile_name,
            latest="latest",  # not sure this is being used
            pretty_print=self.pretty_print,
        )

    def output(self):
        # TODO: not enough - we need a specific export dir
        return luigi.LocalTarget(self.web_export_directory)

    def requires(self):
        return {name: self.clone(task) for name, task in WEB_EXPORT_REQUIRED_TASKS}


@requires(WebExport)
class WebUpload(luigi.WrapperTask):
    gs_prefix: str = luigi.Parameter(default="gs://static-covid/static/v4/")
    channel: str = luigi.Parameter(default="main")
    dir_to_upload: str = luigi.Parameter()  # WebExport.full_export_path and self.input() instead

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs_path = Path(self.gs_prefix).joinpath(Path(self.dir_to_upload).parts[-1])

    def run(self):
        upload_export(
            self.dir_to_upload, gs_prefix=Path(self.gs_prefix), channel=self.channel
        )

    # def output(self):
    # TODO: could be done fancy via GCS, but that
    # requires httplib2, google-auth, google-api-python-client
    # from luigi.contrib.gcs import GCSTarget; return GCSTarget(self.gs_path)
    # if rewritten, then this task could be a regular luigi.Task
