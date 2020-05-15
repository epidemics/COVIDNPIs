import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import dill
import luigi
import yaml
from luigi.util import inherits, requires

from epimodel import Level, RegionDataset, algorithms, imports, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, GleamDefinition
from epimodel.gleam import batch as batch_module

log = logging.getLogger(__name__)


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
    regions_dataset: str = luigi.Parameter(
        config_path=default_from_config("RegionsDatasetTask", "regions_dataset")
    )

    def run(self):
        regions = self.input()["region_file"].path
        gleams = self.input()["gleam_regions"].path
        rds = RegionDataset.load(regions, gleams)
        algorithms.estimate_missing_populations(rds)
        with open(self.regions_dataset, "wb") as ofile:
            dill.dump(rds, ofile)

    def output(self):
        return luigi.LocalTarget(self.regions_dataset)

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
    base_def: str = luigi.Parameter(default="data/various/definition-basic-JK.xml")

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
    comment: str = luigi.Parameter(default="")
    generated_batch_file: str = luigi.Parameter()
    start_date: datetime = luigi.DateParameter(default=datetime.utcnow())
    top: int = luigi.IntParameter(default=2000)

    def requires(self):
        return {
            "base_def": self.clone(BaseDefinition),
            "country_estimates": self.clone(CountryEstimates),
            "regions_dataset": self.clone(RegionsDatasetTask),
            "config_yaml": self.clone(ConfigYaml),
        }

    def output(self):
        return luigi.LocalTarget(self.generated_batch_file)

    def run(self):
        b = Batch.new(path=self.generated_batch_file)
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


@inherits(RegionsDatasetTask, GenerateGleamBatch)
class ExportGleamBatch(luigi.Task):
    stamp_file = "ExportGleamBatch.success"

    exports_dir: str = luigi.Parameter(default="~/GLEAMviz/data/sims/")
    overwrite = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if this file exist in the exports_dir, it's assumed that this tasks has finished
        self.stamp_file_path = Path(self.exports_dir) / self.stamp_file

    def run(self):
        batch_file = self.input()["batch_file"].path
        batch = Batch.open(batch_file)
        gdir = Path(self.exports_dir)
        log.info(
            f"Creating GLEAM XML definitions for batch {batch_file} in dir {gdir} ..."
        )
        batch.export_definitions_to_gleam(
            gdir.expanduser(), overwrite=self.overwrite, info_level=logging.INFO
        )

        # write a dummy stamp file to mark success
        Path(self.stamp_file_path).touch()

    def requires(self):
        return {
            "batch_file": self.clone(GenerateGleamBatch),
            "regions_dataset": self.clone(RegionsDatasetTask),
        }

    def output(self):
        # TODO: improve to get the generated dirs
        return luigi.LocalTarget(self.stamp_file_path)


@requires(ExportGleamBatch)
class GleamvizResults(luigi.ExternalTask):
    """This is done manually by a user via Gleam software"""

    # I expect that this is something like "~/GLEAMviz/data/sims/some-batch-file.hdf5"
    # Or maybe this is not even needed? Confused by the results from gleam
    gleamviz_result = luigi.Parameter(default="blahlah")

    def output(self):
        return luigi.LocalTarget(self.gleamviz_result)


@inherits(GleamvizResults, RegionsDatasetTask, ConfigYaml)
class ImportGleamBatch(luigi.Task):
    allow_missing: bool = luigi.BoolParameter(default=True)
    result_batch_file_path: str = luigi.Parameter(default="data/result-batch-file.hdf5")

    def requires(self):
        return {
            "gleamviz_result": self.clone(GleamvizResults),
            "regions_dataset": self.clone(RegionsDatasetTask),
            "config_yaml": self.clone(ConfigYaml),
        }

    def run(self):
        batch_file = self.input()["gleamviz_result"].path
        # assuming batch file is in the gleamviz results dir
        simulation_directory = os.path.dirname(batch_file)

        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetTask.load_dilled_rds(
            self.input()["regions_dataset"].path
        )

        # copy the gleamviz results into a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        tmp_simulation_dir = Path(temp_dir.name) / "simulations"
        shutil.copytree(simulation_directory, tmp_simulation_dir)

        # work only with the copied data from now on
        tmp_batch_file = Path(tmp_simulation_dir) / os.path.basename(batch_file)
        b = Batch.open(tmp_batch_file)
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
            Path(tmp_simulation_dir),
            Path(tmp_batch_file),
            regions,
            resample=config_yaml["gleam_resample"],
            allow_unfinished=self.allow_missing,
            overwrite=True,
            info_level=logging.INFO,
        )
        # copy the result overwritten batch file to the result export_directory
        shutil.copy(tmp_batch_file, self.result_batch_file_path)

    def output(self):
        return luigi.LocalTarget(self.result_batch_file_path)


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
    "regions_dataset": RegionsDatasetTask,
    "rates": Rates,
    "timezone": Timezones,
    "age_distribution": AgeDistributions,
    "config_yaml": ConfigYaml,
    "country_estimates": CountryEstimates,  # "estimates" in click - is the same?
}


@inherits(*WEB_EXPORT_REQUIRED_TASKS.values())
class WebExport(luigi.Task):
    export_name: str = luigi.Parameter(
        description="Directory name with exported files inside web_export_directory"
    )
    pretty_print: bool = luigi.BoolParameter(
        default=False, description="If true, result JSONs are indented by 4 spaces"
    )
    web_export_directory: str = luigi.Parameter(
        default="web-exports", description="Root directory for all exports"
    )
    main_data_filename: str = luigi.Parameter(
        default="data-v4.json",
        description="The default name of the main JSON data file",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_export_path = Path(self.web_export_directory) / self.export_name

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
            self.full_export_path,
            Path(self.main_data_filename),
            latest="latest",
            pretty_print=self.pretty_print,
        )

    def output(self):
        return luigi.LocalTarget(self.full_export_path / self.main_data_filename)

    def requires(self):
        return {
            name: self.clone(task) for name, task in WEB_EXPORT_REQUIRED_TASKS.items()
        }


@requires(WebExport)
class WebUpload(luigi.Task):
    gs_prefix: str = luigi.Parameter(default="gs://static-covid/static/v4/")
    channel: str = luigi.Parameter(default="main")

    # this together with setting this in self.run and self.complete guarantees
    # that this task always run
    is_complete = False

    def run(self):
        main_data_file = self.input().path
        # directory with all the exported outputs
        base_dir = os.path.dirname(main_data_file)
        upload_export(base_dir, gs_prefix=Path(self.gs_prefix), channel=self.channel)
        self.is_complete = True

    def complete(self):
        return self.is_complete

    # def output(self):
    # TODO: could be done fancy via GCS, but that
    # requires httplib2, google-auth, google-api-python-client
    # from luigi.contrib.gcs import GCSTarget; return GCSTarget(self.gs_path)
    # if rewritten, then this task could be a regular luigi.Task
