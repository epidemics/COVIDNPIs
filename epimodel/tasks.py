import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import dill
import luigi
import yaml
from luigi.util import inherits
from luigi.configuration import get_config

from epimodel import Level, RegionDataset, algorithms, imports, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, GleamDefinition
from epimodel.gleam import batch as batch_module

logger = logging.getLogger(__name__)


class RegionsFile(luigi.ExternalTask):
    """Default regions database used for various country handling"""

    regions = luigi.Parameter(
        description="Input filename relative to the config directory",
    )

    def output(self):
        return luigi.LocalTarget(self.regions)


class GleamRegions(luigi.ExternalTask):
    """Definition of Gleamviz regions"""

    gleams = luigi.Parameter(
        description="Input filename relative to the config directory",
    )

    def output(self):
        return luigi.LocalTarget(self.gleams)


class RegionsAggregates(luigi.ExternalTask):
    """Aggregates used for locations like Balochistan and others"""

    aggregates = luigi.Parameter(
        description="Input filename relative to the config directory",
    )

    def output(self):
        return luigi.LocalTarget(self.aggregates)


class RegionsDatasetTask(luigi.Task):
    """Combines several inputs into a RegionDataset object used in several
    downstream tasks for handling ISO codes and others"""

    regions_dataset: str = luigi.Parameter(
        description="Output filename of the exported data.",
    )

    def requires(self):
        return {
            "region_file": RegionsFile(),
            "gleam_regions": GleamRegions(),
            "aggregates": RegionsAggregates(),
        }

    def output(self):
        return luigi.LocalTarget(self.regions_dataset)

    def run(self):
        regions = self.input()["region_file"].path
        gleams = self.input()["gleam_regions"].path
        aggregates = self.input()["aggregates"].path
        rds = RegionDataset.load(regions, gleams, aggregates)
        algorithms.estimate_missing_populations(rds)
        with open(self.regions_dataset, "wb") as ofile:
            dill.dump(rds, ofile)

    @staticmethod
    def load_dilled_rds(path: str):
        with open(path, "rb") as ifile:
            return dill.load(ifile)


class JohnsHopkins(luigi.Task):
    """Downloads data from Johns Hopkins github and exports them as CSV"""

    hopkins_output: str = luigi.Parameter(
        description="Output filename of the exported data relative to config output dir.",
    )

    def requires(self):
        return {"regions": RegionsDatasetTask()}

    def output(self):
        return luigi.LocalTarget(self.hopkins_output)

    def run(self):
        rds = RegionsDatasetTask.load_dilled_rds(self.input()["regions"].path)
        csse = imports.import_johns_hopkins(rds)
        csse.to_csv(self.hopkins_output)
        logger.info(
            f"Saved CSSE to {self.hopkins_output}, last day is {csse.index.get_level_values(1).max()}"
        )


class UpdateForetold(luigi.Task):
    """Exports prediction data form the Foretold platform and
    dumps them into a CSV. These are part of the inputs to the gleamviz model.
    """

    foretold_output: str = luigi.Parameter(
        description="Output filename of the exported data relative to output dir.",
    )
    foretold_channel: str = luigi.Parameter(
        description="The secret to fetch data from Foretold via API",
    )

    def requires(self):
        return {
            "regions": RegionsDatasetTask(),
        }

    def output(self):
        return luigi.LocalTarget(self.foretold_output)

    def run(self):
        if (
            not isinstance(self.foretold_channel, str)
            or len(self.foretold_channel) < 20
        ):
            raise ValueError(
                "Foretold channel is either not a string or is too short to be valid"
            )

        logger.info("Downloading and parsing foretold")
        rds = RegionsDatasetTask.load_dilled_rds(self.input()["regions"].path)
        foretold = imports.import_foretold(rds, self.foretold_channel)
        foretold.to_csv(self.foretold_output, float_format="%.7g")
        logger.info(f"Saved Foretold to {self.foretold_output}")


class BaseDefinition(luigi.ExternalTask):
    """Base 'template' XML definition for gleamviz simulations."""

    base_def: str = luigi.Parameter(
        description="Path to the input file relative to the configuration input directory",
    )

    def output(self):
        return luigi.LocalTarget(self.base_def)


class CountryEstimates(luigi.ExternalTask):
    """Estimates created manually by forecasters"""

    country_estimates: str = luigi.Parameter(
        description="Path to the input file relative to the configuration input directory",
    )

    def output(self):
        return luigi.LocalTarget(self.country_estimates)


class ConfigYaml(luigi.ExternalTask):
    """Configuration yaml used mostly to customize the gleamviz pipeline and to generate
    the definitions for the simulations"""

    yaml_config_path: str = luigi.Parameter(
        description="Path to the input file relative to the configuration input directory",
    )

    def output(self):
        return luigi.LocalTarget(self.yaml_config_path)

    @staticmethod
    def load(path):
        with open(path, "rt") as f:
            return yaml.safe_load(f)


class GenerateGleamBatch(luigi.Task):
    """Generates a an HDF file similar to what gleamviz outputs.

    The HDF has 2-3 dataframes:

    * `simulations`: indexed by `SimulationID`, contains information
    about what simulation ID had what parameters, and the XML definition file.

    * `initial_compartments`: Indexed by `['SimulationID', 'Code']`, has the
    initial sizes of set compartments (columns Exposed, Infected).

    * `new_fraction`: After `import-gleam-batch` actually contains the
    modelled data for Infected and Recovered (columns). Indexed by `['SimulationID', 'Code', 'Date']`:
      * `SimulationID`: corresponding simulation ID to be able to be
      able to map it to parameters in `simulations`,
      * `Code`: region code (ISOa2 for countries, e.g. `AE`),
      * `Date`: a date for which we model Infected and Recovered.
      Note that the values are *new* elements in the compartment for
      given day (or in case of resampled dates, in the period since last sample).
    """

    generated_batch_filename: str = luigi.Parameter(
        description="Output filename of the generated batch file for gleam",
    )
    top: int = luigi.IntParameter(description="Upper limit for seed compartments.")
    start_date: datetime = luigi.DateParameter(
        default=datetime.utcnow(), description="Start date of the simulations"
    )

    def requires(self):
        return {
            "base_def": self.clone(BaseDefinition),
            "country_estimates": self.clone(CountryEstimates),
            "regions_dataset": self.clone(RegionsDatasetTask),
            "config_yaml": self.clone(ConfigYaml),
        }

    def output(self):
        return luigi.LocalTarget(self.generated_batch_filename)

    def run(self):
        # cleaning up in the case of incomplete runs
        try:
            self._run()
        except:
            if os.path.exists(self.generated_batch_filename):
                os.remove(self.generated_batch_filename)

    def _run(self):
        b = Batch.new(path=self.generated_batch_filename)
        logger.info(f"New batch file {b.path}")

        base_def = self.input()["base_def"].path
        logger.info(f"Reading base GLEAM definition {base_def} ...")
        d = GleamDefinition(base_def)

        country_estimates = self.input()["country_estimates"].path
        rds = RegionsDatasetTask.load_dilled_rds(self.input()["regions_dataset"].path)
        logger.info(f"Reading estimates from CSV {country_estimates} ...")
        est = read_csv_smart(country_estimates, rds, levels=Level.country)
        start_date = (
            utils.utc_date(self.start_date) if self.start_date else d.get_start_date()
        )
        logger.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
        batch_module.generate_simulations(
            b,
            d,
            est,
            rds=rds,
            config=ConfigYaml.load(self.input()["config_yaml"].path),
            start_date=start_date,
            top=self.top,
        )
        logger.info(f"Generated batch {b.path!r}:\n  {b.stats()}")
        b.close()


class GenerateSimulationDefinitions(luigi.Task):
    """
    Creates definitions for simulations which can be used for gleamviz. It
    must not be run when Gleamviz is running otherwise it won't be visible.

    Formerly ExportGleamBatch"""

    simulations_dir: str = luigi.Parameter(
        description=(
            "Where to output the gleamviz input files. Can be "
            "set directly to '~/GLEAMviz/data/simulations/' if you "
            "do not want to copy it there manually later on"
        ),
    )

    stamp_file = "GenerateSimulationDefinitions.success"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if this file exist in the simulations_dir,
        # it's assumed that this tasks has finished
        self.stamp_file_path = Path(self.simulations_dir, self.stamp_file)

    def requires(self):
        return GenerateGleamBatch()

    def output(self):
        # TODO: improve this to actually capture gleamviz input directories
        return luigi.LocalTarget(self.stamp_file_path)

    def run(self):
        batch_file = self.input().path
        batch = Batch.open(batch_file)
        logger.info(
            f"Creating GLEAM XML definitions for batch {batch_file} in dir {self.simulations_dir} ..."
        )
        batch.export_definitions_to_gleam(
            Path(self.simulations_dir).expanduser(),
            overwrite=False,
            info_level=logging.INFO,
        )

        # write a dummy stamp file to mark success
        Path(self.stamp_file_path).touch()


class GleamvizResults(luigi.ExternalTask):
    """This is done manually by a user via Gleam software. You should see the new
    simulations loaded. Run all of them and "Retrieve results"
    (do not export manually). Exit gleamviz."""

    single_result = luigi.Parameter(
        description=(
            "A path to any one `results.h5` gleamviz files you downloaded "
            "via 'retrieve results' in the gleam software. For example, it "
            "could be something like "
            "'~/GLEAMviz/data/simulations/82131231323.ghv5/results.h5'"
        )
    )

    def requires(self):
        return GenerateSimulationDefinitions()

    def output(self):
        return luigi.LocalTarget(self.single_result)


@inherits(GleamvizResults)
class ExtractSimulationsResults(luigi.Task):
    """
    Exports data from the gleamviz results. Gleamviz must be stopped before that.

    After this succeeds, you may delete the simulations from gleamviz.
    Formerly ImportGleamBatch
    """

    models_file: str = luigi.Parameter(
        description="Name of the output HDF file with all traces",
    )
    allow_missing: bool = luigi.BoolParameter(default=True)

    def requires(self):
        return {
            "gleamviz_result": self.clone(GleamvizResults),
            "batch_file": GenerateGleamBatch(),
            "regions_dataset": RegionsDatasetTask(),
            "config_yaml": ConfigYaml(),
        }

    def output(self):
        return luigi.LocalTarget(self.models_file)

    def run(self):
        batch_file = self.input()["batch_file"].path

        simulation_directory = os.path.dirname(self.input()["gleamviz_result"].path)

        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetTask.load_dilled_rds(
            self.input()["regions_dataset"].path
        )

        # copy the batch file into a temporary one
        temp_dir = tempfile.TemporaryDirectory()
        tmp_batch_file = Path(temp_dir.name) / "batch.hdf"
        shutil.copy(batch_file, tmp_batch_file)
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

        logger.info(
            f"Importing results for {len(regions)} from GLEAM into {batch_file} ..."
        )
        b.import_results_from_gleam(
            Path(simulation_directory),
            regions,
            resample=get_config()["DEFAULT"]["gleam_resample"],
            allow_unfinished=self.allow_missing,
            info_level=logging.INFO,
        )
        # copy the result overwritten batch file to the result export_directory
        shutil.copy(tmp_batch_file, self.models_file)


class Rates(luigi.ExternalTask):
    """Rates for number of critical beds and hospital capacity"""

    rates: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.rates)


class Timezones(luigi.ExternalTask):
    """Timezones per country"""

    timezones: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.timezones)


class AgeDistributions(luigi.ExternalTask):
    """Distributions of ages in given countries"""

    age_distributions: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.age_distributions)


class WebExport(luigi.Task):
    """Generates export used by the website."""

    export_name: str = luigi.Parameter(
        description="Directory name with exported files inside web_export_directory"
    )
    pretty_print: bool = luigi.BoolParameter(
        description="If true, result JSONs are indented by 4 spaces"
    )
    web_export_directory: str = luigi.Parameter(
        description="Root subdirectory for all exports.",
    )
    main_data_filename: str = luigi.Parameter(
        description="The default name of the main JSON data file",
    )
    comment: str = luigi.Parameter(description="Optional comment to the export",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_export_path = Path(self.web_export_directory, self.export_name)

    def requires(self):
        return {
            "models": ExtractSimulationsResults(),
            "hopkins": JohnsHopkins(),
            "foretold": UpdateForetold(),
            "regions_dataset": RegionsDatasetTask(),
            "rates": Rates(),
            "timezones": Timezones(),
            "age_distributions": AgeDistributions(),
            "config_yaml": ConfigYaml(),
            "country_estimates": CountryEstimates(),
        }

    def output(self):
        return luigi.LocalTarget(self.full_export_path / self.main_data_filename)

    def run(self):
        models = self.input()["models"].path
        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetTask.load_dilled_rds(
            self.input()["regions_dataset"].path
        )
        estimates = self.input()["country_estimates"].path

        ex = process_export(
            self.input(),
            regions_dataset,
            False,
            self.comment,
            models,
            estimates,
            config_yaml["export_regions"],
            config_yaml["state_to_country"],
            config_yaml["groups"],
            get_config()["DEFAULT"]["gleam_resample"],
        )
        ex.write(
            self.full_export_path,
            Path(self.main_data_filename),
            latest="latest",
            pretty_print=self.pretty_print,
        )


# @requires(WebExport)  # this would require gleamviz-result parameter, I think
# it's not needed and the cost of adding the parameter is a good price
class WebUpload(luigi.Task):
    """Uploads the exported files into GCS bucket"""

    gs_prefix: str = luigi.Parameter(description="A GCS default path for the export",)
    channel: str = luigi.Parameter(
        description="channel to load the data to, basically a subdirectory in gcs_path",
    )
    main_data_file: str = luigi.Parameter(
        description="Path to the main datafile from web-export"
    )

    # this together with setting this in self.run and evaluating in self.complete
    # guarantees that this task always run
    # could be replaced by "stamp_file" approach
    is_complete = False

    def run(self):
        # main_data_file = self.input().path
        # directory with all the exported outputs
        base_dir = os.path.dirname(self.main_data_file)
        upload_export(Path(base_dir), gs_prefix=self.gs_prefix, channel=self.channel)
        self.is_complete = True

    def complete(self):
        return self.is_complete

    # def output(self):
    # TODO: could be done fancy via GCS, but that
    # requires httplib2, google-auth, google-api-python-client
    # from luigi.contrib.gcs import GCSTarget; return GCSTarget(self.gs_path)
    # if rewritten, then this task could be a regular luigi.Task
