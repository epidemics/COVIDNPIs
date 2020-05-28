import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import luigi
import yaml
from luigi.util import inherits

from epimodel import Level, RegionDataset, algorithms, imports
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch
from typing import Dict

logger = logging.getLogger(__name__)

# TODO: could be done in tasks so they are forgiving/creating the
# directory tree by themselves
default_config = luigi.configuration.get_config()["DEFAULT"]
output_dir = default_config.get("output_directory")
if output_dir:
    logger.debug("Creating the output directory %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)


class RegionsFile(luigi.ExternalTask):
    """Default regions database used for various country handling"""

    regions = luigi.Parameter(
        description="Input filename relative to the config directory",
    )

    def output(self):
        return luigi.LocalTarget(self.regions)


class GleamRegions(luigi.ExternalTask):
    """Definition of GLEAMviz basins"""

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


class RegionsDatasetSubroutine:
    """
    Combines several inputs into a RegionDataset object used in several
    downstream tasks for handling ISO codes and others.

    This is not an actual task but a subroutine that encapsulates the
    inputs and process for RegionDataset creation. It fails as an
    independent task because the regions model is unable to to
    pickle/unpickle successfully.
    """

    @staticmethod
    def requires():
        return {
            "region_file": RegionsFile(),
            "gleam_regions": GleamRegions(),
            "aggregates": RegionsAggregates(),
        }

    @staticmethod
    def load_rds(task):
        regions = task.input()["region_file"].path
        gleams = task.input()["gleam_regions"].path
        aggregates = task.input()["aggregates"].path
        logger.info(f"Loading regions from {regions}, {gleams}, {aggregates}...")
        rds = RegionDataset.load(regions, gleams, aggregates)
        logger.info(f"Estimating missing populations...")
        algorithms.estimate_missing_populations(rds)
        logger.info(f"Regions sucessfully loaded.")
        return rds


class JohnsHopkins(luigi.Task):
    """Downloads data from Johns Hopkins github and exports them as CSV"""

    hopkins_output: str = luigi.Parameter(
        description="Output filename of the exported data relative to config output dir.",
    )

    def requires(self):
        return RegionsDatasetSubroutine.requires()

    def output(self):
        return luigi.LocalTarget(self.hopkins_output)

    def run(self):
        rds = RegionsDatasetSubroutine.load_rds(self)
        csse = imports.import_johns_hopkins(rds)
        csse.to_csv(self.hopkins_output)
        logger.info(
            f"Saved CSSE to {self.hopkins_output}, last day is {csse.index.get_level_values(1).max()}"
        )


class SerialIntervalSample(luigi.ExternalTask):
    """Serial Interval data imported manually"""

    serial_interval_sample: str = luigi.Parameter(
        description="Path to the input file relative to the configuration input directory",
    )

    def output(self):
        return luigi.LocalTarget(self.serial_interval_sample)


class EstimateR(luigi.Task):
    """
    R estimation script for all countries in the JohnHopkins database.
    !! This task usually takes about an 30 minutes to complete
    """

    r_estimates_output: str = luigi.Parameter(
        description="Output filename of the estimates file relative to config output dir.",
    )
    r_executable_path: str = luigi.Parameter(
        description="Path to the Rscript executable (/usr/bin/Rscript default).",
    )

    def requires(self):
        return {
            "jhdata": JohnsHopkins(),
            "si_sample": SerialIntervalSample(),
            "config_yaml": ConfigYaml(),
            **RegionsDatasetSubroutine.requires(),
        }

    def output(self):
        return luigi.LocalTarget(self.r_estimates_output)

    def run(self):
        john_hopkins_path = self.input()["jhdata"].path
        serial_interval_file = self.input()["si_sample"].path
        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetSubroutine.load_rds(self)

        algorithms.estimate_r(
            self.r_executable_path,
            self.r_estimates_output,
            john_hopkins_path,
            serial_interval_file,
            regions_dataset,
            config_yaml["state_to_country"],
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
        return RegionsDatasetSubroutine.requires()

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
        rds = RegionsDatasetSubroutine.load_rds(self)
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


class GleamParameters(luigi.ExternalTask):
    """Configuration parameters for GLEAMviz simulations"""

    gleam_parameters: str = luigi.Parameter(
        description="Path to the input file relative to the configuration input directory",
    )

    def output(self):
        return luigi.LocalTarget(self.gleam_parameters)


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
            config = yaml.safe_load(f)

        export_regions = {}
        for item in config["export_regions"]:
            if isinstance(item, Dict):
                CODE_KEY = "code"
                if not CODE_KEY in item:
                    raise Exception(
                        "Malformed entry in `export_regions`, expected key `{}'. Entry: {}".format(
                            CODE_KEY, item
                        )
                    )
                code = item[CODE_KEY]
                del item[CODE_KEY]
                export_regions[code] = item
            else:
                export_regions[item] = {}

        config["export_regions"] = export_regions

        return config


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

    def requires(self):
        return {
            "base_def": self.clone(BaseDefinition),
            "gleam_parameters": self.clone(GleamParameters),
            "country_estimates": self.clone(CountryEstimates),
            "config_yaml": self.clone(ConfigYaml),
            **RegionsDatasetSubroutine.requires(),
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
            raise

    def _run(self):
        batch = Batch.new(path=self.generated_batch_filename)
        logger.info(f"New batch file {batch.path}")
        rds = RegionsDatasetSubroutine.load_rds(self)

        logger.info(f"Generating scenarios...")
        batch.generate_simulations(
            ConfigYaml.load(self.input()["config_yaml"].path),
            self.input()["base_def"].path,
            self.input()["gleam_parameters"].path,
            self.input()["country_estimates"].path,
            rds,
        )
        logger.info(f"Generated batch scenarios {batch.path!r}:\n  {batch.stats()}")
        batch.close()


class ExportSimulationDefinitions(luigi.Task):
    """
    Saves the generated definition.xml files for simulations directly into
    the GLEAMviz data folder. You may need to close and reopen GLEAMviz in
    order for these to show up in the dashboard.

    Formerly ExportGleamBatch"""

    simulations_dir: str = luigi.Parameter(
        description=(
            "Where to output the gleamviz input files. Can be "
            "set directly to '~/GLEAMviz/data/simulations/' if you "
            "do not want to copy it there manually later on"
        ),
    )

    stamp_file = "ExportSimulationDefinitions.success"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if this file exist in the simulations_dir,
        # it's assumed that this tasks has finished
        self.stamp_file_path = Path(self.simulations_dir, self.stamp_file).expanduser()

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
        self.stamp_file_path.touch()


class GleamvizResultsTarget(luigi.Target):
    def __init__(self, path):
        self.path = path

    def exists(self):
        for _, _, files in os.walk(self.path, topdown=False):
            if "results.h5" in files:
                return True
        return False


class GleamvizResults(luigi.ExternalTask):
    """This is done manually by a user via Gleam software. You should see the new
    simulations loaded. Run all of them and "Retrieve results"
    (do not export manually). Exit gleamviz."""

    simulation_directory = luigi.Parameter(
        description=(
            "A path to the gleamviz simulation directory. For example, it "
            "could be something like '~/GLEAMviz/data/simulations/'"
        )
    )

    def requires(self):
        return ExportSimulationDefinitions()

    def output(self):
        return GleamvizResultsTarget(Path(self.simulation_directory).expanduser())


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
    resample: bool = luigi.Parameter(description="Default pandas resample for dates")

    def requires(self):
        return {
            "gleamviz_result": self.clone(GleamvizResults),
            "batch_file": GenerateGleamBatch(),
            "config_yaml": ConfigYaml(),
            **RegionsDatasetSubroutine.requires(),
        }

    def output(self):
        return luigi.LocalTarget(self.models_file)

    def run(self):
        batch_file = self.input()["batch_file"].path

        simulation_directory = self.input()["gleamviz_result"].path

        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetSubroutine.load_rds(self)

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
        for region_code in config_yaml["export_regions"].keys():
            region = regions_dataset[region_code]
            if region.GleamID != "":
                regions.add(region)

        logger.info(
            f"Importing results for {len(regions)} from GLEAM into {batch_file} ..."
        )
        b.import_results_from_gleam(
            Path(simulation_directory),
            regions,
            resample=self.resample,
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
    resample: str = luigi.Parameter(description="Pandas dataseries resample")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_export_path = Path(self.web_export_directory, self.export_name)

    def requires(self):
        return {
            "models": ExtractSimulationsResults(),
            "hopkins": JohnsHopkins(),
            "foretold": UpdateForetold(),
            "rates": Rates(),
            "timezones": Timezones(),
            "age_distributions": AgeDistributions(),
            "config_yaml": ConfigYaml(),
            "country_estimates": CountryEstimates(),
            "r_estimates": EstimateR(),
            **RegionsDatasetSubroutine.requires(),
        }

    def output(self):
        return luigi.LocalTarget(self.full_export_path / self.main_data_filename)

    def run(self):
        models = self.input()["models"].path
        config_yaml = ConfigYaml.load(self.input()["config_yaml"].path)
        regions_dataset = RegionsDatasetSubroutine.load_rds(self)
        estimates = self.input()["country_estimates"].path

        ex = process_export(
            self.input(),
            regions_dataset,
            False,
            self.comment,
            models,
            estimates,
            config_yaml,
            self.resample,
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
    export_data: str = luigi.Parameter(
        description="Full path to the exported data. E.g. `outputs/web-exports/latest"
    )

    # this together with setting this in self.run and evaluating in self.complete
    # guarantees that this task always run
    # could be replaced by "stamp_file" approach
    is_complete = False

    def run(self):
        # main_data_file = self.input().path
        # directory with all the exported outputs
        export_path = Path(self.export_data)
        if not export_path.exists():
            raise IOError(
                f"'{export_path}' directory does not exist. Provide existing directory."
            )
        upload_export(export_path, gs_prefix=self.gs_prefix, channel=self.channel)
        self.is_complete = True

    def complete(self):
        return self.is_complete

    # def output(self):
    # TODO: could be done fancy via GCS, but that
    # requires httplib2, google-auth, google-api-python-client
    # from luigi.contrib.gcs import GCSTarget; return GCSTarget(self.gs_path)
    # if rewritten, then this task could be a regular luigi.Task
