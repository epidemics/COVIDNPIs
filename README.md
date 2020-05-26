# Epidemics data processing and modelling toolkit

A data library and a toolkit for modelling COVID-19 epidemics.

## Issue tracking
Please use the [covid](https://github.com/epidemics/covid/issues/new/choose) repository to submit issues.

## Main concepts

* Region database (continents, countries, provinces, GLEAM basins) - codes, names, basic stats, tree structure (TODO).
* Each region has an ISO-based code, all datasets are organized by those codes (as a row index).
* Built on Pandas with some helpers, using mostly CSVs and HDF5.
* Algorithms and imports assuming common dataframe structure (with `Code` and optionally `Date` row index).
* All dates are UTC timestamps, stored in ISO format with TZ.
* Most of the data is stored in [epimodel-covid-data](https://github.com/epidemics/epimodel-covid-data) repo for asynchronous updates.


## Contributing

For the contribution details and project management, please see [this specification](https://www.notion.so/Development-project-management-476f3c53b0f24171a78146365072d82e).


## Install

* [Get Poetry](https://python-poetry.org/docs/#installation)
* Clone this repository.
* Install the dependencies and this lib `poetry install` (creates a virtual env by default).

```sh
## Clone the repositories (or use their https://... withou github login)
git clone git@github.com:epidemics/epimodel.git
cd epimodel

## Install packages
poetry install  # poetry creates its own virtualenv

poetry shell # activating the virtualenv (if not active already)
poetry run luigi  # running a command in the virtualenv

# Alternatively, you can also install PyMC3 or Pyro, and jupyter (in both cases):
poetry install -E pymc3
poetry install -E pyro
```

## Running the pipeline
We are using [luigi](https://luigi.readthedocs.io/en/stable/index.html) as the workflow framework. This
readme doesn't include description of how to use `luigi` so please refer to the project documentation
to understand how to tweak this.

There is an example of the usage with explanations in `test/example-run.sh` if you want to see a more.

### Example with faked data
This example skips the `UpdateForetold` and `ExtractSimulationsResults` task by providing their output.
In reality, you want to actually run GLEAMviz in between and provide Foretold channel to get data via API.
This by default uses data in `data/inputs` and exports data to `data/outputs/example`.
```
# `poetry shell`  # if you haven't done already
./luigi WebExport \
    --export-name test-export \
    --UpdateForetold-foretold-output data-dir/inputs/fixtures/foretold.csv \
    --ExtractSimulationsResults-models-file data-dir/inputs/fixtures/gleam-models.hdf5 \
    --ExtractSimulationsResults-single-result this-is-now-ignored
```

After the pipeline finishes, you should see the results in `data-dir/outputs/example/`

### The usual flow
You provide all file inputs, foretold_channel and parameters, tweak configs to your liking and then:

1. `./luigi ExportSimulationDefinitions`
2. run GLEAMviz with the simulations created above, retrieve results via it's UI, close it
3. export the data using

    ```
    ./luigi WebExport \
    --export-name my-export \
    --ExtractSimulationsResults-single-result ~/GLEAMviz/data/simulations/82131231323.ghv5/results.h5
    ```

4. upload the result data using `./luigi WebUpload --export-data data-dir/outputs/web-exports/my-export` task

### Actually using it
1. add `foretold_channel` in `luigi.cfg` to `[UpdateForetold]` section. This is a secret and you can get it from others on slack
2. adjust `config.yaml` to your liking, such as scenarios to model or countries to export
3. change `[Configuration].output_directory` to some empty or non-existing folder
4. provide data for any of the ExternalTasks, such as `BaseDefinition`, `ConfigYaml`, `CountryEstimates` and others (see the `epimodel/tasks.py`). If you want to see what does your task depends on, use `luigi-deps-tree` as mentioned above.
5. deal with GLEAMviz and knowing where it's simulation directory is on your installation

### Usage tips
Luigi by default uses `luigi.cfg` from the root of the repository. You can edit it directly or you can created another one and reference it via `LUIGI_CONFIG_PATH=your-config.cfg`. `your-config.cfg` will take precedence over the `luigi.cfg`, so you can change only what's necessary. For example, if you wanted to have a different input and output directory for a specific location run, you could have:

```
# balochistan.cfg

[DEFAULT]
output_directory = my-outputs/balochistan

[RegionsAggregates]
aggregates = data-dir/your-specific-file-for-aggregates.yaml
```

and then `env LUIGI_CONFIG_PATH=balochistan.cfg ./luigi RegionsDatasetTask` would make all outputs go to `my-outputs/balochistan` (instead of the outputs dir from `luigi.cfg`) and the `RegionsAggregates.aggregates` would be taken from the new location too . Parameters from configs can be still overriden from CLI, so `env LUIGI_CONFIG_PATH=balochistan.cfg ./luigi RegionsDatasetTask --RegionsAggregates-aggregates some-other-path.yaml` would still take precedence.


### Getting help
See `epimodel/tasks.py` where the whole pipeline is defined. Read the docstrings and paramter descriptions
to understand and discover all available tasks, their inputs, outputs and their configuration.

You can also use `./luigi <TaskName> --help` or `./luigi <TaskName> --help-all` to get information about the parameters of the task.

`luigi-deps-tree --module epimodel.tasks <TaskName>` enables you to visualize the dependencies and what is and isn't completed. For example:

```
$ luigi-deps-tree --module epimodel.tasks JohnsHopkins --RegionsFile-regions different-regions.csv
└─--[JohnsHopkins-{'hopkins_output': 'data-dir/outputs/john-hopkins.csv'} (PENDING)]
   |--[RegionsFile-{'regions': 'different-regions.csv'} (PENDING)]
   |--[GleamRegions-{'gleams': 'data-dir/inputs/manual/regions-gleam.csv'} (COMPLETE)]
   └─--[RegionsAggregates-{'aggregates': 'data-dir/inputs/manual/regions-agg.yaml'} (COMPLETE)]
```

## Development

* Use Poetry for dependency management.
* We enforce [black](https://github.com/psf/black) formatting (with the default style).
* Use `pytest` for testing, add tests for your code
* Use pull requests for both this and the data repository.

