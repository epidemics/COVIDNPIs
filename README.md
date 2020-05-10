# Epidemics data processing and modelling toolkit

A data library and a toolkit for modelling COVID-19 epidemics.

## Issue tracking
Please use the [covid](https://github.com/epidemics/covid/issues/new/choose) repository to submit issues.

## Main concepts

* Region database (continents, countries, provinces, GLEAM basins) - codes, names, basic stats, tree structure (TODO).
* All data, including the region database, is stored in [epimodel-covid-data](https://github.com/epidemics/epimodel-covid-data) repo for asynchronous updates.
* Each region has an ISO-based code, all datasets are organized by those codes (as a row index).
* Built on Pandas with some helpers, using mostly CSVs and HDF5.
* Algorithms and imports assuming common dataframe structure (with `Code` and optionally `Date` row index).
* All dates are UTC timestamps, stored in ISO format with TZ.

## Install

* [Get Poetry](https://python-poetry.org/docs/#installation)
* Clone this repository.
* Install the dependencies and this lib `poetry install` (creates a virtual env by default).
* Clone the [epimodel-covid-data](https://github.com/epidemics/epimodel-covid-data/) repository. For convenience, I recommend cloninig it inside the `epimodel` repo directory as `data`.

```sh
## Clone the repositories (or use their https://... withou github login)
git clone git@github.com:epidemics/epimodel.git
cd epimodel
git clone git@github.com:epidemics/epimodel-covid-data.git data

## Install packages
poetry install  # Best run it outside virtualenv - poetry will create its own
# Alternatively, you can also install PyMC3 or Pyro, and jupyter (in both cases):
poetry install -E pymc3
poetry install -E pyro

## Or, if using conda:
pip install -r requirements.txt

poetry shell # One way to enter the virtualenv (if not active already)
poetry run jupyter notebook  # For example
```

## Basic usage

```python
from epimodel import RegionDataset, read_csv

# Read regions
rds = RegionDataset.load('data/regions.csv')
# Find by name or by code
cz = rds['CZ']
cz = rds.find_one_by_name('Czech Republic')
# Use attribute access on Region
print(cz.Name)
# TODO: attributes for tree-structure access

# Load John Hopkins CSSE dataset with our helper (creates indexes etc.)
csse = read_csv('data/johns-hopkins.csv')
print(csse.loc[('CZ', "2020-03-28")])
```

## Development

* Use Poetry for dependency management.
* We enforce [black](https://github.com/psf/black) formatting (with the default style).
* Use `pytest` for testing, add tests for your code!
* Use pull requests for both this and the data repository.

## Contributing

For the contribution details and project management, please see [this specification](https://www.notion.so/Development-project-management-476f3c53b0f24171a78146365072d82e).

## Running pipeline to get web export

Assuming you've installed deps via `poetry install` and you are in the root epimodel repo.
Also, you did `cp config.yaml config-local.yaml` (modifying it as fit) and set e.g. `export_regions: [CZ, ES]`. Prepend `-C config_local.yaml` to all commands below to use it rather than `config.yaml` (changing `config.yaml` may later conflict with git update).

1. Clone data repo or update it.
   `git clone https://github.com/epidemics/epimodel-covid-data data`

2. Optional: Update Johns Hopkins data `./do -C config-local.yaml update_johns_hopkins` (not needed if you got fresh data from the repo above).

3. Generate batch file from estimates and basic Gleam XML definition.
   `./do generate_gleam_batch default.xml estimates-2020-04-15.csv -D 2020-04-15 -c JK`
   The batch file now contains all the scenario definitions and initial populations.
   Note the estimate input specification may change.

4. Export Gleam simulation XML files in Gleamviz (not while gleamviz is running!).
   `./do export_gleam_batch out/batch-2020-04-16T03:54:52.910001+00:00.hdf5`

5. Start gleamviz. You should see the new simulations loaded. Run all of them and "Retrieve results" (do not export manually). Exit gleamviz.

6. Import the gleamviz results into the HDF batch file.
   `./do import_gleam_batch out/batch-2020-04-16T03:54:52.910001+00:00.hdf5`
   (Gleamviz must be stopped before that.) After this succeeds, you may delete the simulations from gleamviz.

7. Generate web export (additional data are fetched from [config.yml](https://github.com/epidemics/epimodel/blob/master/config.yaml#L16))

   `./do -C config-local.yaml web_export out/batch-2020-04-16T03:54:52.910001+00:00.hdf5 data/sources/estimates-JK-2020-04-15.csv`

8. Export the generated folder to web! Optionally, set a channel for testing first.
   `./do web_upload out/export-2020-04-03T02:03:28.991629+00:00 -c ttest28`

### Gleam Batch file

Has 2-3 dataframes:

* `simulations`: indexed by `SimulationID`, contains information about what simulation ID had what parameters, and the XML definition file.

* `initial_compartments`: Indexed by `['SimulationID', 'Code']`, has the initial sizes of set compartments (columns Exposed, Infected).

* `new_fraction`: After `import_gleam_batch` actually contains the modelled data for Infected and Recovered (columns). Indexed by `['SimulationID', 'Code', 'Date']`:
  * `SimulationID`: corresponding simulation ID to be able to be able to map it to parameters in `simulations`,
  * `Code`: region code (ISOa2 for countries, e.g. `AE`),
  * `Date`: a date for which we model Infected and Recovered.
  Note that the values are *new* elements in the compartment for given day (or in case of resampled dates, in the period since last sample).
