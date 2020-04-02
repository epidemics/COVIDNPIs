# Epidemics data processing and modelling toolkit

A data library and a toolkit for modelling COVID-19 epidemics.

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
git clone git@github.com:epidemics/epimodel.git
cd epimodel
poetry install
git clone git@github.com:epidemics/epimodel-covid-data.git data
jupyter notebook  # For example
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
csse = read_csv('data/CSSE.csv')
print(csse.loc[('CZ', "2020-03-28")])
```

## Development

* Use Poetry for dependency management.
* We enforce [black](https://github.com/psf/black) formatting (with the default style).
* Use `pytest` for testing, add tests for your code!
* Use pull requests for both this and the data repository.
