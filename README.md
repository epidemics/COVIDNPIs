# COVID-19 Nonpharmaceutical Interventions Effectiveness

This repo contains the data and code used for [Brauner et al. *The effectiveness of eight nonpharmaceutical interventions against COVID-19 in 41 countries* (2020)](https://www.medrxiv.org/content/10.1101/2020.05.28.20116129v2.article-info)

## Just here for the data?
The data we collected, including sources, is available in the 'online supplement/data' folder. Please refer to the paper for a description of the data and the data collection process.

## Additional results
Additional results referenced in the paper (effectiveness of NPI combinations, videos on posterior correlation) are located in 'online supplement/results'


# Code
## Install

* [Get Poetry](https://python-poetry.org/docs/#installation)
* Clone this repository.
* Install the dependencies and this lib `poetry install` (creates a virtual env by default).

##
Stay tuned: more detailed documentation coming soon !

`python scripts/sensitivity_dispatcher.py --max_processes 24 --categories structural npi_timing delay_schools any_npi_active agg_holdout growth_noise oxcgrt deaths_threshold cases_threshold npi_leaveout region_leaveout`