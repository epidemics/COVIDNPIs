#!/bin/sh

# could be replaced by submodules
git clone https://github.com/epidemics/epimodel-covid-data data

./do -C config.yaml update_john_hopkins

# download latest batch file assuming it's there from someone
# else who ran gleamviz, would need auth
gsutil cp gs://.../batches/latest.hdf latest.hdf

./do web_export latest.hdf data/source/estimates-JK-2020-04-15.csv

./do web_upload out/export -c main 
