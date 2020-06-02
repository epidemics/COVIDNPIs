#!/usr/bin/env bash
# assuming that:
# * you have cloned the epimodel repository
# * you are in the root
# * you have installed project dependencies via 'poetry install'

prompt_input() {
 read -p "$1"": " val
 echo $val
}

user_continue() {
  read -p "Press enter to continue..."
  echo ""
}

show_and_do() {
  echo ""
  echo "$@"
  user_continue
  "$@"
}

show_and_do_luigi() {
  echo ""
  echo "${LUIGI} $@"
  user_continue
  $LUIGI "$@"
}


LUIGI="./run_luigi"
FORETOLD_TOKEN=$(prompt_input "Foretold token")
SIM_DIR=$(prompt_input "GLEAMviz simulation directory (for example ~/GLEAMviz-data/sims)")
OUTPUT_DIRECTORY=./data-dir/outputs

cat << EOM > secrets.cfg
[UpdateForetold]
foretold_channel = ${FORETOLD_TOKEN}

[ExportSimulationDefinitions]
simulations_dir = ${SIM_DIR}

[ExtractSimulationsResults]
simulation_directory = ${SIM_DIR}
EOM

cat << EOM
The supplied configuration has been written to secrets.cfg
EOM

cat << EOM
The main way how to operate the pipeline is via the 'luigi' tool. It loads"
the tasks definition from epimodel.tasks where all inputs, dependencies and outputs"
are defined:
   ${LUIGI}

All outputs are stored in the ${OUTPUT_DIRECTORY}. This is defined in the 'luigi.cfg'
under [Configuration]:output_directory setting. [Configuration]:manual_input specifies
where to load the manual input files created out of this pipeline. For most of the tasks
you can override paths manually (or can be added to be able to do so).
EOM
user_continue

cat << EOM
The luigi pipeline has 2 main tasks to run:
 * ExportSimulationDefinitions
 * WebExport

The ExportSimulationDefinitions generates the GLEAMviz definition files from all the required dependencies.
The WebExport uses the GLEAMviz results to generate the web export.
Between these 2 steps you have to run GLEAMviz manually, run the simulations and retrieve their results.

You can see all the dependencies of the ExportSimulationDefinitions in the dependency graph:
EOM

luigi-deps-tree --module epimodel.tasks ExportSimulationDefinitions --simulations-dir $SIM_DIR

cat << EOM


To run the ExportSimulationDefinitions the only parameters you need to set is the simulations directory parameter:
--simulations-dir - your local GLEAMviz simulations folder:
 * on unix probably ~/GLEAMviz/sims
 * on windows probably C://Users/username/GLEAMviz/sims
EOM

show_and_do_luigi ExportSimulationDefinitions
user_continue

cat << EOM


After the simulation definitions are generated, you HAVE to run GLEAMviz.
* run the simulations
* retrieve results
After this is done you are ready to create the WebExport.
You can see the WebExport dependencies below.
EOM

luigi-deps-tree --module epimodel.tasks WebExport \
  --ExtractSimulationsResults-simulation-directory "${SIM_DIR}" \
  --ExportSimulationDefinitions-simulations-dir "${SIM_DIR}" \
  --UpdateForetold-foretold-channel "${FORETOLD_TOKEN}" \
  --export-name test-export

cat << EOM


The WebExport has several parameters that need to be set:
* --export-name: name of the export
EOM

show_and_do_luigi WebExport \
  --export-name test-export

cat << EOM


The final step of the pipeline is the WebUpload step. The parameters here are:
 * --channel: which channel to upload to
 * --exported-data: the path to the export to upload, by default its data-dir/outputs/web-exports/{name of export}
EOM

show_and_do_luigi WebUpload \
  --gs-prefix gs://static-covid/static/v4/ \
  --channel test-export \
  --exported-data $OUTPUT_DIRECTORY/web-exports/test-export
