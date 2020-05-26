#!/usr/bin/env bash
## TODO: this should be moved to a proper test
## was created as part of the https://github.com/epidemics/epimodel/pull/60/
# assuming that:
# * you have cloned the epimodel repository
# * you are in the root
# * you have installed project dependencies via 'poetry install'

# You need to replace <TOKEN> with a foretold_channel
FORETOLD_TOKEN=""

LUIGI="luigi --local-scheduler --module epimodel.tasks"

OUTPUT_DIRECTORY=data-dir/outputs/example-1
rm -rf $OUTPUT_DIRECTORY
mkdir -p $OUTPUT_DIRECTORY

user_continue() {
  # read -p "Press enter to continue..."
  echo ""
}

show_and_do() {
  echo ""
  echo "$@"
  user_continue
  "$@"
}

echo "The main way how to operate the pipeline is via the 'luigi' tool. It loads"
echo "the tasks definition from epimodel.tasks where all inputs, dependencies and outputs"
echo "are defined:"

echo $LUIGI
user_continue

echo ""
echo "All outputs are stored in the $OUTPUT_DIRECTORY. This is defined in the 'luigi.cfg'"
echo "under [Configuration]:output_directory setting. [Configuration]:manual_input specifies"
echo "where to load the manual input files created out of this pipeline. For most of the tasks"
echo "you can override paths manually (or can be added to be able to do so)"
user_continue

echo ""
echo "you can specify a specific task to be executed by providing its name and"
echo "mandatory parameters (if any). Let's try to execute JohnsHopkins task"
show_and_do $LUIGI JohnsHopkins
user_continue

echo ""
echo "If we try it again, luigi will report that there is no work to do because"
echo "it recognized that the output of the previous task is present"
show_and_do $LUIGI JohnsHopkins
user_continue

echo ""
echo "This may not be always what you want. To force luigi to trigger the run again"
echo "you have to either delete the file or (if the task definition allows it) change"
echo "the output."
echo "Let's remove the output '$OUTPUT_DIRECTORY/john-hopkins.csv'"
show_and_do rm $OUTPUT_DIRECTORY/john-hopkins.csv
user_continue

echo ""
echo "And as the dependency tree visualizer will tell you, JohnsHopkins is PENDING"
show_and_do luigi-deps-tree --module epimodel.tasks JohnsHopkins
user_continue

echo ""
echo "So let's try it now again:"
show_and_do $LUIGI JohnsHopkins
user_continue

echo ""
echo "In this case, we could achieve the same with the --hopkins-output parameter":
show_and_do $LUIGI JohnsHopkins --hopkins-output $OUTPUT_DIRECTORY/john-hopkins-2.csv
user_continue

echo ""
echo "Paramters can be set on the command line or via config or env variables"
echo "In the UpdateForetold, we need to pass foretold_channel"
echo "Set this variable at the top of the file - you can ask in Slack for it"
show_and_do $LUIGI UpdateForetold --foretold-channel $FORETOLD_TOKEN
user_continue

echo ""
echo "If you wanted to change a parameter of some upstream task of the task you want to run"
echo "you prefix it with the upstream task name. For example, RegionsFile is an upstream"
echo "task of the JohnsHopkins and the RegionsFile location is fed into JohnsHopkins"
echo "You can change the output of RegionsFile and JohnsHopkins will still work!"
show_and_do luigi-deps-tree --module epimodel.tasks JohnsHopkins --RegionsFile-regions different-regions.csv
user_continue

echo ""
echo "Continuing the pipeline..."
show_and_do $LUIGI GenerateGleamBatch
user_continue

echo ""
SIM_DIR="$OUTPUT_DIRECTORY/simulations"
mkdir -p $SIM_DIR
show_and_do $LUIGI ExportSimulationDefinitions --simulations-dir $SIM_DIR
user_continue

echo ""
FIRST_SIM_DIR=$(ls -1 $SIM_DIR | head -1)
RESULTS_FAKE=$SIM_DIR/$FIRST_SIM_DIR/results.h5
echo "Now faking the results of ImportGleamBatch in $RESULTS_FAKE"
echo ""
touch $RESULTS_FAKE
echo "CAUTION: The following will fail if you haven't retrieved GLEAMviz results."
show_and_do $LUIGI ExtractSimulationsResults --single-result $RESULTS_FAKE
user_continue

echo ""
echo "With some example file though, we can continue to the export for testing purposes"
echo "Using a cached dummy failed for the demonstration purposes"
echo ""
echo "$LUIGI WebExport --export-name test-output"
user_continue
$LUIGI WebExport \
  --ExtractSimulationsResults-single-result $RESULTS_FAKE  \
  --ExtractSimulationsResults-models-file data-dir/inputs/fixtures/gleam-models.hdf5 \
  --export-name test-output \
  --web-export-directory $OUTPUT_DIRECTORY/web-exports

echo ""
echo "And finally upload:"
show_and_do $LUIGI WebUpload --gs-prefix gs://static-covid/static/v4/deleteme --exported-data $OUTPUT_DIRECTORY/web-exports/test-output
