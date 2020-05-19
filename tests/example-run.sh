#!/usr/bin/env bash
## TODO: this should be moved to a proper test
## was created as part of the https://github.com/epidemics/epimodel/pull/60/
# assuming that:
# * you have cloned the epimodel repository
# * you are in the root
# * you have installed project dependencies via 'poetry install'

# You need to replace <TOKEN> with a foretold_channel
FORETOLD_TOKEN=""

#CONTINUE="read -p Press-enter"
CONTINUE="echo"

LUIGI="luigi --local-scheduler --module epimodel.tasks"

OUTPUT_DIRECTORY=data-dir/outputs/example-1
rm -rf $OUTPUT_DIRECTORY
mkdir -p $OUTPUT_DIRECTORY


echo "The main way how to operate the pipeline is via the 'luigi' tool. It loads"
echo "the tasks definition from epimodel.tasks where all inputs, dependencies and outputs"
echo "are defined:"
echo $LUIGI
$CONTINUE

echo "All outputs are stored in the $OUTPUT_DIRECTORY. This is defined in the 'luigi.cfg'"
echo "under [Configuration]:output_directory setting. [Configuration]:manual_input specifies"
echo "where to load the manual input files created out of this pipeline. For most of the tasks"
echo "you can override paths manually (or can be added to be able to do so)"
$CONTINUE

echo ""
echo "you can specify a specific task to be executed by providing its name and"
echo "mandatory parameters (if any). Let's try to execute JohnsHopkins task"
$CONTINUE
echo ""
echo "$LUIGI JohnsHopkins"
echo ""
$LUIGI JohnsHopkins


echo ""
echo "If we try it again, luigi will report that there is no work to do because"
echo "it recognized that the output of the previous task is present"
echo ""
echo "$LUIGI JohnsHopkins"
echo ""
$CONTINUE
$LUIGI JohnsHopkins

echo ""
echo "This may not be always what you want. To force luigi to trigger the run again"
echo "you have to either delete the file or (if the task definition allows it) change"
echo "the output."
echo "Let's remove the output '$OUTPUT_DIRECTORY/john-hopkins.csv'"
echo ""
echo "rm $OUTPUT_DIRECTORY/john-hopkins.csv"
$CONTINUE
rm "$OUTPUT_DIRECTORY/john-hopkins.csv"

echo ""
echo "And as the dependency tree visualizer will tell you, JohnsHopkins is PENDING"
echo "luigi-deps-tree --module epimodel.tasks JohnsHopkins"
luigi-deps-tree --module epimodel.tasks JohnsHopkins
$CONTINUE

echo ""
echo "So let's try it now again:"
echo "$LUIGI JohnsHopkins"
echo ""
$CONTINUE
$LUIGI JohnsHopkins

echo ""
echo "In this case, we could achieve the same with the --hopkins-output parameter":
echo "$LUIGI JohnsHopkins --hopkins-output john-hopkins-2.csv"
$CONTINUE
$LUIGI JohnsHopkins --hopkins-output $OUTPUT_DIRECTORY/john-hopkins-2.csv
ls -la $OUTPUT_DIRECTORY/john-hopkins-2.csv

echo ""
echo "Paramters can be set on the command line or via config or env variables"
echo "In the UpdateForetold, we need to pass foretold_channel"
echo "Set this variable at the top of the file - you can ask in Slack for it"
$LUIGI UpdateForetold --foretold-channel $FORETOLD_TOKEN

echo ""
echo "If you wanted to change a parameter of some upstream task of the task you want to run"
echo "you prefix it with the upstream task name. For example, RegionDatasetTask is an upstream"
echo "task of the JohnsHopkins and output of RegionDatasetTask is fed into JohnsHopkins"
echo "You can change the output of RegionsDatasetTask and JohnsHopkins will still work!"
echo "luigi-deps-tree --module epimodel.tasks JohnsHopkins --RegionsDatasetTask-regions-dataset different-input"


echo ""
echo "Continuing the pipeline..."
$CONTINUE
echo ""
echo "$LUIGI GenerateGleamBatch"
$LUIGI GenerateGleamBatch

echo ""

SIM_DIR="$OUTPUT_DIRECTORY/simulations"
mkdir -p $SIM_DIR
CMD_GSD="$LUIGI GenerateSimulationDefinitions --simulations-dir $SIM_DIR"
echo "$CMD_GSD"
echo ""
$CMD_GSD


FIRST_SIM_DIR=$(ls -1 $SIM_DIR | head -1)
RESULTS_FAKE=$SIM_DIR/$FIRST_SIM_DIR/results.h5
echo "Now faking the results of ImportGleamBatch in $RESULTS_FAKE"
touch $RESULTS_FAKE
CMD_ESR="$LUIGI ExtractSimulationsResults --single-result $RESULTS_FAKE"
echo "$CMD_ESR"
echo "CAUTION: This is going to fail if you haven't retrieved GLEAMviz results"
$CONTINUE
$CMD_ESR

echo "With some example file though, we can continue to the export for testing purposes"
echo "Using a cached dummy failed for the demonstration purposes"
$CONTINUE
$LUIGI WebExport \
  --ExtractSimulationsResults-single-result $RESULTS_FAKE  \
  --ExtractSimulationsResults-models-file data-dir/inputs/fixtures/gleam-models.hdf5 \
  --export-name test-output \
  --web-export-directory $OUTPUT_DIRECTORY/web-exports

echo "And finally upload:"
$CONTINUE
$LUIGI WebUpload --gs-prefix gs://static-covid/static/v4/deleteme --exported-data $OUTPUT_DIRECTORY/web-exports/test-output
