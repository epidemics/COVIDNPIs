#!/usr/bin/sh
# assuming that:
# * you have cloned the epimodel repository
# * you are in the root
# * you have installed project dependencies via `poetry install`

FORETOLD_TOKEN=<TOKEN>

#CONTINUE="read -p Press-enter"
CONTINUE="echo"

OUTPUT_DIRECTORY=example/data/out
rm -rf $OUTPUT_DIRECTORY
mkdir -p $OUTPUT_DIRECTORY

LUIGI="luigi --local-scheduler --module epimodel.tasks"
echo "The main way how to operate the pipeline is via the `luigi` tool. It loads"
echo "the tasks definition from epimodel.tasks where all inputs, dependencies and outputs"
echo "are defined:"
echo $LUIGI
$CONTINUE

echo "All outputs are stored in the $OUTPUT_DIRECTORY. This is defined in the `luigi.cfg`"
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
echo "Let's remove the output `$OUTPUT_DIRECTORY/john-hopkins.csv`"
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
$LUIGI JohnsHopkins --hopkins-output john-hopkins-2.csv
ls -la $OUTPUT_DIRECTORY/john-hopkins-2.csv

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

$CONTINUE
echo "$LUIGI ExportGleamBatch"
echo ""

$LUIGI ExportGleamBatch


echo "Now faking the results of ImportGleamBatch in $OUTPUT_DIRECTORY/output-import-gleam.hdf5"
echo "TODO: this needs more info from testing with gleam"
echo "Set FORETOLD_TOKEN for foretold to be able to execute this"
$CONTINUE
cp example/data/manual_input/output-import-gleam.hdf5 $OUTPUT_DIRECTORY/
$LUIGI WebExport \
  --UpdateForetold-foretold-channel $FORETOLD_TOKEN \
  --GleamvizResults-gleamviz-result output-import-gleam.hdf5 \
  --ImportGleamBatch-result-batch-file output-import-gleam.hdf5 \
  --export-name test-output

echo "And finally upload:"
$CONTINUE
$LUIGI WebUpload --export-name myout3 --gs-prefix gs://static-covid/static/v4/deleteme