from click.testing import CliRunner
import importlib.machinery, importlib.util
from pathlib import Path
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
import os
import yaml
import filecmp
import json
from epimodel.exports.epidemics_org import types_to_json
from distutils.dir_util import copy_tree

loader = importlib.machinery.SourceFileLoader('do', 'do')
spec = importlib.util.spec_from_loader(loader.name, loader)
do = importlib.util.module_from_spec(spec)
loader.exec_module(do)

'''
Not tested:
web-upload
workflow-gleam-to-web
workflow-prepare-gleam
import-countermeasures
update-foretold (requires API keys)
'''

def prepareFS(datadir):
    copy_tree(str(datadir / "cli-dir"), str(Path.cwd()))

def test_update_johns_hopkins(datadir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        prepareFS(datadir)
        result = runner.invoke(do.cli, ['update-johns-hopkins'])
        assert result.exit_code == 0
        df_val = pd.read_csv(str(datadir / "files/johns-hopkins-until-may.csv"))
        df = pd.read_csv("data/johns-hopkins.csv")
        df_before_may = df.loc[df.Date < "2020-05-01"].reset_index(drop=True)
        pd.testing.assert_frame_equal(df_val, df_before_may)

def test_generate_gleam_batch(datadir):
    # assumption: order of timestamp-based simulation IDs remains the same
    runner = CliRunner()
    with runner.isolated_filesystem():
        prepareFS(datadir)
        result = runner.invoke(do.cli, ['generate-gleam-batch', str(datadir / "files/test_definition.xml"), "data/sources/estimates-JK-2020-04-15-no-CD.csv"])
        assert result.exit_code == 0

        hdf_val = pd.HDFStore(datadir / "files/test-generate-gleam-batch.hdf5")
        outdir = Path.cwd() / "out"
        hdf_gen = pd.HDFStore(outdir / os.listdir(outdir)[0])

        pd.testing.assert_frame_equal(hdf_gen.get("initial_compartments"),hdf_val.get("initial_compartments"), check_exact=False, check_less_precise=True)
        hdf_gen_sim = hdf_gen.get("simulations").reset_index().drop(columns=["SimulationID"])
        hdf_val_sim = hdf_val.get("simulations").reset_index().drop(columns=["SimulationID"])

        # check all other columns
        pd.testing.assert_frame_equal(hdf_gen_sim.drop(columns=["DefinitionXML"]),hdf_val_sim.drop(columns=["DefinitionXML"]), check_exact=False, check_less_precise=True)

        # check simulation XML
        assert(len(hdf_gen_sim)==len(hdf_val_sim))
        gen_xml = hdf_gen_sim["DefinitionXML"]
        val_xml = hdf_val_sim["DefinitionXML"]

        for i in range(len(gen_xml)):
            root_gen = ET.fromstring(gen_xml[i])
            root_gen.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('id', 'null')
            root_gen.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('name', 'null')

            root_val = ET.fromstring(val_xml[i])
            root_val.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('id', 'null')
            root_val.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('name', 'null')

            assert ET.tostring(root_gen)==ET.tostring(root_val)

def test_export_gleam_batch(datadir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        prepareFS(datadir)
        result = runner.invoke(do.cli, ['export-gleam-batch', str(datadir / "files/test-generate-gleam-batch.hdf5")])
        assert result.exit_code == 0

        sims_gen = Path.cwd() / "sims"
        sims_val = datadir / "files/sims"

        sims_gen_dirs = sorted(filter(lambda f: not f.startswith('.'), os.listdir(sims_gen)))
        sims_val_dirs = sorted(filter(lambda f: not f.startswith('.'), os.listdir(sims_val)))
        assert len(sims_gen_dirs) == len(sims_val_dirs)

        for i in range(len(sims_gen_dirs)):
            root_gen = ET.parse(str(sims_gen / sims_gen_dirs[i] / "definition.xml")).getroot()
            root_gen.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('id', 'null')
            root_gen.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('name', 'null')

            root_val = ET.parse(str(sims_val / sims_val_dirs[i] / "definition.xml")).getroot()
            root_val.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('id', 'null')
            root_val.find('{http://www.gleamviz.org/xmlns/gleamviz_v4_0}definition').set('name', 'null')

            assert ET.tostring(root_gen)==ET.tostring(root_val)

# import-gleam-batch

# web-export
def test_web_export(datadir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        prepareFS(datadir)
        result = runner.invoke(do.cli, ['update-johns-hopkins'])
        assert result.exit_code == 0
        result = runner.invoke(do.cli, ['web-export', str(datadir / "files/example-output-batch.hdf5"), "data/sources/estimates-JK-2020-04-15-no-CD.csv"])
        assert result.exit_code == 0

        json_gen = Path.cwd() / "out/latest"
        json_val = datadir / "files/export"

        with open(datadir / "cli-dir/config.yaml", "rt") as f:
            config = yaml.safe_load(f)

        json_gen_comp = None
        json_val_comp = None

        with open(json_gen / config["gs_datafile_name"], "r") as blob:
            data_gen = json.load(
                blob,
                parse_constant=(lambda x: raise_("Not valid JSON: detected `" + x + "'")),
            )
            data_gen["created"]="test"
            data_gen["created_by"]="test"
            data_gen["comment"]="test"
            json_gen_comp = json.dumps(
                data_gen, default=types_to_json, allow_nan=False, separators=(",", ":"),
            )
        with open(json_val / config["gs_datafile_name"], "r") as blob:
            data_val = json.load(
                blob,
                parse_constant=(lambda x: raise_("Not valid JSON: detected `" + x + "'")),
            )
            data_val["created"]="test"
            data_val["created_by"]="test"
            data_val["comment"]="test"
            json_val_comp = json.dumps(
                data_val, default=types_to_json, allow_nan=False, separators=(",", ":"),
            )
        assert json_gen_comp is not None and json_gen_comp == json_val_comp

        #gen_files = sorted(filter(lambda f: not f.startswith('.'), os.listdir(json_gen)))
        #val_files = sorted(filter(lambda f: not f.startswith('.'), os.listdir(json_val)))
        #assert gen_files == val_files
        #for i in gen_files:
        #    if i == config["gs_datafile_name"]:
        #        continue
        #    assert filecmp.cmp(json_gen / i, json_val / i)
        assert filecmp.cmp(json_gen / "extdata-GB.json", json_val / "extdata-GB.json")