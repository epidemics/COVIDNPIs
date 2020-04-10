#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import List
import pandas as pd

import yaml

import epimodel

from epimodel import Level, RegionDataset
from epimodel.exports.epidemics_org import WebExport, upload_export
from epimodel.gleam import Batch

log = logging.getLogger("gleambatch")


def import_countermeasures(args):
    log.info(f"Importing countermeasures from {args.SRC} into {args.DEST} ...")
    cms = epimodel.imports.import_countermeasures_csv(args.rds, args.SRC)
    cms.to_csv(args.DEST)
    log.info(
        f"Saved countermeasures to {args.DEST}, {len(cms.columns)} features, "
        f"last day is {cms.index.get_level_values(1).max()}"
    )


def update_johns_hopkins(args):
    log.info("Downloading and parsing CSSE ...")
    csse = epimodel.imports.import_johns_hopkins(args.rds)
    dest = Path(args.config["data_dir"]) / "johns-hopkins.csv"
    csse.to_csv(dest)
    log.info(
        f"Saved CSSE to {dest}, last day is {csse.index.get_level_values(1).max()}"
    )


def update_foretold(args):
    if args.config["foretold_channel"] == "SECRET":
        log.warning(
            "`foretold_channel` in the config file is not set to non-default value."
        )
    else:
        log.info("Downloading and parsing foretold")
        foretold = epimodel.imports.import_foretold(
            args.rds, args.config["foretold_channel"]
        )
        dest = Path(args.config["data_dir"]) / "foretold.csv"
        foretold.to_csv(dest, float_format="%.7g")
        log.info(f"Saved Foretold to {dest}")


def _web_export(
    comment: str,
    export_regions: List[str],
    region_dataset: RegionDataset,
    rates_path: Path,
    output_dir: Path,
) -> None:
    ex = WebExport(comment=comment)
    for code in export_regions:
        ex.new_region(region_dataset[code])

    rates: pd.DataFrame = pd.read_csv(rates_path)
    ex.add_data_from(rates, columns=['beds', 'crits'])
    ex.write(output_dir)


def web_export(args):
    _web_export(
        args.comment,
        args.config["export_regions"],
        args.rds,
        args.rates,
        args.config["output_dir"],
    )


def web_upload(args):
    c = args.config
    upload_export(
        args.EXPORTED_DIR, c["gs_prefix"], c["gs_url_prefix"], channel=args.channel
    )


def import_batch(args):
    batch = Batch.open(args.BATCH_FILE)
    d = args.rds.data
    regions = d.loc[(d.Level == Level.country) & (d.GleamID != "")].Region.values
    batch.import_sims(
        Path(args.config["gleamviz_sims_dir"]).expanduser(),
        regions,
        resample=args.config["gleam_resample"],
        allow_unfinished=args.allow_missing,
    )


def create_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-d", "--debug", action="store_true", help="Debugging logs.")
    ap.add_argument("-C", "--config", default="config.yaml", help="Config file.")
    sp = ap.add_subparsers(title="subcommands", required=True, dest="cmd")

    upp = sp.add_parser(
        "update_johns_hopkins", help="Fetch data from Johns Hopkins CSSE."
    )
    upp.set_defaults(func=update_johns_hopkins)

    upf = sp.add_parser("update_foretold", help="Fetch data from Foretold.")
    upf.set_defaults(func=update_foretold)

    ibp = sp.add_parser("import_gleam_batch", help="Load batch results from GLEAM.")
    ibp.add_argument(
        "BATCH_FILE", help="The batch-*.hdf5 file with batch spec to be updated."
    )
    ibp.add_argument(
        "-M", "--allow-missing", action="store_true", help="Skip missing sim results.",
    )
    ibp.set_defaults(func=import_batch)

    # TODO: candidate to be split into two steps: "integrate data" -> export
    exp = sp.add_parser("web_export", help="Create data export for web.")
    exp.add_argument("-c", "--comment", help="A short comment (to be part of path).")
    exp.add_argument(
        "-b", "--batch-file", help="A result HDF file of import_gleam_batch step"
    )
    exp.add_argument(
        "-r", "--rates", help="Path to a file for hospital rates (M49 indexed)"
    )
    # TODO: additional data files we want to have in the export
    exp.set_defaults(func=web_export)

    uplp = sp.add_parser("web_upload", help="Upload data to the configured GCS bucket")
    uplp.add_argument("EXPORTED_DIR", help="The generated export directory.")
    uplp.add_argument(
        "-c",
        "--channel",
        default="staging",
        help="Channel to upload to ('main' for main site).",
    )
    uplp.set_defaults(func=web_upload)

    iftp = sp.add_parser(
        "import_countermeasures", help="Import one CSV file from countermeasures DB."
    )
    iftp.add_argument("SRC", help="Input CSV.")
    iftp.add_argument("DEST", help="Output CSV.")
    iftp.set_defaults(func=import_countermeasures)

    return ap


def main():
    logging.basicConfig(level=logging.INFO)
    args = create_parser().parse_args()
    if args.debug:
        logging.root.setLevel(logging.DEBUG)
    with open(args.config, "rt") as f:
        args.config = yaml.safe_load(f)
    args.rds = RegionDataset.load(Path(args.config["data_dir"]) / "export_regions.csv")
    args.func(args)


if __name__ == "__main__":
    main()
