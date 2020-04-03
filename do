#!/usr/bin/env python3

import argparse
import logging
import subprocess
from pathlib import Path

import yaml

import epimodel
from epimodel import RegionDataset
from epimodel.exports.epidemics_org import WebExport, upload_export

log = logging.getLogger("gleambatch")


def update_CSSE(args):
    log.info("Downloading and parsing CSSE ...")
    csse = epimodel.imports.import_CSSE(args.rds)
    csse.to_csv(args.dest)
    log.info(
        f"Saved CSSE to {args.dest}, last day {csse.index.get_level_values(1).max()}"
    )


def web_export(args):
    ex = WebExport(comment=args.comment)
    for code in args.config["export_regions"]:
        ex.new_region(args.rds[code])
    # TODO: add data to ex
    ex.write(args.config["output_dir"])


def web_upload(args):
    c = args.config
    upload_export(
        args.EXPORTED_DIR, c["gs_prefix"], c["gs_url_prefix"], channel=args.channel
    )


def create_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-d", "--debug", action="store_true", help="Debugging logs.")
    ap.add_argument("-C", "--config", default="config.yaml", help="Config file.")
    sp = ap.add_subparsers(title="subcommands", required=True, dest="cmd")

    upC = sp.add_parser("update_CSSE", help="Fetch data from John Hopkins CSSE.")
    upC.set_defaults(func=update_CSSE)
    upC.add_argument(
        "-D", "--dest", default="`config.data_dir`/CSSE.csv", help="Destination path."
    )

    exp = sp.add_parser("web_export", help="Create data export for web.")
    exp.add_argument("-c", "--comment", help="A short comment (to be part of path).")
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

    return ap


def main():
    logging.basicConfig(level=logging.INFO)
    args = create_parser().parse_args()
    if args.debug:
        logging.root.setLevel(logging.DEBUG)
    with open(args.config, "rt") as f:
        args.config = yaml.load(f)
    args.rds = RegionDataset.load(Path(args.config["data_dir"]) / "regions.csv")
    args.func(args)


if __name__ == "__main__":
    main()
