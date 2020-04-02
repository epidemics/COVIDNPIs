#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import epimodel

log = logging.getLogger("gleambatch")


def updateCSSE(args):
    rds = epimodel.RegionDataset.load(args.regions)
    log.info("Downloading and parsing CSSE ...")
    csse = epimodel.imports.import_CSSE(rds)
    csse.to_csv(args.dest)
    log.info(f"Saved CSSE to {args.dest}, last day {csse.index.get_level_values(1).max()}")


def create_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "-d", "--debug", action="store_true", help="Display debugging mesages."
    )
    ap.add_argument(
        "-R", "--regions", default="data/regions.csv", help="Regions file path."
    )
    sp = ap.add_subparsers(title="subcommands", required=True, dest="cmd")

    upC = sp.add_parser("updateCSSE", help="Fetch data from John Hopkins CSSE.")
    upC.set_defaults(func=updateCSSE)
    upC.add_argument("-D", "--dest", default="data/CSSE.csv", help="Destination path.")

    return ap


def main():
    logging.basicConfig(level=logging.INFO)
    args = create_parser().parse_args()
    if args.debug:
        logging.root.setLevel(logging.DEBUG)
    args.func(args)


if __name__ == "__main__":
    main()
