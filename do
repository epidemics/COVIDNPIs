#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import yaml

import epimodel

from epimodel import Level, RegionDataset, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, batch

log = logging.getLogger("do")


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


def web_export(args):
    process_export(args)


def web_upload(args):
    c = args.config
    upload_export(
        args.EXPORTED_DIR, c["gs_prefix"], c["gs_url_prefix"], channel=args.channel
    )


def import_batch(args):
    batch = Batch.open(args.BATCH_FILE)
    d = args.rds.data
    regions = d.loc[(d.Level == Level.country) & (d.GleamID != "")].Region.values
    log.info(
        f"Importing results for {len(regions)} from GLEAM into {args.BATCH_FILE} ..."
    )
    batch.import_results_from_gleam(
        Path(args.config["gleamviz_sims_dir"]).expanduser(),
        regions,
        resample=args.config["gleam_resample"],
        allow_unfinished=args.allow_missing,
        # overwrite=True, ## Possible option
        info_level=logging.INFO,
    )


def generate_batch(args):
    b = Batch.new(dir=args.config["output_dir"], comment=args.comment)
    log.info(f"New batch file {b.path}")
    log.info(f"Reading base GLEAM definition {args.BASE_DEF} ...")
    d = epimodel.gleam.GleamDefinition(args.BASE_DEF)
    # TODO: This shuld be somewhat more versatile
    log.info(f"Reading estimates from CSV {args.COUNTRY_ESTIMATES} ...")
    est = read_csv_smart(args.COUNTRY_ESTIMATES, args.rds, levels=Level.country)
    if len(est.columns) > 1:
        raise Exception(f"Multiple columns found: {est.columns}")
    est = est[est.columns[0]]
    if args.start_date:
        start_date = utils.utc_date(args.start_date)
    else:
        start_date = d.get_start_date()
    log.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
    batch.generate_simulations(b, d, est, rds=args.rds, config=args.config, start_date=start_date, top=args.top)
    log.info(f"Generated batch {b.path!r}:\n  {b.stats()}")
    b.close()


def export_batch(args):
    batch = Batch.open(args.BATCH_FILE)
    gdir = args.config["gleamviz_sims_dir"]
    if args.out_dir is not None:
        gdir = args.out_dir
    log.info(
        f"Creating GLEAM XML definitions for batch {args.BATCH_FILE} in dir {gdir} ..."
    )
    batch.export_definitions_to_gleam(
        Path(gdir).expanduser(), overwrite=args.overwrite, info_level=logging.INFO
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

    gbp = sp.add_parser(
        "generate_gleam_batch", help="Create batch of definitions for GLEAM."
    )
    gbp.add_argument("-t", "--top", default=1500, type=int, help="Upper limit for seed compartments.")
    gbp.add_argument("-c", "--comment", help="A short comment (to be part of path).")
    gbp.add_argument(
        "-D",
        "--start_date",
        help="Set a sim start date (default: from the simulation def).",
    )
    gbp.add_argument("BASE_DEF", help="Basic definition file to use.")
    gbp.add_argument(
        "COUNTRY_ESTIMATES", help="The country-level estimate source CSV file."
    )
    gbp.set_defaults(func=generate_batch)

    ebp = sp.add_parser(
        "export_gleam_batch", help="Create batch of definitions for GLEAM."
    )
    ebp.add_argument("-o", "--out_dir", help="Override output dir (must exist).")
    ebp.add_argument(
        "-f", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    ebp.add_argument(
        "BATCH_FILE", help="The batch-*.hdf5 file with batch spec to be updated."
    )
    ebp.set_defaults(func=export_batch)

    exp = sp.add_parser("web_export", help="Create data export for web.")
    exp.add_argument("-c", "--comment", help="A short comment (to be part of path).")
    exp.add_argument("BATCH_FILE", help="A result HDF file of import_gleam_batch step")
    exp.add_argument("estimates", help="CSV file containing the current estimates")
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
    data_dir = Path(args.config["data_dir"])
    args.rds = RegionDataset.load(
        data_dir / "regions.csv", data_dir / "regions-gleam.csv"
    )
    epimodel.algorithms.estimate_missing_populations(args.rds)
    args.func(args)


if __name__ == "__main__":
    main()
