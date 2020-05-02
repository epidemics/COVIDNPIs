#!/usr/bin/env python3

import click
import logging
from pathlib import Path

import yaml

import epimodel

from epimodel import Level, RegionDataset, read_csv_smart, utils
from epimodel.exports.epidemics_org import process_export, upload_export
from epimodel.gleam import Batch, batch

log = logging.getLogger("do")

# Global arguments


@click.group()
@click.option("-d", "--debug", is_flag=True, help="Debugging logs.")
@click.option(
    "-C",
    "--config",
    type=click.Path(exists=True),
    default="config.yaml",
    help="Config file.",
)
@click.pass_context
def cli(ctx, debug, config):
    """
    Epimodel pipeline runner

    See https://github.com/epidemics/epimodel for more details
    """
    # TODO: add environment variable for config
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug

    logging.basicConfig(level=logging.INFO)
    if debug:
        logging.root.setLevel(logging.DEBUG)

    with open(config, "rt") as f:
        ctx.obj["CONFIG"] = yaml.safe_load(f)

    data_dir = Path(ctx.obj["CONFIG"]["data_dir"])
    ctx.obj["RDS"] = RegionDataset.load(
        data_dir / "regions.csv", data_dir / "regions-gleam.csv"
    )
    epimodel.algorithms.estimate_missing_populations(ctx.obj["RDS"])


# Commands


@cli.command()
@click.pass_context
def update_johns_hopkins(ctx):
    """Fetch data from Johns Hopkins CSSE."""
    log.info("Downloading and parsing CSSE ...")
    csse = epimodel.imports.import_johns_hopkins(ctx.obj["RDS"])
    dest = Path(ctx.obj["CONFIG"]["data_dir"]) / "johns-hopkins.csv"
    csse.to_csv(dest)
    log.info(
        f"Saved CSSE to {dest}, last day is {csse.index.get_level_values(1).max()}"
    )


@cli.command()
@click.pass_context
def update_foretold(ctx):
    """Fetch data from Foretold."""
    if ctx.obj["CONFIG"]["foretold_channel"] == "SECRET":
        log.warning(
            "`foretold_channel` in the config file is not set to non-default value."
        )
    else:
        log.info("Downloading and parsing foretold")
        foretold = epimodel.imports.import_foretold(
            ctx.obj["RDS"], ctx.obj["CONFIG"]["foretold_channel"]
        )
        dest = Path(ctx.obj["CONFIG"]["data_dir"]) / "foretold.csv"
        foretold.to_csv(dest, float_format="%.7g")
        log.info(f"Saved Foretold to {dest}")


@cli.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.option("-M", "--allow-missing", is_flag=True, help="Skip missing sim results.")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing `new_fraction` imported table.",
)
@click.pass_context
def import_gleam_batch(ctx, batch_file, allow_missing, overwrite):
    """
    Load batch results from GLEAM.

    BATCH_FILE: The batch-*.hdf5 file with batch spec to be updated.
    """
    b = Batch.open(batch_file)
    d = ctx.obj["RDS"].data
    regions = set(
        d.loc[
            ((d.Level == Level.country) | (d.Level == Level.continent))
            & (d.GleamID != "")
        ].Region.values
    )
    # Add all configured regions
    for rc in ctx.obj["CONFIG"]["export_regions"]:
        r = ctx.obj["RDS"][rc]
        if r.GleamID != "":
            regions.add(r)

    log.info(f"Importing results for {len(regions)} from GLEAM into {batch_file} ...")
    b.import_results_from_gleam(
        Path(ctx.obj["CONFIG"]["gleamviz_sims_dir"]).expanduser(),
        regions,
        resample=ctx.obj["CONFIG"]["gleam_resample"],
        allow_unfinished=allow_missing,
        overwrite=overwrite,
        info_level=logging.INFO,
    )


@cli.command()
@click.argument("base_def", type=click.Path(exists=True))
@click.argument("country_estimates", type=click.Path(exists=True))
@click.option(
    "-t", "--top", default=2000, type=int, help="Upper limit for seed compartments."
)
@click.option("-c", "--comment", type=str, help="A short comment (to be part of path).")
@click.option(
    "-D",
    "--start-date",
    type=click.DateTime(),
    help="Set a sim start date (default: from the simulation def).",
)
@click.pass_context
def generate_gleam_batch(ctx, base_def, country_estimates, top, comment, start_date):
    """
    Create batch of definitions for GLEAM.
    
    BASE_DEF: Basic definition file to use.

    COUNTRY_ESTIMATES: The country-level estimate source CSV file.
    """
    b = Batch.new(dir=ctx.obj["CONFIG"]["output_dir"], comment=comment)
    log.info(f"New batch file {b.path}")
    log.info(f"Reading base GLEAM definition {base_def} ...")
    d = epimodel.gleam.GleamDefinition(base_def)
    # TODO: This should be somewhat more versatile
    log.info(f"Reading estimates from CSV {country_estimates} ...")
    est = read_csv_smart(country_estimates, ctx.obj["RDS"], levels=Level.country)
    if start_date:
        start_date = utils.utc_date(start_date)
    else:
        start_date = d.get_start_date()
    log.info(f"Generating scenarios with start_date {start_date.ctime()} ...")
    batch.generate_simulations(
        b,
        d,
        est,
        rds=ctx.obj["RDS"],
        config=ctx.obj["CONFIG"],
        start_date=start_date,
        top=top,
    )
    log.info(f"Generated batch {b.path!r}:\n  {b.stats()}")
    b.close()


@cli.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(exists=True),
    help="Override output dir (must exist).",
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files.")
@click.pass_context
def export_gleam_batch(ctx, batch_file, out_dir, overwrite):
    """Create batch of definitions for GLEAM.

    BATCH_FILE: The batch-*.hdf5 file with batch spec to be updated.
    """
    batch = Batch.open(batch_file)
    gdir = ctx.obj["CONFIG"]["gleamviz_sims_dir"]
    if out_dir is not None:
        gdir = out_dir
    log.info(f"Creating GLEAM XML definitions for batch {batch_file} in dir {gdir} ...")
    batch.export_definitions_to_gleam(
        Path(gdir).expanduser(), overwrite=overwrite, info_level=logging.INFO
    )


@cli.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.argument("estimates", type=click.Path(exists=True))
@click.option("-c", "--comment", type=str, help="A short comment (to be part of path).")
@click.option(
    "-p", "--pretty-print", is_flag=True, help="Pretty-print exported JSON files."
)
@click.pass_context
def web_export(ctx, batch_file, estimates, comment, pretty_print):
    """
    Create data export for web.

    BATCH_FILE: A result HDF file of import-gleam-batch step

    ESTIMATES: CSV file containing the current estimates
    """
    process_export(
        ctx.obj["CONFIG"],
        ctx.obj["RDS"],
        ctx.obj["DEBUG"],
        comment,
        batch_file,
        estimates,
        pretty_print,
    )


@cli.command()
@click.option(
    "-d",
    "--dir",
    "dir_",
    type=click.Path(exists=True),
    help="The generated export directory to upload from.",
)
@click.option(
    "-c",
    "--channel",
    type=str,
    default="staging",
    help="Channel to upload to ('main' for main site). Default is 'staging'.",
)
@click.pass_context
def web_upload(ctx, dir_, channel):
    """
    Upload data to the configured GCS bucket.

    By default, uploads from the output_latest directory specified in config.yaml (out/latest).
    """
    c = ctx.obj["CONFIG"]

    if dir_ == None:
        dir_ = Path(c["output_dir"]) / c["output_latest"]

    upload_export(dir_, c, channel=channel)


@cli.command()
@click.argument("SRC", type=click.Path(exists=True))
@click.argument("DEST", type=click.Path(exists=True))
@click.pass_context
def import_countermeasures(ctx, src, dest):
    """
    Import one CSV file from countermeasures DB.

    SRC: Input CSV.

    DEST: Output CSV.
    """
    log.info(f"Importing countermeasures from {src} into {dest} ...")
    cms = epimodel.imports.import_countermeasures_csv(ctx.obj["RDS"], src)
    cms.to_csv(dest)
    log.info(
        f"Saved countermeasures to {dest}, {len(cms.columns)} features, "
        f"last day is {cms.index.get_level_values(1).max()}"
    )


# Workflows


@cli.group()
@click.pass_context
def workflow(ctx):
    """
    Workflows to run stages of the pipeline.
    """


@workflow.command()
# import-gleam-batch
@click.argument("batch_file", type=click.Path(exists=True))
@click.option("-M", "--allow-missing", is_flag=True, help="Skip missing sim results.")
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing `new_fraction` imported table.",
)
# web-export
@click.argument("estimates", type=click.Path(exists=True))
@click.option("-c", "--comment", type=str, help="A short comment (to be part of path).")
@click.option(
    "-p", "--pretty-print", is_flag=True, help="Pretty-print exported JSON files."
)
# web-upload
@click.option(
    "-C",
    "--channel",
    type=str,
    default="staging",
    help="Channel to upload to ('main' for main site). Default is 'staging'.",
)
@click.pass_context
def gleam_to_web(
    ctx, batch_file, allow_missing, overwrite, estimates, comment, pretty_print, channel
):
    """
    Runs import-gleam-batch, web-export and web-upload.

    BATCH_FILE: The batch-*.hdf5 file with batch spec to be updated.

    ESTIMATES: CSV file containing the current estimates
    """
    # ctx.invoke(import_gleam_batch, batch_file=batch_file, allow_missing=allow_missing, overwrite=overwrite)
    ctx.invoke(
        web_export,
        batch_file=batch_file,
        estimates=estimates,
        comment=comment,
        pretty_print=pretty_print,
    )
    ctx.invoke(web_upload, channel=channel)


if __name__ == "__main__":
    cli(obj={})
