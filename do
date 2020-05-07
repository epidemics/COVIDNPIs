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

@click.group(chain=True)
@click.option("-d", "--debug", is_flag=True, help="Enable debugging logs.")
@click.option(
    "-C",
    "--config",
    type=click.Path(exists=True),
    default="config.yaml",
    envvar="EPI_CONFIG",
    help="Path to config file (default config.yaml; alternatively set EPI_CONFIG "\
         "environment variable).",
)
@click.pass_context
def cli(ctx, debug, config):
    """
    Epimodel pipeline runner.

    See https://github.com/epidemics/epimodel for more details.

    1. Update Johns Hopkins data:
    
    ./do update-johns-hopkins (not needed if you got fresh data from the repo)

    2. Generate batch file from estimates and basic Gleam XML definition.
    
    ./do generate-gleam-batch -D 2020-04-15 -c JK default.xml
    estimates-2020-04-15.csv

    The batch file now contains all the scenario definitions and initial
    populations. Note the estimate input specification may change.

    3. Export Gleam simulation XML files in Gleamviz (not while gleamviz is
       running!).

    ./do export-gleam-batch out/batch-2020-04-16T03:54:52.910001+00:00.hdf5

    4. Start gleamviz. You should see the new simulations loaded. Run all of
       them and "Retrieve results" (do not export manually). Exit gleamviz.

    5. Import the gleamviz results into the HDF batch file (Gleamviz must be
       stopped before that). After this succeeds, you may delete the
       simulations from gleamviz.

    ./do import-gleam-batch out/batch-2020-04-16T03:54:52.910001+00:00.hdf5

    6. Generate web export (additional data are fetched from config.yml)

    ./do web-export out/batch-2020-04-16T03:54:52.910001+00:00.hdf5
    data/sources/estimates-JK-2020-04-15.csv

    7. Export the generated folder to web! Optionally, set a channel for
       testing first.

    ./do web-upload out/export-2020-04-03T02:03:28.991629+00:00 -c ttest28

    Workflow macros:

    1. Update Johns Hopkins and Foretold data, generate batch file from
       estimates and basic Gleam XML definition and export Gleam simulation
       XML files to Gleamviz (not while gleamviz is running!):
   
    ./do workflow-prepare-gleam -D 2020-04-15 -c JK default.xml
    estimates-2020-04-15.csv

    2. Start gleamviz. You should see the new simulations loaded. Run all of
       them and "Retrieve results" (do not export manually). Exit gleamviz.

    3. Import the gleamviz results into the HDF batch file, generate web
       export and export the generated folder to web (Gleamviz must be stopped
       before that.) After this succeeds, you may delete the simulations from
       gleamviz.
    
    ./do workflow-gleam-to-web -C ttest28
    out/batch-2020-04-16T03:54:52.910001+00:00.hdf5
    data/sources/estimates-JK-2020-04-15.csv
    """
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug

    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    with open(config, "rt") as f:
        ctx.obj["CONFIG"] = yaml.safe_load(f)

    data_dir = Path(ctx.obj["CONFIG"]["data_dir"])
    ctx.obj["RDS"] = RegionDataset.load(
        data_dir / "regions.csv", data_dir / "regions-gleam.csv"
    )
    epimodel.algorithms.estimate_missing_populations(ctx.obj["RDS"])


# Actions

@cli.command()
@click.pass_context
def update_johns_hopkins(ctx):
    """
    Fetch data from Johns Hopkins CSSE.

    Data stored in the directory specified by 'data_dir' in config.yml.
    """
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
    """
    Fetch data from Foretold.

    Data stored in the directory specified by 'data_dir' in config.yml.
    Channel specified by 'foretold_channel' in config.yml.
    """
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

    Simulation directory specified by 'gleamviz_sims_dir' in config.yml.

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

    Saved in the directory specified by 'output_dir' in config.yml.
    
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
    start_date = utils.utc_date(start_date) if start_date else d.get_start_date()
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
    if "invoked_by_subcommand" in ctx.parent.__dict__:
        ctx.parent.batch_file = b.path


@cli.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(exists=True),
    help="Override output directory (must exist).",
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files.")
@click.pass_context
def export_gleam_batch(ctx, batch_file, out_dir, overwrite):
    """
    Export batch of definitions for GLEAM.

    By default exports to 'gleamviz_sims_dir' as specified in config.yml.

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

    Saved in the directory specified by 'output_dir' in config.yml.

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
@click.argument("channel", type=str)
@click.option(
    "-d",
    "--dir",
    "dir_",
    type=click.Path(exists=True),
    help="The generated export directory to upload from.",
)
@click.pass_context
def web_upload(ctx, dir_, channel):
    """
    Upload data to the configured GCS bucket.

    By default, uploads from the output_latest directory specified in
    config.yaml (out/latest).
    
    CHANNEL: Channel to upload to (main, staging, testing or custom channels).
    """
    c = ctx.obj["CONFIG"]

    if dir_ == None:
        dir_ = Path(c["output_dir"]) / c["output_latest"]

    upload_export(dir_, c, channel=channel)


@cli.command()
@click.argument("SRC", type=click.Path(exists=True))
@click.argument("DEST", type=click.Path())
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

@cli.command()
# update-johns-hopkins
# generate-gleam-batch
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
# export-gleam-batch
@click.option(
    "-o",
    "--out_dir",
    type=click.Path(exists=True),
    help="Override output directory (must exist).",
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files.")
@click.pass_context
def workflow_prepare_gleam(
    ctx, base_def, country_estimates, top, comment, start_date, out_dir, overwrite
):
    """
    Creates and exports a batch of definitions for GLEAM. 

    Runs update-johns-hopkins, generate-gleam-batch and
    export-gleam-batch.

    By default exports to 'gleamviz_sims_dir' as specified in config.yml.

    BASE_DEF: Basic definition file to use.
    
    COUNTRY_ESTIMATES: The country-level estimate source CSV file.
    """
    ctx.invoked_by_subcommand = True
    ctx.invoke(update_johns_hopkins)
    ctx.invoke(
        generate_gleam_batch,
        base_def=base_def,
        country_estimates=country_estimates,
        top=top,
        comment=comment,
        start_date=start_date,
    )
    ctx.invoke(
        export_gleam_batch,
        batch_file=ctx.batch_file,
        out_dir=out_dir,
        overwrite=overwrite,
    )


@cli.command()
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
@click.argument("channel", type=str)
@click.pass_context
def workflow_gleam_to_web(
    ctx, batch_file, allow_missing, overwrite, estimates, comment, pretty_print, channel
):
    """
    Imports GLEAM sim results and uploads them to the web.

    Runs import-gleam-batch, web-export and web-upload.

    BATCH_FILE: The batch-*.hdf5 file with batch spec to be updated.

    ESTIMATES: CSV file containing the current estimates

    CHANNEL: Channel to upload to (main, staging, testing or custom channels).
    """
    ctx.invoke(
        import_gleam_batch,
        batch_file=batch_file,
        allow_missing=allow_missing,
        overwrite=overwrite,
    )
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
