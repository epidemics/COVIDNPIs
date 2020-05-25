import os
from subprocess import Popen, call

project_dir = os.path.join(os.path.dirname(__file__), "../..")
script_dir = os.path.join(project_dir, "scripts/r_estimate")


def estimate_r(output_file, john_hopkins_csv, serial_interval_sample):
    # install R dependencies
    rc = call([
        "/usr/bin/Rscript",
        os.path.join(script_dir, "dependencies.R"),
    ])
    if rc != 0:
        raise RuntimeError("Could not install R dependencies")

    process = Popen([
        "/usr/bin/Rscript",
        os.path.join(script_dir, "estimate_R.R"),
        serial_interval_sample,
        john_hopkins_csv,
        output_file
    ])
    _ = process.communicate()
    rc = process.returncode
    if rc != 0:
        raise RuntimeError("Could not estimate R")


if __name__ == "__main__":
    estimate_r(
        os.path.join(project_dir, "data-dir/outputs"),
        os.path.join(project_dir, "data-dir/outputs/john-hopkins-small.csv"),
        os.path.join(project_dir, "data-dir/inputs/manual/si_sample.rds"),
    )
