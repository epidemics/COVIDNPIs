import os
from subprocess import Popen

project_dir = os.path.join(os.path.dirname(__file__), "../..")
script_dir = os.path.join(project_dir, "scripts/r_estimate")


def estimate_r(r_script_executable, output_file, john_hopkins_csv, serial_interval_sample):
    process = Popen([
        r_script_executable,
        os.path.join(script_dir, "estimate_R.R"),
        serial_interval_sample,
        john_hopkins_csv,
        output_file
    ])
    _ = process.communicate()
    rc = process.returncode
    if rc != 0:
        raise RuntimeError("Could not estimate R")
