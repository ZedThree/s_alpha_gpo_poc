#!/usr/bin/env python3

import argparse
import pyrokinetics as pk
import xarray as xr
import subprocess
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def run_gs2(base_dir, filename, gs2_bin):
    old_cwd = os.getcwd()
    os.chdir(base_dir)
    subprocess.run(f"{gs2_bin} {filename}", shell=True, check=True)
    os.chdir(old_cwd)


def read_growthrate(pyro):
    filename = (pathlib.Path(pyro.run_directory) / pyro.file_name).with_suffix(
        ".out.nc"
    )
    with xr.open_dataset(filename) as df:
        return df.omega_average.isel(ri=1, t=-1).squeeze().data


def reshape_results(scan):
    shape = list(len(value) for value in scan.parameter_dict.values())
    return np.array(
        [
            scan.pyro_dict[scan._single_run_name(run)].result
            for run in scan.outer_product()
        ]
    ).reshape(shape)


def plot_gamma(df):
    pk_B, pk_S = np.meshgrid(df.beta_prime, df.shat, indexing="ij")
    df.gamma.plot()
    ax = plt.gca()
    ax.scatter(pk_S, pk_B, marker="x", color="k")
    fig = plt.gcf()
    fig.savefig("s_alpha_pk.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "scan_with_pk.py", description="Uniform scan with pyrokinetics"
    )
    parser.add_argument(
        "--gs2-bin",
        default="/home/peter/Codes/gyrokinetics/gs2/bin/gs2",
        help="Path to GS2 executable",
    )
    args = parser.parse_args()

    N = 5

    pyro = pk.Pyro(gk_file="fields_local_single_mode.in", gk_code="GS2")
    scan = pk.PyroScan(
        pyro,
        {"beta_prime": np.linspace(0, -1, N), "shat": np.linspace(0.01, 5, N)},
        value_separator="_",
        parameter_separator="_",
        file_name="input.in",
        base_directory="salpha_scan_5",
    )
    scan.add_parameter_key("shat", "local_geometry", ["shat"])
    scan.add_parameter_key("beta_prime", "local_geometry", ["beta_prime"])
    scan.write()

    for p in scan.pyro_dict.values():
        run_gs2(p.run_directory, p.file_name, args.gs2_bin)
        p.result = read_growthrate(p)

    df = xr.Dataset(
        {
            "gamma": (list(scan.parameter_dict.keys()), reshape_results(scan)),
            **scan.parameter_dict,
        }
    )

    plot_gamma(df)
