#!/usr/bin/env python3

import argparse
import pyrokinetics as pk
import xarray as xr
import subprocess
import os
import pathlib
import numpy as np
from inference.gp import (
    GpOptimiser,
    UpperConfidenceBound,
    SquaredExponential,
    WhiteNoise,
)
import matplotlib.pyplot as plt
import pickle
import tqdm


BASE_DIR = "salpha_optimise_5_errs"


def run_gs2(base_dir, filename, gs2_bin="/home/peter/Codes/gyrokinetics/gs2/bin/gs2"):
    old_cwd = os.getcwd()
    os.chdir(base_dir)
    subprocess.run(f"mpirun -np 4 {gs2_bin} {filename}", shell=True, check=True)
    os.chdir(old_cwd)


def read_growthrate(pyro):
    filename = (pathlib.Path(pyro.run_directory) / pyro.file_name).with_suffix(
        ".out.nc"
    )
    with xr.open_dataset(filename) as df:
        return df.omega_average.isel(ri=1, t=-1).squeeze().data


def read_growthrate_gpo(base_dir, params, scan):
    filename = (
        pathlib.Path(base_dir)
        / scan._single_run_name({"beta_prime": params[0], "shat": params[1]})
        / "input.out.nc"
    )
    with xr.open_dataset(filename) as df:
        return df.omega_average.isel(ri=1, t=-1).squeeze().data


def gs2_growth_rate(beta, shat, base, gs2_bin):
    print(f"\n** Running beta={beta}, shat={shat}\n")

    params = pk.PyroScan(
        base,
        {"beta_prime": [beta], "shat": [shat]},
        value_separator="_",
        parameter_separator="_",
        file_name="input.in",
        base_directory=BASE_DIR,
    )
    params.add_parameter_key("shat", "local_geometry", ["shat"])
    params.add_parameter_key("beta_prime", "local_geometry", ["beta_prime"])
    params.write()

    run = list(params.pyro_dict.values())[0]
    run_gs2(run.run_directory, run.file_name, gs2_bin)
    gamma = read_growthrate(run)
    return -np.abs(gamma)


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

    pyro = pk.Pyro(gk_file="fields_local_single_mode.in", gk_code="GS2")
    x = [np.array((0.0, 0.01)), np.array((-1, 5))]
    y = [gs2_growth_rate(x[0], x[1], pyro, args.gs2_bin) for x_ in x]
    bounds = [(-1, 0), (0.01, 5)]

    scan = pk.PyroScan(
        pyro,
        {"beta_prime": [x[0][0]], "shat": [x[0][0]]},
        value_separator="_",
        parameter_separator="_",
        file_name="input.in",
        base_directory=BASE_DIR,
    )

    GPO = GpOptimiser(x, y, bounds=bounds, acquisition=UpperConfidenceBound)

    N = 5

    for _ in tqdm.tqdm(range(N**2 - 2)):
        new_x = GPO.propose_evaluation()
        new_y = gs2_growth_rate(new_x[0], new_x[1], pyro, args.gs2_bin)
        GPO.add_evaluation(new_x, new_y)

    opt_result = np.array([read_growthrate_gpo(BASE_DIR, x_, scan) for x_ in GPO.x])

    with open(f"{BASE_DIR}/opt_results.pickle", "wb") as f:
        pickle.dump(GPO, f)
        pickle.dump(opt_result, f)

    eval_beta = np.array(GPO.x)[:, 0]
    eval_shat = np.array(GPO.x)[:, 1]

    max_growth_rate = np.max(np.abs(opt_result))

    M = 50
    beta_gp = np.linspace(*bounds[0], M)
    shat_gp = np.linspace(*bounds[1], M)
    beta_ij, shat_ij = np.meshgrid(beta_gp, shat_gp, indexing="ij")
    mean, std = GPO(np.column_stack([beta_ij.flat, shat_ij.flat]))

    fig, ax = plt.subplots()
    cntr = ax.pcolormesh(
        shat_ij,
        beta_ij,
        -mean.reshape((M, M)),
        cmap="RdBu_r",
        vmin=-max_growth_rate,
        vmax=max_growth_rate,
    )
    ax.set_xlabel("shat")
    ax.set_ylabel("beta_prime")
    cbar = fig.colorbar(cntr)
    cbar.set_label("gamma")
    ax.scatter(eval_shat, eval_beta, marker="x", color="k")
    fig.savefig(f"{BASE_DIR}/s_alpha_opt.png")
