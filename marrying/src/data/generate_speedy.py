import os
from typing import Callable

import numpy as np
import utils.spaces as spaces

# ShallowWaterModel only depend on layer_thickness
PARAM_SPACES = {
    "atmosphere": {
        "pres_ref": spaces.NBoxSpace(min_=9.2e4, max_=100e4),
        "temp_ref": spaces.NBoxSpace(min_=275, max_=300),
        "moist_lapse_rate": spaces.NBoxSpace(min_=3.5e-3, max_=9.8e-3),
        "layer_thickness": spaces.NBoxSpace(min_=8e3, max_=20e3),
    },
}

# compute parameter grid in discrete case
latent_sizes = [2] * 4
param_spaces = {f"{k}.{kk}": vv for k, v in PARAM_SPACES.items() for kk, vv in v.items()}
param_samples = {k: np.linspace(v.min_, v.max_, latent_sizes[i]) for i, (k, v) in enumerate(param_spaces.items())}
PARAM_GRID = np.stack(np.meshgrid(*[list(v) for v in param_samples.values()], indexing="ij"), axis=-1).reshape(
    -1, len(param_samples)
)


def discrete_uniform(output_path, simulation_days=30):
    for run_id, args in enumerate(PARAM_GRID):
        arguments = ""
        for i, k in enumerate(param_samples.keys()):
            arguments += f" --{k}={args[i]}"
        arguments += f" --output.path {output_path} --simulation_days {simulation_days} --output.id {run_id+1:04d}"

        os.chdir("~/ode_discovery/sw_jl")
        os.system(f"julia --project='.' src/speedy_weather_jl.jl {arguments}")


def simulate_weather(
    sim_fn: Callable = discrete_uniform,
    simulation_days=30,
    output_path="~/DATA/SpeedyWeather/ShallowWaterModel",
):
    """
    Simulate weather using the Speedy Weather model
    """
    os.makedirs(output_path, exist_ok=True)
    return sim_fn(output_path, simulation_days)
