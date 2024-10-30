import json
import random
from math import cos, cosh, exp, log, sin, sinh, tan, tanh  # import required math functions
from os.path import abspath, dirname

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

project_root = dirname(dirname(abspath(__file__)))


ode_bench_json_file = f"{project_root}/DATA/strogatz_extended.json"

random.seed(234)
np.random.seed(234)


# Function to create the rhs function string and execute it
def create_rhs_function(eq, symbolic_states, params):
    # Determine if 'states' is expected to be a 1D or 2D array
    # ndim_states = len(states[0]) if isinstance(states[0], (list, np.ndarray)) else 1

    # Check if there's a '|' to determine if it's a tuple output
    has_pipe = "|" in eq

    func_str = "def rhs(states, *params):\n"

    if has_pipe:
        # Split the equation by '|'
        output_parts = eq.split("|")

        # Create the function string for tuple output

        # Process each part of the output
        outputs = []
        for part in output_parts:
            part = part.strip()
            # Replace ^ with ** for exponentiation
            part = part.replace("^", "**")
            # Replace state and parameter placeholders with function arguments
            for i, state in enumerate(symbolic_states):
                part = part.replace(f"x_{i}", f"states[{i}]")
            for i, param in enumerate(params):
                part = part.replace(f"c_{i}", f"params[{i}]")
            # Replace common functions with numpy equivalents
            part = part.replace("exp(", "np.exp(")
            part = part.replace("cos(", "np.cos(")
            part = part.replace("sin(", "np.sin(")
            part = part.replace("tan(", "np.tan(")
            part = part.replace("cot(", "1/np.tan(")
            part = part.replace("log(", "np.log(")

            outputs.append(part)

        # Join outputs with a comma to return them as a tuple
        func_str += f"    return ({', '.join(outputs)})\n"  # (ndim, ts)

    else:
        # Create the function string for scalar output
        eq = eq.strip()
        # Replace ^ with ** for exponentiation
        eq = eq.replace("^", "**")
        # Replace state and parameter placeholders with function arguments
        for i, state in enumerate(symbolic_states):
            eq = eq.replace(f"x_{i}", f"states[{i}]")
        for i, param in enumerate(params):
            eq = eq.replace(f"c_{i}", f"params[{i}]")
        # Replace common functions with numpy equivalents
        # Replace common functions with numpy equivalents
        eq = eq.replace("exp(", "np.exp(")
        eq = eq.replace("cos(", "np.cos(")
        eq = eq.replace("sin(", "np.sin(")
        eq = eq.replace("tan(", "np.tan(")
        eq = eq.replace("cot(", "1/np.tan(")
        eq = eq.replace("log(", "np.log(")
        func_str += f"    return {eq}\n"

    # Execute the function string to define the function
    local_vars = {}
    exec(func_str, globals(), local_vars)
    return local_vars["rhs"]


# Load JSON data
with open(ode_bench_json_file, "r") as file:
    data = json.load(file)

# Number of samples
N = 100


# Access individual terms and create the function
for id, item in enumerate(data):
    eq = item.get("eq")
    consts = item.get("consts", [])
    ndim = item.get("dim", 1)
    if len(consts[0]) == 0:
        print(f"Skipping id={id} as no consts are provided")
        continue
    const_description = item.get("const_description")
    var_description = item.get("var_description")

    # Extract states and parameters from descriptions
    states = [var.split(":")[0].strip() for var in var_description.split(",")]
    params = [const.split(":")[0].strip() for const in const_description.split(",")]

    rhs = create_rhs_function(eq, states, params)

    # Print the function definition for verification
    print(f"id = {item.get('id', [])}, eq={eq}")

    # Now you can call the rhs function with appropriate arguments
    y0 = np.array(item.get("init", [])[0])
    lower_bounds = consts[0]
    upper_bounds = [l + 0.25 for l in lower_bounds]

    solutions = item.get("solutions", [])
    t_eval = solutions[0][0].get("t")
    t_span = (t_eval[0], t_eval[-1])

    test_ys = np.array(solutions[0][0].get("y"))

    sampled_params = np.random.uniform(lower_bounds, upper_bounds, (N, len(lower_bounds)))
    est_params = []

    for run_id, params in enumerate(sampled_params):
        # print(f"------------------- Run {run_id} -------------------")
        # Solve ODE
        sol = solve_ivp(lambda t, y: rhs(y, *params), t_span, y0, t_eval=t_eval, rtol=1e-8)
        ys = np.array(sol.y)  # (state_dim, ts)
        ts = sol.t

        res = rhs(ys, *params)

        target = np.gradient(ys.T, ts, axis=0).T

        def residuals(params, input, target):
            return (rhs(input, *params) - target).ravel()

        bounds = (lower_bounds, upper_bounds)
        res = least_squares(residuals, lower_bounds, args=(ys, target), bounds=bounds)
        est_params += [res.x.tolist()]

    est_params = np.array(est_params).reshape(-1, len(lower_bounds))
    erros = np.square(est_params - sampled_params) ** 0.5
    mean = erros.mean(axis=0)
    std = erros.std(axis=0)
    print("mse mean +- std", mean.mean(), std.mean())
