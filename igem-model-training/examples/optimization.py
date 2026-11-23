"""Solve a miniature flux model, an operating-room load model and a smooth fit."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, linprog, milp
from scipy.sparse import coo_matrix


def phosphate_flux():
    # Rows are internal P and C. Columns are their uptake, growth, and P storage.
    stoichiometry = np.array([[1, 0, -1, -1], [0, 1, -2, -0.5]])
    objective = np.array([0, 0, 0, -1])
    rows = []
    for maximum_p in (1, 2, 3, 4, 5, 6):
        result = linprog(
            objective,
            A_eq=stoichiometry,
            b_eq=np.zeros(2),
            bounds=[(0, maximum_p), (0, 4), (1, None), (0, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(result.message)
        np.testing.assert_allclose(stoichiometry @ result.x, 0, atol=1e-9)
        rows.append((maximum_p, *result.x))
    return rows


def solve_bigg_model(path):
    with path.open(encoding="utf-8") as stream:
        model = json.load(stream)
    metabolite_index = {entry["id"]: index for index, entry in enumerate(model["metabolites"])}
    reactions = model["reactions"]
    row_indices, column_indices, coefficients = [], [], []
    for column, reaction in enumerate(reactions):
        for metabolite, coefficient in reaction["metabolites"].items():
            row_indices.append(metabolite_index[metabolite])
            column_indices.append(column)
            coefficients.append(coefficient)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(metabolite_index), len(reactions)),
    ).tocsc()
    objective = np.array([entry.get("objective_coefficient", 0) for entry in reactions])
    if not np.any(objective):
        raise ValueError("No objective_coefficient found in the BiGG JSON")
    bounds = [(entry["lower_bound"], entry["upper_bound"]) for entry in reactions]
    result = linprog(-objective, A_eq=matrix, b_eq=np.zeros(matrix.shape[0]), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    residual = np.max(np.abs(matrix @ result.x))
    print(f"BiGG {model['id']}: {matrix.shape[0]} metabolites, {matrix.shape[1]} reactions")
    print(f"BiGG default objective {objective @ result.x:.7f}; max mass-balance residual {residual:.2e}")


def room_loads():
    # Pulido et al. Tables 2 and 3, instance 2. One common reference set of
    # durations (surgeon k=1) makes this a room-capacity abstraction.
    types = ("E", "E", "D", "F", "G")
    duration = np.array([150, 150, 105, 210, 275])
    n_jobs, n_rooms = len(types), 3
    n_binary = n_jobs * n_rooms
    lower = np.zeros(n_binary + 1)
    upper = np.ones(n_binary + 1)
    upper[-1] = duration.sum()
    d_index = types.index("D") * n_rooms
    upper[d_index + 1 : d_index + n_rooms] = 0

    assign = np.zeros((n_jobs, n_binary + 1))
    for job in range(n_jobs):
        assign[job, job * n_rooms : (job + 1) * n_rooms] = 1
    capacity = np.zeros((n_rooms, n_binary + 1))
    for room in range(n_rooms):
        capacity[room, room:n_binary:n_rooms] = duration
    capacity[:, -1] = -1
    objective = np.zeros(n_binary + 1)
    objective[-1] = 1

    result = milp(
        objective,
        integrality=np.r_[np.ones(n_binary, dtype=int), 0],
        bounds=Bounds(lower, upper),
        constraints=[LinearConstraint(assign, 1, 1), LinearConstraint(capacity, -np.inf, 0)],
    )
    if not result.success:
        raise RuntimeError(result.message)
    choices = np.argmax(result.x[:n_binary].reshape((n_jobs, n_rooms)), axis=1)
    loads = np.bincount(choices, weights=duration, minlength=n_rooms)
    exact = min(
        np.bincount(plan, weights=duration, minlength=n_rooms).max()
        for plan in itertools.product(range(n_rooms), repeat=n_jobs)
        if plan[types.index("D")] == 0
    )
    assert np.isclose(result.fun, exact) and np.isclose(loads.max(), exact)
    return types, duration, choices, loads


def logistic_nonlinear_fit():
    # Synthetic normalized biomass proxies. See the real iGEM yeast case in lesson 13.
    t = np.arange(0, 26, 2)
    seed = np.random.default_rng(2027)
    truth = 1.3 / (1 + ((1.3 / 0.052) - 1) * np.exp(-0.3 * t))
    observations = truth + seed.normal(0, 0.02, t.size)

    def predict(parameters, time):
        limit, rate, initial = parameters
        return limit / (1 + (limit / initial - 1) * np.exp(-rate * time))

    fit = least_squares(
        lambda parameters: predict(parameters, t) - observations,
        x0=[1.0, 0.2, 0.06],
        bounds=([0.2, 0.01, 0.005], [3.0, 2.0, 0.19]),
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return t, observations, predict(fit.x, t), fit.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/optimization"))
    parser.add_argument("--bigg-json", type=Path, help="optional downloaded BiGG iND750 JSON")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = phosphate_flux()
    with (args.output / "phosphate_flux.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["P_uptake_cap", "P_uptake", "C_uptake", "growth", "P_storage"])
        writer.writerows(rows)

    types, durations, choices, loads = room_loads()
    with (args.output / "room_assignments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_type", "duration_min", "room"])
        writer.writerows(zip(types, durations, choices + 1))
    t, observations, fitted, parameters = logistic_nonlinear_fit()
    print(f"Phosphate storage at caps 1–6: {[round(float(row[-1]), 2) + 0.0 for row in rows]}")
    print(f"Room load (minutes): {loads.tolist()}, max {loads.max():.0f}")
    print(f"Nonlinear fit (K, r, X0): {parameters.round(4).tolist()}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    axes[0].plot([row[0] for row in rows], [row[-1] for row in rows], marker="o")
    axes[0].set(xlabel="P uptake bound", ylabel="Max P storage flux")
    axes[1].bar(np.arange(1, len(loads) + 1), loads)
    axes[1].set(xlabel="Room", ylabel="Workload (min)", xticks=[1, 2, 3])
    axes[2].scatter(t, observations, label="Synthetic observations")
    axes[2].plot(t, fitted, label="Fit")
    axes[2].set(xlabel="Time (h)", ylabel="Normalized biomass")
    axes[2].legend(fontsize=7)
    fig.savefig(args.output / "optimization.png", dpi=150)
    plt.close(fig)
    if args.bigg_json is not None:
        solve_bigg_model(args.bigg_json)


if __name__ == "__main__":
    main()
