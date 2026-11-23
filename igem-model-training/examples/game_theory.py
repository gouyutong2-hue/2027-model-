"""Zero-sum search, yeast coexistence and an exact Shapley exercise."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, linprog


def zero_sum_game(payoff):
    """Solve for both strategies. A is the row player's payoff; the column player minimizes it."""
    payoff = np.asarray(payoff, dtype=float)
    rows, columns = payoff.shape
    row_lp = linprog(
        [0.0] * rows + [-1.0],
        A_ub=np.column_stack((-payoff.T, np.ones(columns))),
        b_ub=np.zeros(columns),
        A_eq=[[*np.ones(rows), 0.0]],
        b_eq=[1.0],
        bounds=[(0, None)] * rows + [(None, None)],
        method="highs",
    )
    column_lp = linprog(
        [0.0] * columns + [1.0],
        A_ub=np.column_stack((payoff, -np.ones(rows))),
        b_ub=np.zeros(rows),
        A_eq=[[*np.ones(columns), 0.0]],
        b_eq=[1.0],
        bounds=[(0, None)] * columns + [(None, None)],
        method="highs",
    )
    if not row_lp.success or not column_lp.success:
        raise RuntimeError(f"Row LP: {row_lp.message}; column LP: {column_lp.message}")
    row_policy, column_policy = row_lp.x[:-1], column_lp.x[:-1]
    guaranteed = min(row_policy @ payoff)
    upper_bound = max(payoff @ column_policy)
    if not np.isclose(guaranteed, upper_bound, atol=1e-8, rtol=1e-8):
        raise RuntimeError("The two players' bounds disagree.")
    return guaranteed, row_policy, column_policy


def search_game(capture_probability, relocation_risk):
    """Alpern et al. Section 6.2.2. Row: search location. Column: hiding location."""
    capture_probability = np.asarray(capture_probability, dtype=float)
    relocation_risk = np.asarray(relocation_risk, dtype=float)
    eventual_values = np.zeros(capture_probability.size)
    for _ in range(500):
        found_values = capture_probability + (1 - capture_probability) * eventual_values
        updated = []
        for risk in relocation_risk:
            # Every search decision wins on capture during transit. The diagonal
            # also captures the event of discovering prey and catching it later.
            payoff = np.broadcast_to(risk, relocation_risk.shape).copy()
            payoff += np.diag((1 - risk) * found_values)
            update, _, _ = zero_sum_game(payoff)
            updated.append(update)
        if np.max(np.abs(updated - eventual_values)) < 1e-8:
            break
        eventual_values = np.array(updated)
    else:
        raise RuntimeError("Value iteration did not converge.")
    found_values = capture_probability + (1 - capture_probability) * np.array(updated)
    return zero_sum_game(np.diag(found_values)), updated


def pure_nash(row_payoff, column_payoff):
    """Both players maximize their own payoffs in a bimatrix game."""
    row_best = np.isclose(row_payoff, row_payoff.max(axis=0))
    column_best = np.isclose(column_payoff, column_payoff.max(axis=1, keepdims=True))
    return np.argwhere(row_best & column_best)


def delta(fraction, efficiency, cost, exponent):
    """Gore et al. Figure 2; benefit minus cost for C, minus benefit for D."""
    public = (1 - efficiency) * fraction
    return (efficiency + public) ** exponent - cost - public ** exponent


def yeast_equilibrium(efficiency, cost, exponent):
    rare = delta(0, efficiency, cost, exponent)
    common = delta(1, efficiency, cost, exponent)
    if np.isclose(rare, 0, atol=1e-12) and np.isclose(common, 0, atol=1e-12):
        return np.nan
    if rare <= 0:
        return 0.0
    if common >= 0:
        return 1.0
    return brentq(delta, 0, 1, args=(efficiency, cost, exponent))


def plot_yeast(output):
    # Paper: alpha = 0.15; ~0.01 privately captured. Cost is a teaching assumption.
    efficiency, cost = 0.01, 0.02
    equilibrium = yeast_equilibrium(efficiency, cost, 0.15)
    print(f"Yeast, alpha=0.15: stable fraction={equilibrium:.4f}; "
          f"linear baseline={yeast_equilibrium(efficiency, cost, 1.0):.1f}")

    figure, (phase, curves) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    efficiency_grid = np.linspace(0, 1, 201)
    both_produce_threshold = delta(1, efficiency_grid, 0, 0.15)
    rare_producer_threshold = delta(0, efficiency_grid, 0, 0.15)
    phase.fill_betweenx(efficiency_grid, 0, both_produce_threshold, label="C fixes")
    phase.fill_betweenx(efficiency_grid, both_produce_threshold, rare_producer_threshold,
                        label="Coexistence")
    phase.fill_betweenx(efficiency_grid, rare_producer_threshold, 1, label="D fixes")
    phase.plot([0, 1], [0, 1], "k--", label="Linear boundary")
    phase.set(xlabel="Normalized cooperation cost c", ylabel="Private capture fraction",
              title="Phase boundaries (Figure 2 equations)", xlim=(0, 1), ylim=(0, 1))
    phase.legend(loc="lower right", fontsize=8)

    def trajectory_rhs(_time, state):
        # Bound solver roundoff at the invariant interval; the exact ODE stays in [0, 1].
        fraction = float(np.clip(state[0], 0, 1))
        selection = delta(fraction, efficiency, cost, 0.15)
        return [fraction * (1 - fraction) * selection]

    times = np.linspace(0, 500, 301)
    for initial in (0.02, 0.1, 0.3, 0.8):
        simulation = solve_ivp(trajectory_rhs, (times[0], times[-1]), [initial],
                               t_eval=times, rtol=1e-8, atol=1e-10)
        if not simulation.success:
            raise RuntimeError(simulation.message)
        curves.plot(times, simulation.y[0], label=f"f(0)={initial:g}")
    curves.axhline(equilibrium, color="black", linestyle="--", label="Stable fraction")
    curves.set(xlabel="Model time (arbitrary units)", ylabel="Fraction of cooperators",
               title="Teaching simulation", ylim=(0, 1))
    curves.legend(fontsize=8)
    figure.savefig(output / "yeast_game.png", dpi=140)
    plt.close(figure)


def teaching_shapley():
    # A synthetic two-feature predictor. Missing inputs use a two-row reference sample.
    # The second rule has an interaction; order averaging shares its credit.
    def predict(first, second):
        return 0.1 + 0.2 * (first > 0.5) + 0.4 * (first > 0.5 and second > 0.5)

    item = (1.0, 1.0)
    background = [(0.0, 0.0), (1.0, 0.0)]

    def coalition_value(known):
        return np.mean([
            predict(*(item[index] if index in known else baseline[index] for index in range(2)))
            for baseline in background
        ])

    contribution = np.zeros(2)
    orders = list(permutations(range(2)))
    for order in orders:
        known = set()
        for index in order:
            contribution[index] += coalition_value(known | {index}) - coalition_value(known)
            known.add(index)
    contribution /= len(orders)
    baseline = coalition_value(set())
    predicted = predict(*item)
    if not np.isclose(baseline + contribution.sum(), predicted):
        raise RuntimeError("Shapley contributions must sum to prediction minus baseline.")
    print(f"Teaching Shapley: {baseline:.2f} + {contribution[0]:.2f} + "
          f"{contribution[1]:.2f} = {predicted:.2f}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/game-theory"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    one_round = np.diag([0.1, 0.8])
    print(f"One round: maximin={max(one_round.min(axis=1)):.3f}, "
          f"minimax={min(one_round.max(axis=0)):.3f}")
    value, row, column = zero_sum_game(one_round)
    print(f"Mixed: capture={value:.4f}, Searcher={row.round(4)}, Hider={column.round(4)}")

    for risk in (0.0, 0.3):
        result, states = search_game([0.1, 0.8], [[0.0, risk], [risk, 0.0]])
        value, row, column = result
        print(f"Repeated risk={risk:g}: capture={value:.4f}, Searcher={row.round(4)}, "
              f"Hider={column.round(4)}, states={np.round(states, 4)}")

    # Synthetic standard snowdrift game; the pure equilibria are asymmetric.
    row_payoff = np.array([[3.0, 1.0], [4.0, 0.0]])
    column_payoff = row_payoff.T
    print(f"Pure Nash (0=C, 1=D): {pure_nash(row_payoff, column_payoff).tolist()}")
    # Solve the indifference equation for the other player's probability of C.
    row_indifference = row_payoff[0] - row_payoff[1]
    column_indifference = column_payoff[:, 0] - column_payoff[:, 1]
    probability_column_C = row_indifference[1] / (row_indifference[1] - row_indifference[0])
    probability_row_C = column_indifference[1] / (column_indifference[1] - column_indifference[0])
    print(f"Mixed Nash: P(row C)={probability_row_C:.2f}, P(column C)={probability_column_C:.2f}")
    plot_yeast(args.output)
    teaching_shapley()


if __name__ == "__main__":
    main()
