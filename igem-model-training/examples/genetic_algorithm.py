"""Compare genetic and random search for a small simulated optical duty schedule."""
import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Each slot is one time window, and one represents using the brighter setting.
WEIGHTS = np.array([1, 2, 1, 4, 6, 4, 2, 1, 3, 5, 4, 2])
MAX_ON = 5


def fitness(schedule):
    on = sum(schedule)
    if on > MAX_ON:
        return -10 * (on - MAX_ON)
    coverage_proxy = np.dot(schedule, WEIGHTS)
    consecutive_on = sum(left and right for left, right in zip(schedule, schedule[1:]))
    return float(coverage_proxy - 3 * consecutive_on)


def genetic_search(rng, population_size=24, generations=24):
    population = rng.integers(0, 2, size=(population_size, WEIGHTS.size))
    history = []
    for generation in range(generations):
        scores = np.array([fitness(schedule) for schedule in population])
        winner = np.argmax(scores)
        history.append(scores[winner])
        if generation == generations - 1:
            return population[winner], history

        def tournament():
            candidates = rng.integers(population_size, size=3)
            return population[candidates[np.argmax(scores[candidates])]]

        next_population = [population[winner].copy()]
        while len(next_population) < population_size:
            first, second = tournament(), tournament()
            if rng.random() < 0.85:
                split = rng.integers(1, WEIGHTS.size)
                child = np.concatenate((first[:split], second[split:]))
            else:
                child = first.copy()
            child[rng.random(WEIGHTS.size) < 1 / WEIGHTS.size] ^= 1
            next_population.append(child)
        population = np.stack(next_population)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, default=Path("build/genetic_algorithm"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    population_size, generations = 24, 24
    exact = max(fitness(s) for s in itertools.product((0, 1), repeat=WEIGHTS.size))
    ga_histories, random_histories = [], []
    ga_successes = random_successes = 0
    for seed in np.random.SeedSequence(args.seed).spawn(20):
        ga_seed, random_seed = seed.spawn(2)
        schedule, ga_history = genetic_search(np.random.default_rng(ga_seed), population_size, generations)
        random_rng = np.random.default_rng(random_seed)
        random_candidates = random_rng.integers(0, 2, size=(generations, population_size, WEIGHTS.size))
        random_history = np.maximum.accumulate([max(fitness(s) for s in group) for group in random_candidates])
        ga_histories.append(ga_history)
        random_histories.append(random_history)
        ga_successes += fitness(schedule) == exact
        random_successes += random_history[-1] == exact

    assert max(map(max, ga_histories)) <= exact
    assert max(map(max, random_histories)) <= exact
    print(f"Exact optimum {exact:g}; {population_size * generations} evaluations per run.")
    print(f"Reached optimum: GA {ga_successes}/20; uniform search {random_successes}/20.")
    print(f"Final example schedule {schedule.tolist()}; score {fitness(schedule):g}.")

    fig, ax = plt.subplots(figsize=(7, 4))
    evaluations = population_size * np.arange(1, generations + 1)
    for history, label in [(ga_histories, "Genetic algorithm"), (random_histories, "Uniform search")]:
        q25, median, q75 = np.percentile(history, [25, 50, 75], axis=0)
        ax.plot(evaluations, median, label=label)
        ax.fill_between(evaluations, q25, q75, alpha=0.18)
    ax.axhline(exact, ls="--", color="#404857", label="Exact optimum")
    ax.set(xlabel="Fitness evaluations", ylabel="Best score so far")
    ax.legend()
    fig.savefig(args.output / "search.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
