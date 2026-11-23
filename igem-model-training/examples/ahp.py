"""AHP microalgae table audit and a Nebula-inspired instrument route."""

from argparse import ArgumentParser
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# Saaty-style RI reference. Use one documented RI series throughout.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

CRITERIA = ["Energy", "Cost", "Contamination risk", "Efficiency",
            "Technology availability", "Strain flexibility", "Production time"]
OPTIONS = ["Centrifugation", "Crossflow filtration", "Inorganic flocculation",
           "Organic flocculation", "Bioflocculation", "Electrocoagulation",
           "Flocculation-sedimentation"]

# Albaqi et al. Table 2, row and column order matching CRITERIA.
CRITERIA_MATRIX = np.array([
    [1,   3,   3,   5,   4,   3,   3],
    [1/3, 1,   4,   4,   2,   3,   2],
    [1/3, 1/4, 1,   3,   1,   1,   1],
    [1/5, 1/4, 1/3, 1,   1,   1,   1],
    [1/4, 1/2, 1,   1,   1,   4,   3],
    [1/3, 1/3, 1,   1,   1/4, 1,   2],
    [1/3, 1/2, 1,   1,   1/3, 1/2, 1],
])

# Printed Table 3 ordering, confirmed by the paper's worked centrifugal example.
PUBLISHED_WEIGHTS = np.array([0.339, 0.214, 0.098, 0.066, 0.133, 0.079, 0.071])

# Rounded local priorities from Table 8; row and column order as above.
PUBLISHED_LOCAL = np.array([
    [0.061, 0.079, 0.099, 0.241, 0.032, 0.182, 0.257],
    [0.207, 0.217, 0.138, 0.104, 0.032, 0.070, 0.166],
    [0.108, 0.136, 0.094, 0.192, 0.155, 0.261, 0.074],
    [0.108, 0.136, 0.239, 0.158, 0.297, 0.182, 0.105],
    [0.189, 0.167, 0.134, 0.065, 0.155, 0.070, 0.133],
    [0.121, 0.113, 0.051, 0.058, 0.032, 0.053, 0.133],
    [0.207, 0.151, 0.246, 0.183, 0.297, 0.182, 0.133],
])
PUBLISHED_RANK_SCORES = np.array([0.100, 0.158, 0.134, 0.161, 0.153, 0.092, 0.202])


def priorities(matrix):
    """Return Perron weights, local consistency index, and consistency ratio."""
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    if matrix.shape != (size, size) or size not in RANDOM_INDEX:
        raise ValueError("Provide one square pairwise matrix with size 1 to 10.")
    if np.any(matrix <= 0) or not np.allclose(matrix * matrix.T, 1):
        raise ValueError("Judgments must be positive and reciprocal.")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    position = np.argmax(eigenvalues.real)
    largest = eigenvalues[position].real
    weights = eigenvectors[:, position].real
    weights /= weights.sum()
    ci = max(0.0, (largest - size) / (size - 1)) if size > 1 else 0.0
    ri = RANDOM_INDEX[size]
    return weights, ci, ci / ri if ri else 0.0


def reciprocal_three(first_second, first_third, second_third):
    return np.array([
        [1, first_second, first_third],
        [1 / first_second, 1, second_third],
        [1 / first_third, 1 / second_third, 1],
    ])


def changed_weight(weight, index, new_value):
    revised = weight * (1 - new_value) / (1 - weight[index])
    revised[index] = new_value
    return revised


def algae_case(output):
    weights, _, ratio = priorities(CRITERIA_MATRIX)
    # Preserve source rounding. Removing it would conceal the published inputs.
    scores = PUBLISHED_LOCAL @ PUBLISHED_WEIGHTS
    audited_scores = PUBLISHED_LOCAL @ weights
    print(f"Table 2 recomputed CR={ratio:.4f}; printed Table 3 CR=0.0760")
    for criterion, published, recomputed in zip(CRITERIA, PUBLISHED_WEIGHTS, weights):
        print(f"  {criterion:24s} printed={published:.3f} | matrix={recomputed:.3f}")
    print("Ranking using the printed Table 3 weights and Table 8 local priorities")
    for index in np.argsort(-scores):
        print(f"  {OPTIONS[index]:26s} {scores[index]:.4f} "
              f"(Table 9: {PUBLISHED_RANK_SCORES[index]:.3f})")
    print(f"Ranking leader using recomputed Table 2 weights: {OPTIONS[np.argmax(audited_scores)]}")
    low = PUBLISHED_LOCAL @ changed_weight(PUBLISHED_WEIGHTS, 1, 0.05)
    high = PUBLISHED_LOCAL @ changed_weight(PUBLISHED_WEIGHTS, 1, 0.75)
    print(f"Cost weight scenario 0.05 winner: {OPTIONS[np.argmax(low)]}")
    print(f"Cost weight scenario 0.75 winner: {OPTIONS[np.argmax(high)]}")

    with (output / "algae_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["alternative", "table9_published", "table3_table8_score",
                         "table2_table8_audit", "low_cost_weight", "high_cost_weight"])
        for index in np.argsort(-scores):
            writer.writerow([OPTIONS[index], PUBLISHED_RANK_SCORES[index], scores[index],
                             audited_scores[index], low[index], high[index]])


def nebula_analogue():
    # Three alternative processing routes. All judgments are synthetic.
    weight, _, criterion_ratio = priorities(reciprocal_three(3, 2, 1/2))
    local = [priorities(matrix) for matrix in (
        reciprocal_three(2, 3, 2),
        reciprocal_three(1/3, 1/2, 2),
        reciprocal_three(1/2, 3, 3),
    )]
    local_weights = np.column_stack([entry[0] for entry in local])
    scores = local_weights @ weight
    global_ci = weight @ np.array([entry[1] for entry in local])
    global_ri = weight @ np.full(3, RANDOM_INDEX[3])
    print(f"Teaching routes: weights={weight.round(3)}, criterion CR={criterion_ratio:.3f}, "
          f"global CR={global_ci / global_ri:.3f}, "
          f"local CRs={[round(float(entry[2]), 3) for entry in local]}")

    # Each edge has synthetic scaled penalties for speed, cost, and reliability.
    # Reusing a leg is allowed because edges represent actions shared by routes.
    edges = [
        (0, 1, (0.1, 0.7, 0.15)),  # Start -> Direct scanner
        (0, 2, (0.4, 0.2, 0.25)),  # Start -> Batch reader
        (1, 3, (0.2, 0.55, 0.3)),  # Direct scanner -> Review
        (1, 4, (0.4, 0.6, 0.2)),   # Direct scanner -> Done
        (2, 3, (0.1, 0.2, 0.2)),   # Batch reader -> Review
        (3, 4, (0.1, 0.1, 0.1)),   # Review -> Done
    ]
    graph = np.zeros((5, 5))
    for start, end, penalties in edges:
        graph[start, end] = weight @ penalties
    distance, predecessor = dijkstra(csr_matrix(graph), indices=0, return_predecessors=True)
    path = [4]
    while path[-1] != 0:
        path.append(predecessor[path[-1]])
    names = ["Start", "Direct scanner", "Batch reader", "Review", "Done"]
    path_labels = " -> ".join(names[index] for index in reversed(path))
    print(f"Teaching alternative ranking: {scores.round(3)}")
    print(f"Teaching graph shortest path: {path_labels}, cost={distance[4]:.3f}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/ahp"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    algae_case(args.output)
    nebula_analogue()


if __name__ == "__main__":
    main()
