"""Recompute Luo et al. Table 2 and explore common queueing models."""

from argparse import ArgumentParser
import csv
from dataclasses import dataclass
from heapq import heapreplace
from math import exp, expm1, floor, log, sqrt
from pathlib import Path
from random import Random

from scipy.optimize import Bounds, LinearConstraint, milp


# Luo et al. (2020), Table 2. Rates are per minute.
TABLE2 = [
    ("Monday", 1, 5.23, 9, 1.33),
    ("Monday", 2, 2.16, 4, 1.69),
    ("Monday", 3, 0.79, 2, 0.96),
    ("Wednesday", 1, 4.56, 8, 1.21),
    ("Wednesday", 2, 1.61, 3, 2.29),
    ("Wednesday", 3, 0.67, 2, 0.59),
    ("Saturday", 1, 2.45, 5, 0.65),
    ("Saturday", 2, 0.89, 2, 1.46),
    ("Saturday", 3, 0.42, 1, 2.98),
]
SERVICE_RATE = 0.64
TIME_LIMIT = 15.0
TAIL_LIMIT = 0.05


@dataclass(frozen=True)
class QueueMetrics:
    utilization: float
    probability_wait: float
    mean_wait: float
    mean_queue: float
    mean_stay: float
    wait_tail: float
    stay_tail: float


def erlang_b(offered_load, servers):
    """Blocking probability of a Poisson-arrival system with no waiting."""
    loss_probability = 1.0
    for current in range(1, servers + 1):
        load_times_loss = offered_load * loss_probability
        loss_probability = load_times_loss / (current + load_times_loss)
    return loss_probability


def mmc(arrival_rate, service_rate, servers, limit):
    """M/M/c with a common infinite FIFO line; steady state only."""
    if not (arrival_rate >= 0 and service_rate > 0 and servers > 0 and limit >= 0):
        raise ValueError("Provide nonnegative arrival rate/time and positive service rate/count.")
    gap = servers * service_rate - arrival_rate
    if gap <= 0:
        raise ValueError("No stationary queue when arrival rate >= total service rate.")
    offered_load = arrival_rate / service_rate
    utilization = offered_load / servers

    # Convert Erlang B to Erlang C without factorials.
    loss_probability = erlang_b(offered_load, servers)
    probability_wait = loss_probability / (1 - utilization + utilization * loss_probability)
    mean_wait = probability_wait / gap

    # When a customer waits, their time in system is a sum of two exponentials.
    # expm1 avoids cancellation when their rates are close.
    slower_rate, faster_rate = sorted((gap, service_rate))
    rate_difference = faster_rate - slower_rate
    if rate_difference == 0:
        conditional_stay_tail = exp(-slower_rate * limit) * (1 + slower_rate * limit)
    else:
        conditional_stay_tail = exp(-slower_rate * limit) * (
            1 - slower_rate * expm1(-rate_difference * limit) / rate_difference
        )

    stay_tail = (1 - probability_wait) * exp(-service_rate * limit)
    stay_tail += probability_wait * conditional_stay_tail
    return QueueMetrics(
        utilization=utilization,
        probability_wait=probability_wait,
        mean_wait=mean_wait,
        mean_queue=arrival_rate * mean_wait,
        mean_stay=mean_wait + 1 / service_rate,
        wait_tail=probability_wait * exp(-gap * limit),
        stay_tail=stay_tail,
    )


def minimum_c(arrival_rate, measure):
    """Find the first stable c that satisfies the chosen tail constraint."""
    if exp(-SERVICE_RATE * TIME_LIMIT) > TAIL_LIMIT and measure == "stay_tail":
        raise ValueError("The service time alone already exceeds the tail limit.")
    servers = floor(arrival_rate / SERVICE_RATE) + 1
    while True:
        metrics = mmc(arrival_rate, SERVICE_RATE, servers, TIME_LIMIT)
        if getattr(metrics, measure) <= TAIL_LIMIT:
            return servers, metrics
        servers += 1


def stationary_birth_death(birth_rates, death_rates):
    """Finite-state process. birth_rates[n] joins n to n+1; death_rates[n] reverses that edge."""
    if len(birth_rates) != len(death_rates):
        raise ValueError("Each pair of adjacent states needs forward and backward rates.")
    log_weights = [0.0]
    for birth, death in zip(birth_rates, death_rates):
        if birth <= 0 or death <= 0:
            raise ValueError("This example uses an irreducible chain with positive edge rates.")
        log_weights.append(log_weights[-1] + log(birth) - log(death))
    maximum = max(log_weights)
    weights = [exp(weight - maximum) for weight in log_weights]
    total = sum(weights)
    return [weight / total for weight in weights]


def simulate_queue(arrival_rate, service_rate, servers, count, warmup, seed, service="exponential"):
    """Simulate FIFO on the earliest-free server. Return waits and stays after warmup."""
    random = Random(seed)
    available_at = [0.0] * servers
    time = 0.0
    waits = []
    stays = []
    for index in range(count + warmup):
        time += random.expovariate(arrival_rate)
        start = max(time, available_at[0])
        if service == "deterministic":
            duration = 1 / service_rate
        elif service == "erlang2":
            duration = random.gammavariate(2, 1 / (2 * service_rate))
        else:
            duration = random.expovariate(service_rate)
        heapreplace(available_at, start + duration)
        if index >= warmup:
            waits.append(start - time)
            stays.append(start - time + duration)
    return waits, stays


def replicate_table(output):
    rows = []
    for day, period, rate, published_c, published_wait in TABLE2:
        servers, metrics = minimum_c(rate, "stay_tail")
        wait_servers, _ = minimum_c(rate, "wait_tail")
        rows.append({
            "day": day,
            "period": period,
            "arrivals_per_minute": rate,
            "published_c": published_c,
            "recomputed_stay_constraint_c": servers,
            "recomputed_wait_constraint_c": wait_servers,
            "published_mean_wait_min": published_wait,
            "recomputed_mean_wait_min": round(metrics.mean_wait, 4),
            "wait_exceeds_15": round(metrics.wait_tail, 6),
            "stay_exceeds_15": round(metrics.stay_tail, 6),
        })
        print(f"{day:9s} | {period} | c={servers:2d} (paper {published_c}) "
              f"| E[wait]={metrics.mean_wait:6.3f} min | P(stay>15)={metrics.stay_tail:.4f}")
    with (output / "table2_recomputed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def teaching_roster(demand):
    """Toy one-day cover problem using Table 2 demand and synthetic shifts/costs."""
    # Rows are periods, columns are pattern F (whole day), A, B, C, and AB.
    coverage = [[1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
    costs = [5.0, 2.0, 2.0, 2.0, 3.5]
    result = milp(
        c=costs,
        integrality=[1] * len(costs),
        bounds=Bounds([0] * len(costs), [float("inf")] * len(costs)),
        constraints=LinearConstraint(coverage, demand, [float("inf")] * len(demand)),
    )
    if not result.success:
        raise RuntimeError(result.message)
    counts = [round(value) for value in result.x]
    print(f"Toy roster, counts [F, A, B, C, AB] = {counts}, cost = {result.fun:g}")


def admission_lounge_baseline():
    # Veneklaas et al. Table 2 pooled values before the admission lounge.
    offered_load = 9.86 * 3.17
    print("\nErlang B: Veneklaas et al. Table 2 pooled baseline")
    for target in (0.05, 0.01):
        beds = 1
        while erlang_b(offered_load, beds) > target:
            beds += 1
        blocked = erlang_b(offered_load, beds)
        occupied = (1 - blocked) * offered_load / beds
        print(f"Target {target:.0%}: beds={beds}, block={blocked:.4f}, occupancy={occupied:.1%}")


def teaching_variants():
    print("\nQueue variants with synthetic inputs")
    for capacity in (3, 6):
        servers, arrival_rate, service_rate = 3, 1.4, 0.6
        probabilities = stationary_birth_death(
            [arrival_rate] * capacity,
            [min(n, servers) * service_rate for n in range(1, capacity + 1)],
        )
        admitted_rate = arrival_rate * (1 - probabilities[-1])
        length = sum(max(n - servers, 0) * p for n, p in enumerate(probabilities))
        print(f"M/M/3/{capacity}: blocked={probabilities[-1]:.3f}, "
              f"accepted wait={length / admitted_rate:.3f} min")
    instruments = 5
    probabilities = stationary_birth_death(
        [(instruments - n) * 0.04 for n in range(instruments)],
        [0.25] * instruments,
    )
    down = sum(n * p for n, p in enumerate(probabilities))
    print(f"Finite source: {down:.3f} of {instruments} instruments down on average")
    balking_probabilities = stationary_birth_death(
        [1.4 / (1 + n) for n in range(6)],
        [min(n, 3) * 0.6 for n in range(1, 7)],
    )
    print(f"State-dependent entry: P(full)={balking_probabilities[-1]:.4f}")
    optimal_mu = 0.42 + sqrt(0.42 * 1.5 / 2.0)
    print(f"M/M/1 continuous cost example: optimal service rate={optimal_mu:.3f}/min")

    arrival_rate, service_rate = 0.42, SERVICE_RATE
    utilization = arrival_rate / service_rate
    for service, second_moment in (
        ("exponential", 2 / service_rate**2),
        ("erlang2", 1.5 / service_rate**2),
        ("deterministic", 1 / service_rate**2),
    ):
        exact = arrival_rate * second_moment / (2 * (1 - utilization))
        waits, _ = simulate_queue(arrival_rate, service_rate, 1, 60_000, 3_000, 19, service)
        print(f"M/G/1 {service:<13s}: theory E[Wq]={exact:.3f}, simulation={sum(waits) / len(waits):.3f} min")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/queueing"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    replicate_table(args.output)
    analytic = mmc(5.23, SERVICE_RATE, 9, TIME_LIMIT)
    for seed in (7, 19, 2027):
        waits, stays = simulate_queue(5.23, SERVICE_RATE, 9, 150_000, 5_000, seed)
        empirical_tail = sum(stay > TIME_LIMIT for stay in stays) / len(stays)
        print(f"Monday period 1 seed={seed}: E[wait]={sum(waits) / len(waits):.3f} min "
              f"(analytic {analytic.mean_wait:.3f}), P(stay>15)={empirical_tail:.4f}")
    teaching_roster([9, 4, 2])
    admission_lounge_baseline()
    teaching_variants()


if __name__ == "__main__":
    main()
