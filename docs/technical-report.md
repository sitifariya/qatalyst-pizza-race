# Pizza Race Technical Report (Draft)

**Authors:** Siti Fariya (Qatalyst Quantum Ltd)
**Date:** To be published at launch (25 May 2026)
**Status:** Draft

## Abstract

We present Pizza Race, an open benchmark suite and browser game for
comparing classical and quantum optimisation approaches to vehicle routing
problems. The suite consists of 9 puzzles (Cluster 1) at N ≤ 8 customers,
where classical brute-force enumeration produces provably optimal solutions,
and (in future work) 9 additional puzzles (Cluster 2) at N ≥ 10 where
quantum approaches become algorithmically relevant. The benchmark suite is
released under MIT license as a public artefact for educational and research
use.

## 1. Introduction

[To be written]

## 2. Problem formulation

Each stage in the suite is a vehicle routing problem variant. The core
decision variables are:

- Assignment of customers to vans (in multi-van stages)
- Ordering of customers within each van's route

Constraints that appear across stages:

- **Time windows** ("hot-by" times): customers must be served before a
  deadline, else a cold-delivery penalty applies
- **Fuel capacity**: per-van distance limit
- **VIP priorities**: some customers carry 3x the cold penalty
- **Shared charger**: at most one van can charge at a time
- **Stochastic demand**: some customers have a 50% cancellation probability

The objective function is a weighted sum:

```
score = 100·hot − 50·cold − 150·vip_cold − penalties
```

where penalties cover fuel overruns and charger conflicts.

## 3. Algorithms

### 3.1 Brute-force enumeration (Cluster 1)

For N ≤ 8, we enumerate all N! permutations and return the solution with the
highest score. In multi-van stages, we additionally enumerate all 2^N ways
of splitting customers between two vans.

### 3.2 Brute-force stochastic (Stage 8)

For stages with stochastic demand, we enumerate all permutations and all
2^k cancellation scenarios, returning the route with the highest expected
score across scenarios.

### 3.3 Simulated annealing (Cluster 2, reserved)

For future stages at N ≥ 10, we employ simulated annealing with 2-opt moves
and geometric cooling. This is chosen for methodological parity with quantum
annealing and QAOA, enabling direct comparison.

## 4. Results

### 4.1 Cluster 1 benchmark results

| Stage | N | Vans | Solver | Optimal score | Time (ms) |
|-------|---|------|--------|---------------|-----------|
| 0 | 3 | 1 | Brute force | -948 | < 1 |
| 1 | 5 | 1 | Brute force | -472 | 2 |
| 2 | 6 | 2 | Brute force multi | -191 | 22 |
| 3 | 5 | 1 | Brute force | -472 | 2 |
| 4 | 6 | 2 | Brute force multi | -191 | 40 |
| 5 | 7 | 1 | Brute force | -717 | 82 |
| 6 | 8 | 2 | Brute force multi | -331 | 1,700 |
| 7 | 8 | 2 | Brute force multi | +20 | 2,000 |
| 8 | 8+3 | 1 | Brute force stochastic | -551 | 5,700 |

All times measured on a single CPU core (local development machine). Render
free-tier deployment is approximately 5-10x slower due to shared CPU.

### 4.2 Solver behaviour observations

- Stage 5 (VIP rush) has a large optimal cold count because the geometry
  makes all-hot delivery impossible under the fuel budget.
- Stage 7 (shared charger) shows the highest optimal score among constrained
  stages because the scoring function rewards hot deliveries strongly and
  the two-van split mitigates charger conflicts.
- Stage 8's solve time is dominated by the 2^3 = 8 scenario enumeration on
  top of 8! permutations, but remains tractable under 6 seconds.

## 5. Limitations

- Brute force does not scale beyond N = 10.
- The game's scoring function is pedagogical rather than derived from an
  operational cost model. Standard VRP metrics (total distance, total
  tardiness) are reported alongside in future work.
- Render free-tier deployment has cold-start latency (20-30s after idle)
  which affects reported runtimes for remote solves.

## 6. Future work

Cluster 2 stages (10-18) will extend the suite to larger problems and wire
in quantum hardware integration via IonQ. The full paper, including
classical-quantum comparison data, is targeted for July 2026.

## 7. Reproducibility

All code, stage definitions, and benchmark logs are available at:

https://github.com/sitifariya/qatalyst-pizza-race

Under MIT license. Citable as:

> Fariya, S. (2026). Pizza Race: A public benchmark for classical-quantum
> vehicle routing across problem scales. Qatalyst Quantum Ltd.

## Acknowledgements

This work was supported by the Qollab Creative Challenge 2026, with direct
funding from Qollab and compute credits from IonQ.
