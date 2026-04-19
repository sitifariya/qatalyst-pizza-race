# Qatalyst Pizza Race

> An open-source benchmark and browser game for comparing classical and quantum optimisation approaches to vehicle routing problems.

**Status:** In development · Target launch: 25 May 2026
**Author:** Siti Fariya, Qatalyst Quantum Ltd
**Funding:** Supported by Qollab and IonQ through the Qollab Creative Challenge 2026

---

## What is this?

Pizza Race is a browser game that turns vehicle routing problems into pizza
delivery puzzles. Players plan routes, then watch classical and quantum solvers
attempt the same problem.

Behind the scenes, this repository contains:

1. **An open benchmark suite** of 9 vehicle routing puzzles with increasing
   complexity (time windows, fuel limits, multi-van, VIPs, shared chargers,
   stochastic demand).
2. **A classical optimisation engine** written in Python, with brute-force
   enumeration and simulated annealing solvers.
3. **The game itself**, a single-page browser app with educational explainers.
4. **Paper-ready benchmark logging**, so every solve contributes to an empirical
   dataset for classical-quantum comparison research.

## Why this exists

Most quantum optimisation demos either oversell near-term quantum hardware or
undersell classical baselines. Pizza Race is designed to be honest:

- On small problems (Stages 0-8), classical solvers find provably optimal
  solutions in milliseconds. We show this.
- On larger and stochastic problems (Stages 10+, coming soon), quantum
  approaches begin to match classical approximate algorithms in specific
  regimes. We will show this too, once IonQ hardware access is wired up.

The goal is educational honesty, not quantum hype.

## Repository layout

```
qatalyst-pizza-race/
├── engine/          Classical solver API (Python + FastAPI)
├── game/            Pizza Race browser game (HTML + JS)
├── docs/            Technical report, methodology, benchmark data
├── LICENSE          MIT
└── README.md        This file
```

## Quick start

### Play the game

Live version: https://qatalyst-quantum.co.uk/play

The live game is deployed from a separate marketing site repository. The
`game/play.html` in this repository is a **reference snapshot** included for
academic reproducibility, under MIT license.

To run the reference snapshot locally:
```bash
cd game
python -m http.server 8080
# Open http://localhost:8080/play.html
```

### Run the engine locally

```bash
cd engine
pip install -r requirements.txt
python api.py
# API on http://localhost:8000
```

Test the health check:
```bash
curl http://localhost:8000/api/pizza/health
```

See `engine/README.md` for full API documentation.

## Cluster 1: classical territory (available now)

Stages 0-8. Small vehicle routing problems with constraints that teach one
concept at a time:

| Stage | Name | N | Vans | Special constraint |
|-------|------|---|------|--------------------|
| 0 | First delivery | 3 | 1 | Tutorial, no constraints |
| 1 | The rush | 5 | 1 | Time windows, fuel |
| 2 | The split | 6 | 2 | Multi-van |
| 3 | The rush (practice) | 5 | 1 | Time windows, fuel |
| 4 | The split (practice) | 6 | 2 | Multi-van |
| 5 | VIP rush | 7 | 1 | VIP customers |
| 6 | Friday night madness | 8 | 2 | VIPs + peak hour (boss) |
| 7 | Charger shuffle | 8 | 2 | Shared charger |
| 8 | The maybe list | 8 | 1 | Stochastic demand |

At these sizes, classical brute-force enumeration is tractable and returns
provably optimal solutions. This cluster establishes ground-truth benchmarks.

## Cluster 2: quantum territory (coming, after Qollab kickoff)

Stages 10-18 (in development). Larger problems where classical brute force
becomes infeasible and quantum approaches start to matter:

- Stages 10-13: near-term quantum (QAOA on IonQ Forte, 36 qubits)
- Stages 14-18: future-scale problems, simulated today, targeting hardware
  that ships 2026-2030

Cluster 2 will include live IonQ hardware integration once credits are
provisioned.

## Citing this work

If you use this benchmark suite in academic work, please cite:

> Fariya, S. (2026). *Pizza Race: A public benchmark for classical-quantum
> vehicle routing across problem scales*. Qatalyst Quantum. [forthcoming
> technical report, May 2026].

## Acknowledgements

This project was made possible by the Qollab Creative Challenge 2026, with
direct funding from Qollab and compute credits from IonQ.

## Licence

MIT. See [LICENSE](LICENSE).
