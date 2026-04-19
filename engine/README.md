# Qatalyst Pizza Engine

Classical optimisation engine for the Pizza Race benchmark suite.

## What it does

Takes a Pizza Race problem (customers, vans, constraints) and returns the
optimal delivery route. Pure Python, no quantum, no LLM, no agents.

## Algorithms

| Stages | Algorithm | Why |
|--------|-----------|-----|
| 0, 1, 3, 5 | Brute force (single van) | N ≤ 7, provable optimum in milliseconds |
| 2, 4, 6, 7 | Brute force (multi-van) | 2^N splits × permutations, tractable at N=8 |
| 8 | Brute force stochastic | Enumerates 2^k cancellation scenarios |
| Reserved | Simulated annealing | For Cluster 2 (N ≥ 10) where brute force breaks |

### Why brute force for Cluster 1

At N ≤ 8, the solution space has at most 40,320 permutations. Brute force
enumerates all of them and returns the provable optimum in under 2 seconds.
Any heuristic would trade optimality for speed without meaningful speed gain
at this scale. Brute force establishes ground-truth benchmarks for
classical-quantum comparison.

### Why simulated annealing reserved for Cluster 2

Brute force becomes infeasible at N ≥ 10 (3.6 million permutations) and
impossible at N ≥ 12 (479 million). Simulated annealing scales to any N and is
the algorithmic peer to quantum annealing and QAOA, enabling direct
classical-quantum comparison at scales where both approaches face comparable
challenges.

## API

### POST /api/pizza/solve

Request:
```json
{
  "stage_id": 1,
  "shop_x": 350,
  "shop_y": 200,
  "constraints": true,
  "fuel_tank": 40,
  "customers": [
    {"id": 0, "x": 140, "y": 115, "hotBy": 24},
    {"id": 1, "x": 560, "y": 105, "hotBy": 28},
    {"id": 2, "x": 100, "y": 295, "hotBy": 40},
    {"id": 3, "x": 580, "y": 300, "hotBy": 32},
    {"id": 4, "x": 350, "y": 340, "hotBy": 20}
  ]
}
```

Response:
```json
{
  "solver": "bruteforce_single",
  "runtime_ms": 1.5,
  "route": [4, 2, 0, 1, 3],
  "score": -472,
  "distance": 144.5,
  "hot": 2,
  "cold": 3,
  "vip_cold": 0,
  "over_fuel": false,
  "charger_conflict": false,
  "valid": true,
  "vans": [{"d": 144.5, "fuel": 144.5, "hot": 2, "cold": 3, ...}],
  "iterations": 120
}
```

### GET /api/pizza/health

Health check and list of available solvers.

### GET /api/pizza/stages

List the pre-baked stages on record.

## Running locally

```bash
pip install -r requirements.txt
python api.py
# API on http://localhost:8000
```

## Deploying to Render

This engine deploys as a web service on Render. The `render.yaml` at the repo
root configures the deployment with `rootDir: engine`, so Render builds and
runs from this subdirectory.

Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

First request after 15 minutes of inactivity takes 20-30 seconds due to free
tier cold start. Subsequent requests return in milliseconds.

## Benchmark logging

Every solve appends one row to `benchmarks.csv`. Columns:

```
timestamp, stage_id, solver, route, score, distance,
hot, cold, vip_cold, over_fuel, charger_conflict, valid,
runtime_ms, iterations, n_customers, n_vans, has_stochastic,
expected_score, client_request_id
```

This data feeds the forthcoming technical report. On Render, the file lives in
the instance's ephemeral filesystem. For persistent benchmarks, set
`QATALYST_BENCHMARK_FILE` to a persistent disk path or use a database backend
(future work).

## Files

```
engine/
├── api.py            FastAPI entry point
├── scoring.py        Evaluates routes (matches game scoring exactly)
├── solver.py         The four classical solvers
├── stages.py         Stage definitions (mirrors play.html STAGES dict)
└── requirements.txt  Python dependencies
```

## What this engine does NOT do

- No QUBO construction (not needed for classical solvers)
- No LangGraph / agentic orchestration (not needed for known problems)
- No LLM calls
- No external optimisers (Gurobi, OR-Tools) — brute force matches their
  optimum at N ≤ 8 without the dependency weight
- No quantum hardware integration — that's Cluster 2 (coming)

## Licence

MIT. See [../LICENSE](../LICENSE).
