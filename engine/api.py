"""
Qatalyst Pizza Engine - FastAPI wrapper

One endpoint: POST /api/pizza/solve

Takes a Pizza Race problem (customers, vans, constraints) in the game's
native format. Returns the optimal route computed by our classical engine.

Also:
  GET /api/pizza/health    - health check
  GET /api/pizza/stages    - list of known stage IDs (for cached fast responses)

All solves are logged to benchmarks.csv for the paper.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union
import csv
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scoring import Customer as EngineCustomer, Stage as EngineStage
from solver import solve as engine_solve


# ========= Request / response schemas =========

class CustomerIn(BaseModel):
    id: int
    x: float
    y: float
    hot_by: Optional[float] = Field(default=None, alias="hotBy")
    vip: bool = False
    maybe: bool = False

    model_config = {"populate_by_name": True}


class ProblemIn(BaseModel):
    """
    Request body for /solve.

    Matches the game's STAGE format plus a client-side request_id for
    traceability (useful when the game caches results).
    """
    stage_id: int
    shop_x: float
    shop_y: float
    customers: list[CustomerIn]
    multi_van: bool = False
    constraints: bool = False
    fuel_tank: Optional[float] = None
    shared_charger: bool = False
    has_vip: bool = False
    has_maybe: bool = False
    client_request_id: Optional[str] = None


class VanResultOut(BaseModel):
    d: float
    fuel: float
    hot: int
    cold: int
    vip_cold: int
    over_fuel: bool
    has_charged: bool


class SolveResponse(BaseModel):
    solver: str
    runtime_ms: float
    route: list  # list[int] for single van, list[list[int]] for multi
    score: int
    distance: float
    hot: int
    cold: int
    vip_cold: int
    over_fuel: bool
    charger_conflict: bool
    valid: bool
    vans: list[VanResultOut]
    iterations: int = 0
    expected_score: Optional[float] = None


# ========= App =========

app = FastAPI(
    title="Qatalyst Pizza Engine",
    description="Classical optimisation engine for Pizza Race, stages 0-8.",
    version="0.1.0",
)

# CORS: allow the game to call us from qatalyst-quantum.co.uk and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qatalyst-quantum.co.uk",
        "https://www.qatalyst-quantum.co.uk",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ========= Benchmark logging =========

BENCHMARK_FILE = Path(os.environ.get("QATALYST_BENCHMARK_FILE", "benchmarks.csv"))
_BENCHMARK_HEADERS = [
    "timestamp", "stage_id", "solver", "route", "score", "distance",
    "hot", "cold", "vip_cold", "over_fuel", "charger_conflict", "valid",
    "runtime_ms", "iterations", "n_customers", "n_vans", "has_stochastic",
    "expected_score", "client_request_id",
]


def _log_benchmark(record: dict):
    """Append one solve result to the benchmark CSV."""
    write_header = not BENCHMARK_FILE.exists()
    try:
        with BENCHMARK_FILE.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_BENCHMARK_HEADERS)
            if write_header:
                w.writeheader()
            w.writerow({k: record.get(k) for k in _BENCHMARK_HEADERS})
    except Exception as e:
        # Never fail a solve because logging failed
        print(f"[benchmark-log-error] {e}")


# ========= Routes =========

@app.get("/api/pizza/health")
def health():
    return {
        "status": "ok",
        "service": "qatalyst-pizza-engine",
        "version": "0.1.0",
        "solvers_available": [
            "bruteforce_single",
            "bruteforce_multi",
            "bruteforce_stochastic",
            "sa_single",
        ],
    }


@app.get("/api/pizza/stages")
def known_stages():
    """List the pre-baked stages we have on record."""
    from stages import STAGES
    return {
        "stages": [
            {
                "id": s.stage_id,
                "customers": len(s.customers),
                "multi_van": s.multi_van,
                "has_maybe": s.has_maybe,
                "shared_charger": s.shared_charger,
            }
            for s in STAGES.values()
        ]
    }


@app.post("/api/pizza/solve", response_model=SolveResponse)
def solve_problem(problem: ProblemIn):
    """Solve a Pizza Race problem and return the optimal route."""

    # Convert request -> engine types
    engine_customers = [
        EngineCustomer(
            id=c.id, x=c.x, y=c.y, hot_by=c.hot_by, vip=c.vip, maybe=c.maybe,
        )
        for c in problem.customers
    ]

    stage = EngineStage(
        stage_id=problem.stage_id,
        shop_x=problem.shop_x, shop_y=problem.shop_y,
        customers=engine_customers,
        multi_van=problem.multi_van,
        constraints=problem.constraints,
        fuel_tank=problem.fuel_tank,
        shared_charger=problem.shared_charger,
        has_vip=problem.has_vip,
        has_maybe=problem.has_maybe,
    )

    # Run the solver
    try:
        result = engine_solve(stage)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=501,
            detail=f"This problem size/shape is not yet supported: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Solver error: {e}",
        )

    # Log for paper benchmark
    record = asdict(result.to_benchmark(stage))
    record["client_request_id"] = problem.client_request_id
    _log_benchmark(record)

    # Response
    vans_out = [
        VanResultOut(
            d=v.d, fuel=v.fuel, hot=v.hot, cold=v.cold, vip_cold=v.vip_cold,
            over_fuel=v.over_fuel, has_charged=v.has_charged,
        )
        for v in result.score.vans
    ]

    return SolveResponse(
        solver=result.solver,
        runtime_ms=round(result.runtime_ms, 2),
        route=result.route,
        score=result.score.score,
        distance=result.score.d,
        hot=result.score.hot,
        cold=result.score.cold,
        vip_cold=result.score.vip_cold,
        over_fuel=result.score.over_fuel,
        charger_conflict=result.score.charger_conflict,
        valid=result.score.valid,
        vans=vans_out,
        iterations=result.iterations,
        expected_score=result.expected_score,
    )


# ========= Local dev entry =========

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
