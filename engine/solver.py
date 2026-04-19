"""
Qatalyst classical solver for Pizza Race stages 0-8.

Algorithms used:
  - For N <= 7 customers, single van: brute-force enumerate all permutations
  - For N <= 6 customers, multi-van: brute-force all van assignments + brute permute
  - For larger: simulated annealing with 2-opt moves

The goal is to score as well as or better than the in-browser Dreamer robot.
Since the game uses brute force up to 8 customers, we should match or beat that.

All runs emit a BenchmarkRecord for future paper data.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from itertools import permutations
import time
import random
import math

from scoring import Stage, SolutionScore, eval_solution


# ========= Benchmark record (feeds the paper) =========

@dataclass
class BenchmarkRecord:
    """One row in the benchmark log. Persist these for paper analysis."""
    timestamp: float
    stage_id: int
    solver: str              # 'bruteforce', 'sa', 'bruteforce_stochastic', etc
    route: list              # [int] for single van, [[int], [int]] for multi
    score: int
    distance: float
    hot: int
    cold: int
    vip_cold: int
    over_fuel: bool
    charger_conflict: bool
    valid: bool
    runtime_ms: float
    iterations: Optional[int] = None
    n_customers: int = 0
    n_vans: int = 1
    has_stochastic: bool = False
    expected_score: Optional[float] = None  # only for stochastic


# ========= Solver output =========

@dataclass
class SolveResult:
    route: list   # [int] for single van, [[int], [int]] for multi-van
    score: SolutionScore
    solver: str
    runtime_ms: float
    iterations: int = 0
    expected_score: Optional[float] = None  # for stochastic (Stage 8)

    def to_benchmark(self, stage: Stage) -> BenchmarkRecord:
        routes = self.route if isinstance(self.route[0], list) else [self.route]
        return BenchmarkRecord(
            timestamp=time.time(),
            stage_id=stage.stage_id,
            solver=self.solver,
            route=self.route,
            score=self.score.score,
            distance=self.score.d,
            hot=self.score.hot,
            cold=self.score.cold,
            vip_cold=self.score.vip_cold,
            over_fuel=self.score.over_fuel,
            charger_conflict=self.score.charger_conflict,
            valid=self.score.valid,
            runtime_ms=self.runtime_ms,
            iterations=self.iterations or None,
            n_customers=len(stage.customers),
            n_vans=len(routes),
            has_stochastic=stage.has_maybe,
            expected_score=self.expected_score,
        )


# ========= Brute force single van =========

def solve_bruteforce_single(stage: Stage) -> SolveResult:
    """
    Enumerate all N! permutations. Use for small N (<= 7).

    Matches what the game's Dreamer robot does on single-van stages.
    """
    t0 = time.perf_counter()
    n = len(stage.customers)
    stops = list(range(n))

    best_route = None
    best_score = None
    count = 0

    # For stage 0 (no constraints), score is -distance so higher = better
    # For stage 1+, score is hot*100 - cold*50 - vip_cold*150 - fuel_penalty
    # In both cases: maximise score

    for p in permutations(stops):
        s = eval_solution(list(p), stage)
        count += 1
        if best_score is None or s.score > best_score.score:
            best_score = s
            best_route = list(p)

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SolveResult(
        route=best_route,
        score=best_score,
        solver="bruteforce_single",
        runtime_ms=runtime_ms,
        iterations=count,
    )


# ========= Brute force multi-van (2 vans) =========

def solve_bruteforce_multi(stage: Stage) -> SolveResult:
    """
    For each of 2^N ways to split customers between 2 vans, brute-force the
    order within each van. Use for N <= 7 multi-van stages.

    This matches the Dreamer's logic in play.html exactly.
    """
    t0 = time.perf_counter()
    n = len(stage.customers)

    best_route = None
    best_score = None
    count = 0

    for mask in range(1 << n):
        van_a = []
        van_b = []
        for i in range(n):
            if mask & (1 << i):
                van_a.append(i)
            else:
                van_b.append(i)
        if not van_a or not van_b:
            continue

        # Best order for each van (minimise local cost)
        best_a = _best_order_for_van(van_a, stage)
        best_b = _best_order_for_van(van_b, stage)

        sol = [best_a, best_b]
        s = eval_solution(sol, stage)
        count += 1
        if best_score is None or s.score > best_score.score:
            best_score = s
            best_route = sol

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SolveResult(
        route=best_route,
        score=best_score,
        solver="bruteforce_multi",
        runtime_ms=runtime_ms,
        iterations=count,
    )


def _best_order_for_van(route: list[int], stage: Stage) -> list[int]:
    """
    Find best order of a single van's customer set, using the game's local
    scoring heuristic (matches JS bestOrder() in the multi-van branch of
    quantumSolve).
    """
    if len(route) <= 1:
        return route

    from scoring import eval_van_route
    best_p = None
    best_s = -1e18
    for p in permutations(route):
        p = list(p)
        vr = eval_van_route(p, stage)
        # Game's local score for ranking orderings within a van:
        vs = vr.hot * 100 - vr.cold * 50 - vr.vip_cold * 150
        if stage.fuel_tank and vr.over_fuel:
            vs -= (vr.fuel - stage.fuel_tank) * 5
        if vs > best_s:
            best_s = vs
            best_p = p
    return best_p


# ========= Stochastic (Stage 8) =========

def solve_stochastic(stage: Stage) -> SolveResult:
    """
    For Stage 8 with maybe-customers. For each permutation, compute expected
    score across all 2^k cancellation scenarios. Return route with best average.

    Matches quantumSolveStochastic() in play.html.
    """
    t0 = time.perf_counter()
    n = len(stage.customers)
    maybe_ids = [c.id for c in stage.customers if c.maybe]
    nm = len(maybe_ids)
    n_scenarios = 1 << nm

    # Pre-generate all realised sets
    scenarios = []
    for mask in range(n_scenarios):
        rs = set()
        for c in stage.customers:
            if not c.maybe:
                rs.add(c.id)
        for i in range(nm):
            if mask & (1 << i):
                rs.add(maybe_ids[i])
        scenarios.append(rs)

    stops = list(range(n))
    best_route = None
    best_expected = -1e18
    count = 0

    for p in permutations(stops):
        p = list(p)
        expected = 0.0
        for rs in scenarios:
            s = eval_solution(p, stage, rs)
            expected += s.score
        expected /= n_scenarios
        count += 1
        if expected > best_expected:
            best_expected = expected
            best_route = p

    # Eval best route on the "all show up" scenario for reporting
    all_show = set(range(n))
    final_score = eval_solution(best_route, stage, all_show)

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SolveResult(
        route=best_route,
        score=final_score,
        solver="bruteforce_stochastic",
        runtime_ms=runtime_ms,
        iterations=count,
        expected_score=round(best_expected, 2),
    )


# ========= Simulated annealing (for larger problems, reserved for Cluster 2) =========

def solve_sa_single(stage: Stage, iters: int = 5000, seed: int = 42) -> SolveResult:
    """
    Simulated annealing with 2-opt moves for single-van routes.
    Used when N > 7 (outside Cluster 1 but ready for it).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(stage.customers)

    # NN construction for initial solution
    visited = [False] * n
    route = []
    prev = stage.shop()
    for _ in range(n):
        best_j, best_d = -1, 1e18
        for j in range(n):
            if visited[j]:
                continue
            from scoring import dist
            dj = dist(prev, (stage.customers[j].x, stage.customers[j].y))
            if dj < best_d:
                best_d = dj
                best_j = j
        route.append(best_j)
        visited[best_j] = True
        prev = (stage.customers[best_j].x, stage.customers[best_j].y)

    best_route = route[:]
    best_score = eval_solution(best_route, stage)

    T = 1.0
    T_end = 0.001
    cooling = (T_end / T) ** (1.0 / iters)
    current = best_route[:]
    current_score = best_score

    for _ in range(iters):
        # 2-opt move
        i = rng.randint(0, n - 1)
        k = rng.randint(0, n - 1)
        if i == k:
            continue
        if i > k:
            i, k = k, i
        candidate = current[:i] + current[i:k+1][::-1] + current[k+1:]
        cand_score = eval_solution(candidate, stage)
        delta = cand_score.score - current_score.score
        if delta > 0 or rng.random() < math.exp(delta / (T * 1000)):
            current = candidate
            current_score = cand_score
            if cand_score.score > best_score.score:
                best_score = cand_score
                best_route = candidate[:]
        T *= cooling

    runtime_ms = (time.perf_counter() - t0) * 1000
    return SolveResult(
        route=best_route,
        score=best_score,
        solver="sa_single",
        runtime_ms=runtime_ms,
        iterations=iters,
    )


# ========= Router: pick the right solver =========

def solve(stage: Stage) -> SolveResult:
    """
    Main entry point. Routes to the right solver based on stage properties.
    """
    n = len(stage.customers)

    if stage.has_maybe:
        return solve_stochastic(stage)

    if stage.multi_van:
        if n <= 8:
            return solve_bruteforce_multi(stage)
        # For Cluster 2, larger multi-van would use SA (future)
        raise NotImplementedError("Multi-van SA reserved for Cluster 2 (N > 8)")

    if n <= 8:
        return solve_bruteforce_single(stage)
    return solve_sa_single(stage)
