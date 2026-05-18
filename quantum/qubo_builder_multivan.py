"""
qubo_builder_multivan.py — Multi-van extension to qubo_builder.

Imports the single-van primitives from qubo_builder.py and adds multi-van
support. Variables become x[i*N*V + t*V + v] = 1 if customer i is at
position t in van v.

Variable count: N * N * V qubits.
  N=6 customers, V=2 vans → 72 qubits (simulator only, far beyond NISQ).
"""
from __future__ import annotations
import math
from itertools import permutations
import numpy as np

from qubo_builder import (
    distance, build_distance_matrix, add_qubo_term,
)


def var_index_mv(i: int, t: int, v: int, n: int, V: int) -> int:
    """Flatten (customer i, position t, van v) to a flat qubit index.

    Variables are ordered: customer-major, then position, then van.
      x[i, t, v]  ↔  bit index  i*N*V + t*V + v
    """
    return i * n * V + t * V + v


def assignment_penalty_mv(Q: dict, n: int, V: int, penalty: float):
    """Multi-van assignment constraints:
      (a) each customer visited exactly once across all vans and positions
      (b) each (position, van) slot holds at most one customer
    """
    # (a) customer-uniqueness: each customer appears once total across (t, v)
    for i in range(n):
        # diagonal terms: -penalty per (i, t, v)
        for t in range(n):
            for v in range(V):
                idx = var_index_mv(i, t, v, n, V)
                add_qubo_term(Q, idx, idx, -penalty)
        # quadratic pairs: any two (t,v), (t',v') for the same customer
        slots = [(t, v) for t in range(n) for v in range(V)]
        for a in range(len(slots)):
            for b in range(a + 1, len(slots)):
                t1, v1 = slots[a]
                t2, v2 = slots[b]
                add_qubo_term(
                    Q,
                    var_index_mv(i, t1, v1, n, V),
                    var_index_mv(i, t2, v2, n, V),
                    2 * penalty,
                )

    # (b) slot-uniqueness: each (t, v) slot has at most one customer
    # Use a soft 'at most one' (sum_i x[i,t,v]) penalty: a_2 * sum_i + a_pair pairs
    # (sum_i x)^2 - sum_i x → 2 * sum_{i<j} x_i x_j (after binary squaring)
    for t in range(n):
        for v in range(V):
            for i in range(n):
                for j in range(i + 1, n):
                    add_qubo_term(
                        Q,
                        var_index_mv(i, t, v, n, V),
                        var_index_mv(j, t, v, n, V),
                        2 * penalty,
                    )


def distance_cost_mv(Q: dict, D: np.ndarray, n: int, V: int):
    """Distance cost across both vans.

    For each van independently:
      cost = D[depot][i] * x[i,0,v] + D[i,j] * x[i,t,v]*x[j,t+1,v] + D[i][depot] * x[i,N-1,v]
    """
    depot = n
    for v in range(V):
        # depot → first
        for i in range(n):
            add_qubo_term(
                Q,
                var_index_mv(i, 0, v, n, V),
                var_index_mv(i, 0, v, n, V),
                D[depot][i],
            )
        # consecutive
        for t in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        add_qubo_term(
                            Q,
                            var_index_mv(i, t, v, n, V),
                            var_index_mv(j, t + 1, v, n, V),
                            D[i][j],
                        )
        # last → depot
        for i in range(n):
            add_qubo_term(
                Q,
                var_index_mv(i, n - 1, v, n, V),
                var_index_mv(i, n - 1, v, n, V),
                D[i][depot],
            )


def cold_penalty_mv(Q, customers, D, n, V, cold_weight):
    """Linear cold penalty per (customer, position, van).

    Best-case arrival time at position t in any van uses the smallest depot edge.
    """
    depot = n
    avg_step = float(D[np.triu_indices(n + 1, 1)].mean())
    min_first_leg = min(D[depot][k] for k in range(n))
    for i in range(n):
        deadline = customers[i].get("hotBy", None)
        if deadline is None:
            continue
        for t in range(n):
            best_case = min_first_leg + t * avg_step
            if best_case > deadline:
                for v in range(V):
                    add_qubo_term(
                        Q,
                        var_index_mv(i, t, v, n, V),
                        var_index_mv(i, t, v, n, V),
                        cold_weight,
                    )


def fuel_overrun_penalty_mv(Q, D, n, V, fuel_tank, penalty):
    """Per-van fuel overrun bias: amplify long edges per van."""
    threshold = fuel_tank / n
    depot = n
    for v in range(V):
        for i in range(n):
            if D[depot][i] > threshold:
                add_qubo_term(
                    Q,
                    var_index_mv(i, 0, v, n, V),
                    var_index_mv(i, 0, v, n, V),
                    (D[depot][i] - threshold) * penalty,
                )
        for t in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i != j and D[i][j] > threshold:
                        add_qubo_term(
                            Q,
                            var_index_mv(i, t, v, n, V),
                            var_index_mv(j, t + 1, v, n, V),
                            (D[i][j] - threshold) * penalty,
                        )
        for i in range(n):
            if D[i][depot] > threshold:
                add_qubo_term(
                    Q,
                    var_index_mv(i, n - 1, v, n, V),
                    var_index_mv(i, n - 1, v, n, V),
                    (D[i][depot] - threshold) * penalty,
                )


def decode_mv(bits, n: int, V: int):
    """Decode a flat bitstring into multi-van routes.

    Returns (routes, valid) where routes is a list of V lists of customer ids,
    or (None, False) if the bitstring isn't a valid multi-van permutation.
    """
    routes = [[] for _ in range(V)]
    used_customers = set()

    # Read assignment: which slot does each customer occupy?
    for i in range(n):
        slot = None
        for t in range(n):
            for v in range(V):
                if bits[var_index_mv(i, t, v, n, V)] == 1:
                    if slot is not None:
                        return None, False
                    slot = (t, v)
        if slot is None:
            return None, False  # customer not assigned
        if i in used_customers:
            return None, False
        used_customers.add(i)
        t, v = slot
        # We'll re-order by position after collecting
        routes[v].append((t, i))

    # Check no slot collision; reorder each van's list by position
    for v in range(V):
        positions = [t for t, _ in routes[v]]
        if len(set(positions)) != len(positions):
            return None, False
        routes[v].sort(key=lambda x: x[0])
        routes[v] = [i for _, i in routes[v]]

    return routes, True


def enumerate_brute_force_mv(customers, shop, fuel_tank=None, V=2):
    """Brute force over partitions + orderings for verification.

    Returns (best_routes, best_score, best_qubo_energy_if_provided).
    For N=6 V=2 this is 5040 candidates, fast.
    """
    n = len(customers)
    best = None

    def dist2(a, b): return math.hypot(a[0] - b[0], a[1] - b[1]) / 10.0

    def eval_van(route):
        if not route:
            return {"d": 0, "fuel": 0, "hot": 0, "cold": 0, "overFuel": False}
        d = 0; fuel = 0; t = 0; cold = 0
        prev = shop
        for ci in route:
            c = customers[ci]
            leg = dist2(prev, (c["x"], c["y"]))
            d += leg; fuel += leg; t += leg
            if t > c.get("hotBy", 1e9):
                cold += 1
            prev = (c["x"], c["y"])
        back = dist2(prev, shop)
        d += back; fuel += back
        return {"d": d, "fuel": fuel, "hot": len(route) - cold, "cold": cold,
                "overFuel": fuel > fuel_tank if fuel_tank else False}

    def score(sol):
        vans = [eval_van(r) for r in sol]
        h = sum(v["hot"] for v in vans)
        c = sum(v["cold"] for v in vans)
        s = h * 100 - c * 50
        for v in vans:
            if v["overFuel"]:
                s -= (v["fuel"] - fuel_tank) * 5
        return s, h, c, vans

    for mask in range(1 << n):
        partitions = [[], []]
        for i in range(n):
            partitions[1 if mask & (1 << i) else 0].append(i)
        for pa in permutations(partitions[0]):
            for pb in permutations(partitions[1]):
                sol = [list(pa), list(pb)]
                s, h, c, vans = score(sol)
                if best is None or s > best[0]:
                    best = (s, h, c, vans, sol)

    return {"score": best[0], "hot": best[1], "cold": best[2],
            "vans": best[3], "sol": best[4]}
