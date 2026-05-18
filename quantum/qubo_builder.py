"""
qubo_builder.py — Shared QUBO + QAOA primitives for the Qatalyst Pizza Race.

Each per-stage script (stage0_qaoa.py, stage1_qaoa.py, …) imports from here.

Conventions:
  - Customers numbered 0..N-1, depot index = N.
  - QAOA variable x[i*N + t] = 1 iff customer i is at position t in the route.
  - Distance uses the same /10 scaling as the in-browser game so the
    Python energy matches the game's score after sign-flipping.
  - All QUBO builders return (Q, penalty_strength).
    Q is a dict {(i,j): coefficient} with i <= j.
"""
from __future__ import annotations
import math
import os
import json
from datetime import datetime, timezone
from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import minimize

# Optional load of .env so IONQ_API_KEY is picked up automatically.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================================
# DISTANCE / EVALUATION (mirrors the game's scoring.js)
# ============================================================================
def distance(a: tuple, b: tuple) -> float:
    """Game uses /10 scale for distance throughout."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 10.0


def build_distance_matrix(shop: tuple, customers: list) -> np.ndarray:
    """(N+1) x (N+1) symmetric matrix. Index N = depot."""
    n = len(customers)
    pts = [(c["x"], c["y"]) for c in customers] + [shop]
    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                D[i][j] = distance(pts[i], pts[j])
    return D


def simulate_route_times(route: list, shop: tuple, customers: list) -> list:
    """Walk the route and return arrival time at each stop."""
    arrivals = []
    t = 0.0
    prev = shop
    for ci in route:
        c = customers[ci]
        t += distance(prev, (c["x"], c["y"]))
        arrivals.append(t)
        prev = (c["x"], c["y"])
    return arrivals


def game_score(route: list, shop: tuple, customers: list,
               constraints: bool = False, fuel_tank: float = 999.0) -> dict:
    """Replicates scoring.js evalSolution for a single-van route."""
    arrivals = simulate_route_times(route, shop, customers)
    hot = 0
    cold = 0
    vip_cold = 0
    fuel = 0.0
    prev = shop
    for idx, ci in enumerate(route):
        c = customers[ci]
        fuel += distance(prev, (c["x"], c["y"]))
        if constraints and arrivals[idx] > c.get("hotBy", 1e9):
            cold += 1
            if c.get("vip"):
                vip_cold += 1
        else:
            hot += 1
        prev = (c["x"], c["y"])
    fuel += distance(prev, shop)
    if constraints:
        score = hot * 100 - cold * 50 - vip_cold * 150
        if fuel > fuel_tank:
            score -= (fuel - fuel_tank) * 5
    else:
        score = -round(sum(distance(
            (customers[route[i]]["x"], customers[route[i]]["y"]) if i >= 0 else shop,
            (customers[route[i+1]]["x"], customers[route[i+1]]["y"]) if i+1 < len(route) else shop,
        ) for i in range(-1, len(route))) * 10)
    return {"score": score, "hot": hot, "cold": cold, "vipCold": vip_cold,
            "fuel": fuel, "overFuel": fuel > fuel_tank if constraints else False,
            "arrivals": arrivals}


# ============================================================================
# QUBO HELPERS (single-van one-hot TSP variables)
# ============================================================================
def var_index(i: int, t: int, n: int) -> int:
    """Flatten (customer i, position t) to a flat qubit index."""
    return i * n + t


def add_qubo_term(Q: dict, i: int, j: int, coeff: float):
    """Insert / accumulate a coefficient into the QUBO dict."""
    if abs(coeff) < 1e-15:
        return
    key = (min(i, j), max(i, j))
    Q[key] = Q.get(key, 0.0) + coeff


def assignment_penalty(Q: dict, n: int, penalty: float):
    """Each customer exactly once, each position exactly one customer."""
    # Customer-uniqueness: (sum_t x[i,t] - 1)^2
    for i in range(n):
        for t in range(n):
            add_qubo_term(Q, var_index(i, t, n), var_index(i, t, n), -penalty)
            for tp in range(t + 1, n):
                add_qubo_term(Q, var_index(i, t, n), var_index(i, tp, n), 2 * penalty)
    # Position-uniqueness: (sum_i x[i,t] - 1)^2
    for t in range(n):
        for i in range(n):
            add_qubo_term(Q, var_index(i, t, n), var_index(i, t, n), -penalty)
            for ip in range(i + 1, n):
                add_qubo_term(Q, var_index(i, t, n), var_index(ip, t, n), 2 * penalty)


def distance_cost(Q: dict, D: np.ndarray, n: int):
    """Add TSP distance terms to the QUBO."""
    depot = n
    # Depot -> first
    for i in range(n):
        add_qubo_term(Q, var_index(i, 0, n), var_index(i, 0, n), D[depot][i])
    # Consecutive
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    add_qubo_term(Q, var_index(i, t, n),
                                  var_index(j, t + 1, n), D[i][j])
    # Last -> depot
    for i in range(n):
        add_qubo_term(Q, var_index(i, n - 1, n),
                      var_index(i, n - 1, n), D[i][depot])


def cold_penalty(Q: dict, customers: list, D: np.ndarray, n: int,
                 cold_weight: float):
    """Approximate cold-pizza penalty:
    For each customer i at position t, add `cold_weight` if the *expected*
    cumulative travel time to reach position t exceeds customer i's hotBy.

    Expected time is computed assuming the average inter-stop distance, since
    the QUBO is single-order (each term references at most one variable here).
    This is a linear approximation; we are NOT trying to perfectly recreate
    the game's scoring inside the QUBO. The goal is for QAOA to *prefer*
    routes that respect deadlines.
    """
    depot = n
    # Average distance between distinct points (rough cycle time estimate).
    avg_step = float(D[np.triu_indices(n + 1, 1)].mean())
    for i in range(n):
        deadline = customers[i].get("hotBy", None)
        if deadline is None:
            continue
        for t in range(n):
            # Best-case arrival time at position t: t * shortest_step + depot->i_min
            min_first_leg = min(D[depot][k] for k in range(n))
            best_case = min_first_leg + t * avg_step
            # If even the best case is past the deadline, this assignment is cold.
            if best_case > deadline:
                add_qubo_term(Q, var_index(i, t, n),
                              var_index(i, t, n), cold_weight)


# ============================================================================
# QUBO -> ISING -> QAOA CIRCUIT
# ============================================================================
def qubo_to_ising(Q: dict, num_vars: int) -> tuple:
    """Convert QUBO {(i,j): c} to Ising via x = (1 - z)/2.
    Returns (h, J, offset)."""
    h = {i: 0.0 for i in range(num_vars)}
    J = {}
    offset = 0.0
    for (i, j), c in Q.items():
        if i == j:
            offset += c / 2.0
            h[i] -= c / 2.0
        else:
            offset += c / 4.0
            h[i] -= c / 4.0
            h[j] -= c / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + c / 4.0
    return h, J, offset


def build_qaoa_circuit(h: dict, J: dict, num_vars: int,
                       gamma, beta, p: int = 1):
    """Build a depth-p QAOA circuit (returns a Qiskit QuantumCircuit)."""
    from qiskit import QuantumCircuit
    gammas = [gamma] if p == 1 else list(gamma)
    betas = [beta] if p == 1 else list(beta)

    qc = QuantumCircuit(num_vars, num_vars)
    qc.h(range(num_vars))
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        for i, hi in h.items():
            if abs(hi) > 1e-12:
                qc.rz(2 * g * hi, i)
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * g * Jij, j)
                qc.cx(i, j)
        for i in range(num_vars):
            qc.rx(2 * b, i)
    qc.measure(range(num_vars), range(num_vars))
    return qc


# ============================================================================
# BACKENDS
# ============================================================================
def run_local(circuit, shots: int = 2048):
    from qiskit_aer import AerSimulator
    return AerSimulator().run(circuit, shots=shots).result().get_counts()


def run_ionq(circuit, target: str, shots: int = 1024):
    """target: 'simulator' | 'qpu.aria-1' | 'qpu.forte'"""
    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        raise RuntimeError("pip install qiskit-ionq")
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        raise RuntimeError("IONQ_API_KEY not set (check your .env file)")
    backend = IonQProvider(token=api_key).get_backend(f"ionq_{target}")
    print(f"Submitting to {backend.name}, {shots} shots…")
    job = backend.run(circuit, shots=shots)
    print(f"  job id: {job.job_id()}")
    return job.result().get_counts()


# ============================================================================
# QUBO EVALUATION + PARAMETER SEARCH
# ============================================================================
def qubo_energy(Q: dict, bits) -> float:
    e = 0.0
    for (i, j), c in Q.items():
        if i == j:
            e += c * bits[i]
        else:
            e += c * bits[i] * bits[j]
    return e


def decode_route(bits, n: int):
    """Return list of customer ids in visit order; None if invalid permutation."""
    x = np.array(bits).reshape(n, n)
    if not (all(x[i].sum() == 1 for i in range(n))
            and all(x[:, t].sum() == 1 for t in range(n))):
        return None
    route = []
    for t in range(n):
        for i in range(n):
            if x[i][t] == 1:
                route.append(int(i))
                break
    return route


def expected_energy(counts: dict, Q: dict, num_vars: int) -> float:
    total = sum(counts.values())
    e_avg = 0.0
    for bitstr, n in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        if len(bits) < num_vars:
            bits = bits + [0] * (num_vars - len(bits))
        e_avg += qubo_energy(Q, bits) * n / total
    return e_avg


def optimise_qaoa(Q: dict, num_vars: int, p: int = 1, shots: int = 1024,
                  max_iter: int = 30):
    """COBYLA over (gamma, beta) using local Aer simulator."""
    h, J, offset = qubo_to_ising(Q, num_vars)

    def objective(params):
        if p == 1:
            qc = build_qaoa_circuit(h, J, num_vars, params[0], params[1], p=1)
        else:
            qc = build_qaoa_circuit(h, J, num_vars,
                                    params[:p], params[p:], p=p)
        counts = run_local(qc, shots=shots)
        return expected_energy(counts, Q, num_vars) + offset

    x0 = np.array([0.5] * p + [0.4] * p)
    print(f"Optimising QAOA (p={p}, max_iter={max_iter})…")
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": max_iter, "rhobeg": 0.3})
    print(f"  best <H> = {res.fun:.3f}")
    return res.x


def best_valid_route_from_counts(counts: dict, Q: dict, num_vars: int, n: int):
    total = sum(counts.values())
    valid_count = 0
    best = None
    for bitstr, k in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        if len(bits) < num_vars:
            bits = bits + [0] * (num_vars - len(bits))
        route = decode_route(bits[:num_vars], n)
        if route is None:
            continue
        valid_count += k
        e = qubo_energy(Q, bits[:num_vars])
        if best is None or e < best[1]:
            best = (route, e, k, bitstr)
    valid_frac = valid_count / total
    if best is None:
        return None, None, 0, total, valid_frac
    return best[0], best[1], best[2], total, valid_frac


def brute_force_qubo_optimum(Q: dict, num_vars: int, n: int):
    """For small problems, enumerate to verify QAOA isn't doing something weird."""
    best_e = float("inf")
    best_route = None
    for bits in product([0, 1], repeat=num_vars):
        route = decode_route(bits, n)
        if route is None:
            continue
        e = qubo_energy(Q, bits)
        if e < best_e:
            best_e = e
            best_route = route
    return best_route, best_e


# ============================================================================
# CACHE WRITER
# ============================================================================
def write_cache(out_path: str, *, stage_id: int, route: list,
                customers: list, backend: str, energy: float,
                valid_fraction: float, total_shots: int,
                num_qubits: int, p: int, qubo_size: int,
                extra: dict = None):
    """Standardised JSON cache the game can consume."""
    payload = {
        "stage_id": int(stage_id),
        "route": [int(x) for x in route],
        "backend": backend,
        "energy": float(energy),
        "valid_fraction": float(valid_fraction),
        "total_shots": int(total_shots),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_qubits": int(num_qubits),
        "qubo_size_terms": int(qubo_size),
        "encoding": f"one-hot TSP, N*N variables, p={p} QAOA",
        "customer_names": [c["name"] for c in customers],
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Cache written: {out_path}")
    return payload
