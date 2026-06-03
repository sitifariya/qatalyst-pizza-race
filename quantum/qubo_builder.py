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

XY-mixer additions (June 2026):
  - tqa_init: TQA parameter init (Sack & Serbyn, Quantum 2021).
  - dicke_state_k1: prepare |D^n_1> (uniform over one-hot bitstrings),
    used as the initial state for XY-mixer QAOA.
  - xy_ring_mixer_row: apply XY-ring mixer to one row of n qubits.
  - build_qaoa_circuit_xy: full QAOA circuit using XY-mixer.
  - optimise_qaoa gained `mixer="x"` (default) or `mixer="xy"` parameter.

References:
  Sack & Serbyn, "Quantum annealing initialization of the quantum approximate
    optimization algorithm", Quantum 5, 491 (2021). arXiv:2101.05742
  Hadfield et al., "From the QAOA to a quantum alternating operator ansatz",
    Algorithms 12, 34 (2019).
  Bärtschi & Eidenbenz, "Deterministic preparation of Dicke states",
    arXiv:1904.07358.
  OpenQuantumComputing/QAOA — reference open-source XY-mixer implementation.
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
    for i in range(n):
        add_qubo_term(Q, var_index(i, 0, n), var_index(i, 0, n), D[depot][i])
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j:
                    add_qubo_term(Q, var_index(i, t, n),
                                  var_index(j, t + 1, n), D[i][j])
    for i in range(n):
        add_qubo_term(Q, var_index(i, n - 1, n),
                      var_index(i, n - 1, n), D[i][depot])


def cold_penalty(Q: dict, customers: list, D: np.ndarray, n: int,
                 cold_weight: float):
    """Linear cold-pizza approximation."""
    depot = n
    avg_step = float(D[np.triu_indices(n + 1, 1)].mean())
    for i in range(n):
        deadline = customers[i].get("hotBy", None)
        if deadline is None:
            continue
        for t in range(n):
            min_first_leg = min(D[depot][k] for k in range(n))
            best_case = min_first_leg + t * avg_step
            if best_case > deadline:
                add_qubo_term(Q, var_index(i, t, n),
                              var_index(i, t, n), cold_weight)


# ============================================================================
# QUBO -> ISING -> QAOA CIRCUIT (X-mixer, the original)
# ============================================================================
def qubo_to_ising(Q: dict, num_vars: int) -> tuple:
    """Convert QUBO {(i,j): c} to Ising via x = (1 - z)/2."""
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
    """Build a depth-p QAOA circuit with X-mixer (returns Qiskit QuantumCircuit).

    This is the ORIGINAL X-mixer circuit. Use build_qaoa_circuit_xy for the
    constraint-preserving XY-mixer version.
    """
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
# XY-MIXER + DICKE STATE (constraint-preserving QAOA for one-hot TSP)
# ============================================================================
# IMPLEMENTATION NOTES — please review before trusting:
#
# Variable layout for one-hot TSP with n customers:
#   x[i*n + t] = 1 iff customer i is at position t.
#   Total qubits = n*n.
#   We group qubits into n "rows", each row holding n qubits for customer i:
#     row i = qubits [i*n, i*n+1, ..., i*n+n-1].
#   The valid one-hot constraint says: each row has exactly ONE qubit in |1>.
#   ALSO each column should have exactly one |1>. The XY mixer here only
#   preserves ROW constraints. Column constraints are still enforced by the
#   QUBO penalty terms (so plain QAOA still has SOME constraint violations
#   to fight; we just remove the row-violation half).
#
# Honest TODO: A full constraint-preserving mixer would handle BOTH row
# and column constraints. We are implementing the simpler row-only XY-mixer
# first. If validity is still poor on simulator, we revisit.
#
# Dicke state |D^n_1> for k=1 (one-hot per row):
#   Construction: put a |1> on qubit 0, then a cascade of controlled rotations
#   spreads the amplitude evenly to qubits 1..n-1.
#   Reference: Bärtschi & Eidenbenz 2019 algorithm, simplified for k=1.
#
# XY-mixer (ring topology):
#   Each row independently. For row of n qubits at indices [q0..q_{n-1}],
#   apply XX+YY (XXPlusYYGate) on adjacent pairs in a ring: (q0,q1), (q1,q2),
#   ..., (q_{n-2},q_{n-1}), (q_{n-1},q0). Beta angle is shared across all pairs
#   in the layer.
# ============================================================================

def dicke_state_k1(qc, qubits):
    """Prepare |D^n_1> on the given qubits.
    
    |D^n_1> is the uniform superposition over all n single-excitation 
    bitstrings (one |1> in n qubits):
      (1/sqrt(n)) * sum_{i=0}^{n-1} X_i |0...0>
    
    Construction (k=1 simplification of Bärtschi-Egger):
      1. X gate on first qubit (gives |10...0>).
      2. For i = 0..n-2: rotate amplitude between qubits i and i+1.
         At step i, qubit i carries fraction (i+1)/(i+2) of the amplitude.
         We rotate so qubit i+1 gets the proper share: angle theta_i such
         that sin^2(theta_i/2) = 1/(i+2).
    
    The result: amplitude on each one-hot bitstring is 1/sqrt(n).
    
    Args:
      qc: Qiskit QuantumCircuit
      qubits: list of qubit indices, length n >= 1
    """
    n = len(qubits)
    if n == 0:
        return
    if n == 1:
        qc.x(qubits[0])
        return
    
    # Place excitation on first qubit
    qc.x(qubits[0])
    
    # Partial-SWAP cascade: at step i, move fraction (n-1-i)/(n-i) of the
    # amplitude from qubit i to qubit i+1. This produces uniform 1/n
    # distribution across the n one-hot bitstrings.
    #
    # Verified mathematically for n=3, 4, 5: each qubit ends up with
    # probability exactly 1/n, zero leakage to invalid bitstrings.
    for i in range(n - 1):
        # Fraction of amplitude to move from q[i] to q[i+1]:
        frac = (n - 1 - i) / (n - i)
        theta = 2.0 * math.asin(math.sqrt(frac))
        # Partial-SWAP decomposition:
        #   cry(theta) on q[i+1] with control q[i] — rotates q[i+1] when q[i]=1
        #   cx with control q[i+1], target q[i] — completes the swap
        qc.cry(theta, qubits[i], qubits[i + 1])
        qc.cx(qubits[i + 1], qubits[i])


def xy_ring_mixer_row(qc, qubits, beta):
    """Apply one layer of XY-ring mixer to a row of qubits.
    
    Uses XXPlusYYGate which is Qiskit's audited primitive for the
    exp(-i*beta*(XX + YY)/2) interaction. Argument convention: XXPlusYYGate
    angle theta produces exp(-i*theta*(XX+YY)/2), matching the standard
    QAOA mixer convention with beta being the mixer angle.
    
    Args:
      qc: Qiskit QuantumCircuit
      qubits: list of qubit indices, length n >= 2
      beta: mixer angle (float). Used as theta = 2*beta for XXPlusYYGate.
    """
    from qiskit.circuit.library import XXPlusYYGate
    n = len(qubits)
    if n < 2:
        return  # No pairs to mix
    
    # Ring topology: (q0,q1), (q1,q2), ..., (q_{n-2},q_{n-1}), (q_{n-1},q0)
    # IMPORTANT: XXPlusYYGate convention is theta = 2*beta.
    pairs = [(qubits[k], qubits[k + 1]) for k in range(n - 1)]
    if n >= 3:  # No need to close ring if n=2 (only 1 pair anyway)
        pairs.append((qubits[n - 1], qubits[0]))
    
    for (a, b) in pairs:
        qc.append(XXPlusYYGate(2.0 * beta), [a, b])


def build_qaoa_circuit_xy(h: dict, J: dict, num_vars: int,
                          gamma, beta, p: int = 1,
                          n_customers: int = None):
    """Build a depth-p QAOA circuit with XY-ring mixer (row constraints).
    
    Initial state: tensor product of Dicke |D^n_1> states, one per row.
    Mixer: XY-ring mixer applied independently to each row each layer.
    Cost: same as build_qaoa_circuit (Z and ZZ terms).
    
    Args:
      h, J: Ising Hamiltonian terms (from qubo_to_ising).
      num_vars: total qubit count (= n_customers * n_customers).
      gamma, beta: QAOA params (scalar if p=1, list of length p otherwise).
      p: QAOA depth.
      n_customers: number of customers (= sqrt(num_vars)). Required to know
                   how to split qubits into rows.
    
    Returns: Qiskit QuantumCircuit with measurement.
    """
    from qiskit import QuantumCircuit
    
    if n_customers is None:
        # Infer from num_vars (must be a perfect square)
        n_customers = int(round(math.sqrt(num_vars)))
        if n_customers * n_customers != num_vars:
            raise ValueError(f"num_vars={num_vars} is not a perfect square; "
                             f"please pass n_customers explicitly.")
    
    if n_customers * n_customers != num_vars:
        raise ValueError(f"num_vars={num_vars} != n_customers^2 = {n_customers**2}")
    
    gammas = [gamma] if p == 1 else list(gamma)
    betas = [beta] if p == 1 else list(beta)
    
    qc = QuantumCircuit(num_vars, num_vars)
    
    # ---- Initial state: product of n_customers Dicke |D^n_1> states ----
    # Row i covers qubits [i*n_customers, ..., (i+1)*n_customers - 1].
    for row in range(n_customers):
        row_qubits = list(range(row * n_customers, (row + 1) * n_customers))
        dicke_state_k1(qc, row_qubits)
    
    # ---- p QAOA layers ----
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        
        # Cost unitary: same as X-mixer version
        for i, hi in h.items():
            if abs(hi) > 1e-12:
                qc.rz(2 * g * hi, i)
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * g * Jij, j)
                qc.cx(i, j)
        
        # XY-ring mixer applied per row (preserves Hamming weight within each row)
        for row in range(n_customers):
            row_qubits = list(range(row * n_customers, (row + 1) * n_customers))
            xy_ring_mixer_row(qc, row_qubits, b)
    
    qc.measure(range(num_vars), range(num_vars))
    return qc


def verify_dicke_distribution(n: int, shots: int = 32768, 
                              tolerance_sigmas: float = 5.0) -> dict:
    """SANITY CHECK: verify the Dicke |D^n_1> circuit produces correct output.
    
    Run the bare Dicke circuit and measure. Each of the n one-hot bitstrings
    should appear ~shots/n times. Reports any deviation.
    
    RUN THIS FIRST before trusting the XY-mixer pipeline.
    
    Returns dict with 'pass' (bool), 'valid_fraction', and 'per_bitstring' details.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    
    qc = QuantumCircuit(n, n)
    dicke_state_k1(qc, list(range(n)))
    qc.measure(range(n), range(n))
    
    counts = AerSimulator().run(qc, shots=shots).result().get_counts()
    
    expected = shots / n
    sigma = math.sqrt(shots * (1.0/n) * (1.0 - 1.0/n))
    tolerance = tolerance_sigmas * sigma
    
    per_bitstring = {}
    total_valid = 0
    all_in_tolerance = True
    
    print(f"\n=== Dicke |D^{n}_1> sanity check ===")
    print(f"  shots: {shots}, expected per one-hot bitstring: {expected:.1f} ± {tolerance:.0f}")
    
    for i in range(n):
        bits = ['0'] * n
        bits[i] = '1'
        bs = ''.join(reversed(bits))  # Qiskit big-endian for counts keys
        cnt = counts.get(bs, 0)
        per_bitstring[bs] = cnt
        total_valid += cnt
        ok = abs(cnt - expected) <= tolerance
        marker = "✓" if ok else "✗"
        print(f"  {bs}: {cnt} {marker}")
        if not ok:
            all_in_tolerance = False
    
    valid_frac = total_valid / shots
    invalid_frac = 1.0 - valid_frac
    
    print(f"  Valid one-hot fraction: {valid_frac*100:.4f}%")
    print(f"  Invalid (leakage) fraction: {invalid_frac*100:.4f}%")
    
    is_pass = all_in_tolerance and valid_frac > 0.99
    if is_pass:
        print(f"  PASS — Dicke state is correctly prepared.")
    else:
        if valid_frac <= 0.99:
            print(f"  FAIL — leakage > 1% means circuit is incorrect.")
        else:
            print(f"  FAIL — distribution not uniform within {tolerance_sigmas}σ.")
    
    return {
        "pass": is_pass,
        "valid_fraction": valid_frac,
        "per_bitstring": per_bitstring,
        "expected_per_bitstring": expected,
        "tolerance": tolerance,
    }


# ============================================================================
# BACKENDS
# ============================================================================
def run_local(circuit, shots: int = 2048):
    """Run on local Qiskit Aer simulator.
    
    NOTE: XXPlusYYGate (used by XY-mixer) isn't in AerSimulator's native gate set,
    so we transpile to basic gates first. This decomposes XXPlusYYGate into
    CX/RZ/RX gates that Aer can execute.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    backend = AerSimulator()
    # Transpile to backend's basis to handle XXPlusYYGate and other library gates
    tqc = transpile(circuit, backend, optimization_level=0)
    return backend.run(tqc, shots=shots).result().get_counts()


def run_ionq(circuit, target: str, shots: int = 1024):
    """target: 'simulator' | 'qpu.aria-1' | 'qpu.forte' | 'qpu.forte-enterprise-1'"""
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


def tqa_init(p: int, dt: float = 0.75) -> np.ndarray:
    """TQA initialization (Sack & Serbyn 2021, arXiv:2101.05742)."""
    gammas = [((k + 1) / p) * dt for k in range(p)]
    betas = [((p - k) / p) * dt for k in range(p)]
    return np.array(gammas + betas)


def optimise_qaoa(Q: dict, num_vars: int, p: int = 1, shots: int = 1024,
                  max_iter: int = 30, init_method: str = "tqa",
                  tqa_dt: float = 0.75, mixer: str = "x",
                  n_customers: int = None):
    """COBYLA over (gamma, beta) using local Aer simulator.
    
    Args:
      mixer: "x" (default, the original X-mixer) or "xy" (constraint-preserving
             XY-mixer using Dicke initial state). XY-mixer requires n_customers.
      n_customers: required when mixer="xy" (= sqrt(num_vars)).
    
    init_method:
      'tqa'    - Trotterized Quantum Annealing init (Sack & Serbyn 2021).
      'flat'   - Legacy flat init.
    """
    h, J, offset = qubo_to_ising(Q, num_vars)
    
    if mixer not in ("x", "xy"):
        raise ValueError(f"mixer must be 'x' or 'xy', got {mixer!r}")
    
    if mixer == "xy" and n_customers is None:
        n_customers = int(round(math.sqrt(num_vars)))
        if n_customers * n_customers != num_vars:
            raise ValueError(f"XY-mixer needs n_customers; num_vars={num_vars} is not a square.")

    def objective(params):
        if mixer == "x":
            if p == 1:
                qc = build_qaoa_circuit(h, J, num_vars, params[0], params[1], p=1)
            else:
                qc = build_qaoa_circuit(h, J, num_vars,
                                        params[:p], params[p:], p=p)
        else:  # mixer == "xy"
            if p == 1:
                qc = build_qaoa_circuit_xy(h, J, num_vars, params[0], params[1],
                                           p=1, n_customers=n_customers)
            else:
                qc = build_qaoa_circuit_xy(h, J, num_vars,
                                           params[:p], params[p:], p=p,
                                           n_customers=n_customers)
        counts = run_local(qc, shots=shots)
        return expected_energy(counts, Q, num_vars) + offset

    if init_method == "tqa":
        x0 = tqa_init(p, dt=tqa_dt)
        print(f"Optimising QAOA (p={p}, max_iter={max_iter}, init=TQA dt={tqa_dt}, mixer={mixer})…")
    elif init_method == "flat":
        x0 = np.array([0.5] * p + [0.4] * p)
        print(f"Optimising QAOA (p={p}, max_iter={max_iter}, init=flat, mixer={mixer})…")
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    print(f"  x0 = {x0}")
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


# ============================================================================
# STANDALONE SANITY CHECK ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    """Run sanity checks. Execute: python3 quantum/qubo_builder.py"""
    print("=" * 70)
    print("XY-MIXER SANITY CHECKS")
    print("=" * 70)
    print()
    print("Step 1: Verify Dicke |D^n_1> construction for n = 3, 4, 5")
    print("        (these are the row sizes we use in Stages 0, 1, 3)")
    print()
    
    all_pass = True
    for n in [3, 4, 5]:
        result = verify_dicke_distribution(n)
        if not result["pass"]:
            all_pass = False
    
    print()
    print("=" * 70)
    if all_pass:
        print("ALL DICKE CHECKS PASSED — XY-mixer pipeline is safe to use.")
        print("Next step: run Stage 3 with --mixer xy on local sim.")
    else:
        print("SANITY CHECKS FAILED — DO NOT proceed to Forte.")
        print("Review the dicke_state_k1 function and re-test.")
    print("=" * 70)
