from dotenv import load_dotenv
load_dotenv()
"""
Qatalyst Pizza Race - Stage 0 QAOA pipeline for IonQ hardware.

This script formulates Stage 0 (3 customers, 1 van, no constraints) as a QUBO,
builds a QAOA circuit, and runs it on three backends:
  1. local Aer simulator           (free, instant)
  2. IonQ ideal simulator          (free, queue)
  3. IonQ Forte hardware           (costs credits, queue, may be in maintenance)
  4. IonQ Forte Enterprise 1       (costs credits, queue, current production)

The output is a JSON cache file consumed by the game's stages.js (replacing the
hardcoded cachedQuantum field for Stage 0 with a real hardware result).

Usage:
    python stage0_qaoa.py --backend local             # quick local test
    python stage0_qaoa.py --backend ionq_sim          # free IonQ cloud simulator
    python stage0_qaoa.py --backend ionq_forte        # regular Forte (in maintenance until Jun 22, 2026)
    python stage0_qaoa.py --backend ionq_forte_ent    # Forte Enterprise 1 (CURRENT)
    python stage0_qaoa.py --backend ionq_aria         # Aria (older, cheaper)

Requires IONQ_API_KEY env var for all ionq_* backends.
"""

import os
import json
import math
import argparse
from datetime import datetime, timezone
from itertools import product

import numpy as np

# Qiskit + IonQ provider
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize


# ============================================================================
# STAGE 0 PROBLEM DATA
# ============================================================================
# Mirrors stages.js Stage 0 exactly. Update here if the game's Stage 0 changes.
SHOP = (350, 200)
CUSTOMERS = [
    {"id": 0, "name": "Ria",  "x": 160, "y": 120},
    {"id": 1, "name": "Adam", "x": 560, "y": 160},
    {"id": 2, "name": "Luna", "x": 340, "y": 310},
]
N = len(CUSTOMERS)  # 3 customers => N*N = 9 qubits


# ============================================================================
# QUBO BUILDER
# ============================================================================
def distance(a, b):
    """Game uses /10 scale for distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 10.0


def build_distance_matrix():
    """Returns (N+1)x(N+1) matrix. Index N = depot."""
    pts = [(c["x"], c["y"]) for c in CUSTOMERS] + [SHOP]
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i][j] = distance(pts[i], pts[j])
    return D


def build_qubo(penalty_strength=None):
    """Build the TSP QUBO as a dict {(i,j): coefficient}.

    Variables: x[i*N + t] = 1 if customer i is at position t.
    Total qubits: N*N = 9.

    Penalty must exceed max distance to ensure constraints win.
    """
    D = build_distance_matrix()
    if penalty_strength is None:
        penalty_strength = float(D.max()) * 2.0  # safe margin

    def v(i, t):
        return i * N + t

    Q = {}

    def add(i, j, coeff):
        key = (min(i, j), max(i, j))
        Q[key] = Q.get(key, 0.0) + coeff

    # ---- Penalty A: each customer visited exactly once ----
    # (sum_t x[i,t] - 1)^2 = sum_t x[i,t]^2 + 2*sum_{t<t'} x[i,t]*x[i,t'] - 2*sum_t x[i,t] + 1
    # x^2 = x for binary, so x[i,t]^2 contributes -1 net after the -2*x term
    # → diagonal: -1 per var. Off-diagonal: +2 between same-customer positions.
    for i in range(N):
        for t in range(N):
            add(v(i, t), v(i, t), -penalty_strength)  # the -2*x + x^2 → -x
            for tp in range(t + 1, N):
                add(v(i, t), v(i, tp), 2 * penalty_strength)

    # ---- Penalty B: each position has exactly one customer ----
    for t in range(N):
        for i in range(N):
            add(v(i, t), v(i, t), -penalty_strength)
            for ip in range(i + 1, N):
                add(v(i, t), v(ip, t), 2 * penalty_strength)

    # The constant term from (.. - 1)^2 expansion is +1 per constraint;
    # 2N constraints contribute +2N. We omit constants (they don't affect optimisation).

    # ---- Distance cost ----
    # Depot -> first customer at position 0
    for i in range(N):
        add(v(i, 0), v(i, 0), D[N][i])
    # Consecutive customers
    for t in range(N - 1):
        for i in range(N):
            for j in range(N):
                if i != j:
                    add(v(i, t), v(j, t + 1), D[i][j])
    # Last customer at position N-1 -> depot
    for i in range(N):
        add(v(i, N - 1), v(i, N - 1), D[i][N])

    return Q, penalty_strength


def qubo_energy(Q, bitstring):
    """Evaluate QUBO energy for a bitstring (list/tuple of 0s and 1s)."""
    e = 0.0
    for (i, j), c in Q.items():
        if i == j:
            e += c * bitstring[i]
        else:
            e += c * bitstring[i] * bitstring[j]
    return e


def decode_route(bitstring):
    """Bitstring -> list of customer ids in visit order. Returns None if invalid."""
    x = np.array(bitstring).reshape(N, N)
    # valid permutation check
    if not (all(x[i].sum() == 1 for i in range(N)) and all(x[:, t].sum() == 1 for t in range(N))):
        return None
    route = []
    for t in range(N):
        for i in range(N):
            if x[i][t] == 1:
                route.append(int(i))
                break
    return route


# ============================================================================
# QUBO -> ISING -> QAOA CIRCUIT
# ============================================================================
def qubo_to_ising(Q, num_vars):
    """Convert QUBO {(i,j): c} to Ising (h, J, offset) via x = (1 - z)/2.

    Returns:
      h: dict {i: coefficient on Z_i}
      J: dict {(i,j): coefficient on Z_i Z_j}, i < j
      offset: scalar
    """
    h = {i: 0.0 for i in range(num_vars)}
    J = {}
    offset = 0.0
    for (i, j), c in Q.items():
        if i == j:
            # c * x_i = c * (1 - z_i)/2 = c/2 - c/2 * z_i
            offset += c / 2.0
            h[i] -= c / 2.0
        else:
            # c * x_i x_j = c * (1 - z_i)(1 - z_j)/4
            #             = c/4 (1 - z_i - z_j + z_i z_j)
            offset += c / 4.0
            h[i] -= c / 4.0
            h[j] -= c / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + c / 4.0
    return h, J, offset


def build_qaoa_circuit(h, J, num_vars, gamma, beta, p=1):
    """Build a depth-p QAOA circuit.

    For p > 1, pass gamma and beta as lists of length p.
    """
    if p == 1:
        gammas = [gamma]
        betas = [beta]
    else:
        gammas = list(gamma)
        betas = list(beta)

    qc = QuantumCircuit(num_vars, num_vars)

    # Initial state: |+>^n
    qc.h(range(num_vars))

    for layer in range(p):
        g = gammas[layer]
        b = betas[layer]

        # Problem unitary e^(-i gamma * H_C)
        # Single Z terms
        for i, hi in h.items():
            if abs(hi) > 1e-12:
                qc.rz(2 * g * hi, i)
        # ZZ terms (CNOT-RZ-CNOT)
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * g * Jij, j)
                qc.cx(i, j)

        # Mixer unitary e^(-i beta * H_M), H_M = sum X_i
        for i in range(num_vars):
            qc.rx(2 * b, i)

    qc.measure(range(num_vars), range(num_vars))
    return qc


# ============================================================================
# BACKEND RUNNERS
# ============================================================================
def run_local(circuit, shots=2048):
    """Run on local Qiskit Aer simulator."""
    backend = AerSimulator()
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def run_ionq(circuit, backend_name, shots=1024):
    """Submit to IonQ cloud (simulator or hardware).

    backend_name:
      'simulator'              - free, ideal simulator
      'qpu.forte'              - real Forte hardware (in maintenance until Jun 22, 2026)
      'qpu.forte-enterprise-1' - real Forte Enterprise 1 hardware (CURRENT production)
      'qpu.aria-1'             - real Aria hardware (older, cheaper)
    """
    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        raise RuntimeError(
            "Install qiskit-ionq: pip install qiskit-ionq"
        )
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set IONQ_API_KEY environment variable")

    provider = IonQProvider(token=api_key)
    backend = provider.get_backend(f"ionq_{backend_name}")
    print(f"Submitting to {backend.name}, {shots} shots... (queue may take minutes)")
    job = backend.run(circuit, shots=shots)
    print(f"Job ID: {job.job_id()}")
    result = job.result()
    counts = result.get_counts()
    return counts


# ============================================================================
# QAOA OPTIMISATION LOOP
# ============================================================================
def expected_energy(counts, Q, num_vars):
    """Average QUBO energy weighted by measurement counts.

    Bitstrings from Qiskit are big-endian: bit 0 is rightmost char.
    """
    total = sum(counts.values())
    e_avg = 0.0
    for bitstr, n in counts.items():
        bits = [int(b) for b in bitstr[::-1]]  # reverse to little-endian
        if len(bits) < num_vars:
            bits = bits + [0] * (num_vars - len(bits))
        e_avg += qubo_energy(Q, bits) * n / total
    return e_avg


def optimise_qaoa_local(Q, num_vars, p=1, shots=1024, max_iter=30):
    """Run COBYLA over QAOA parameters using local simulator.

    Returns the best (gamma, beta) and the counts from the best run.
    """
    h, J, offset = qubo_to_ising(Q, num_vars)

    def objective(params):
        if p == 1:
            qc = build_qaoa_circuit(h, J, num_vars, params[0], params[1], p=1)
        else:
            gammas = params[:p]
            betas = params[p:]
            qc = build_qaoa_circuit(h, J, num_vars, gammas, betas, p=p)
        counts = run_local(qc, shots=shots)
        e = expected_energy(counts, Q, num_vars) + offset
        return e

    # Initial guess: small angles
    x0 = np.array([0.5] * p + [0.4] * p)
    print(f"\nOptimising QAOA (p={p}, max_iter={max_iter})...")
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": max_iter, "rhobeg": 0.3})
    print(f"Optimisation finished. Best <H> = {res.fun:.3f}")
    return res.x


def best_route_from_counts(counts, Q, num_vars):
    """From measurement counts, find the bitstring with lowest QUBO energy
    that is also a valid permutation.

    Returns (route, energy, count, total_shots, valid_fraction)
    """
    total = sum(counts.values())
    best_valid = None  # (route, energy, count, bitstring)
    valid_count = 0
    for bitstr, n in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        if len(bits) < num_vars:
            bits = bits + [0] * (num_vars - len(bits))
        route = decode_route(bits[:num_vars])
        if route is not None:
            valid_count += n
            e = qubo_energy(Q, bits[:num_vars])
            if best_valid is None or e < best_valid[1]:
                best_valid = (route, e, n, bitstr)
    valid_fraction = valid_count / total
    if best_valid is None:
        return None, None, 0, total, valid_fraction
    return best_valid[0], best_valid[1], best_valid[2], total, valid_fraction


# ============================================================================
# CACHE WRITER (matches stages.js cachedQuantum format)
# ============================================================================
def write_cache(route, backend, energy, valid_fraction, total_shots, out_path):
    """Write a JSON file the game can consume.

    Format mirrors what cachedQuantum needs in stages.js (single-van case):
      cachedQuantum: [0, 2, 1]   (a single array of customer ids)
    """
    payload = {
        "stage_id": 0,
        "route": [int(x) for x in route],
        "backend": backend,
        "energy": float(energy),
        "valid_fraction": float(valid_fraction),
        "total_shots": int(total_shots),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_qubits": N * N,
        "encoding": "TSP one-hot, N*N variables, p=1 QAOA",
        "customer_names": [c["name"] for c in CUSTOMERS],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nCache written: {out_path}")
    return payload


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["local", "ionq_sim", "ionq_forte", "ionq_forte_ent", "ionq_aria"],
        default="local",
        help="Where to run the final QAOA circuit. "
             "ionq_forte_ent = Forte Enterprise 1 (current production hardware).",
    )
    parser.add_argument("--p", type=int, default=1, help="QAOA depth (default 1)")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=30,
                        help="Max COBYLA iterations during local parameter search")
    parser.add_argument("--out", default="stage0_results/stage0_qaoa_result.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Qatalyst Pizza Race — Stage 0 QAOA on {args.backend}")
    print("=" * 70)

    # Step 1: build the QUBO
    Q, penalty = build_qubo()
    print(f"QUBO built. {N*N} variables, penalty strength = {penalty:.2f}")

    # Step 2: brute-force the QUBO to know what the answer should be
    print("\nBrute-force enumeration (sanity check):")
    best_bits, best_e = None, float("inf")
    for bits in product([0, 1], repeat=N * N):
        e = qubo_energy(Q, bits)
        if e < best_e:
            best_e, best_bits = e, bits
    bf_route = decode_route(best_bits)
    print(f"  Optimal energy: {best_e:.3f}, route: {bf_route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in bf_route)})")

    # Step 3: optimise QAOA parameters using local simulator
    # (Even when running on IonQ for the final shot, we tune params locally to save credits.)
    best_params = optimise_qaoa_local(Q, N * N, p=args.p,
                                      shots=args.shots, max_iter=args.max_iter)
    print(f"Best QAOA params: {best_params}")

    # Step 4: build final circuit with optimised parameters
    h, J, _ = qubo_to_ising(Q, N * N)
    if args.p == 1:
        final_qc = build_qaoa_circuit(h, J, N * N,
                                      best_params[0], best_params[1], p=1)
    else:
        gammas = best_params[:args.p]
        betas = best_params[args.p:]
        final_qc = build_qaoa_circuit(h, J, N * N, gammas, betas, p=args.p)

    print(f"\nFinal circuit: {final_qc.num_qubits} qubits, depth {final_qc.depth()}, "
          f"{sum(final_qc.count_ops().values())} gates")

    # Step 5: run on chosen backend
    print(f"\nRunning on backend: {args.backend} ({args.shots} shots)...")
    if args.backend == "local":
        counts = run_local(final_qc, shots=args.shots)
        backend_label = "qiskit_aer_local"
    elif args.backend == "ionq_sim":
        counts = run_ionq(final_qc, "simulator", shots=args.shots)
        backend_label = "ionq_simulator"
    elif args.backend == "ionq_forte":
        counts = run_ionq(final_qc, "qpu.forte", shots=args.shots)
        backend_label = "ionq_forte"
    elif args.backend == "ionq_forte_ent":
        counts = run_ionq(final_qc, "qpu.forte-enterprise-1", shots=args.shots)
        backend_label = "ionq_forte_enterprise_1"
    elif args.backend == "ionq_aria":
        counts = run_ionq(final_qc, "qpu.aria-1", shots=args.shots)
        backend_label = "ionq_aria"

    # Step 6: decode best valid route
    route, energy, route_count, total, valid_frac = best_route_from_counts(
        counts, Q, N * N
    )

    print(f"\n--- RESULTS ---")
    print(f"Valid permutations: {valid_frac*100:.1f}% of shots")
    if route is None:
        print("WARNING: no valid permutation found in the measurement samples.")
        print("Try: increase shots, increase QAOA depth p, or increase max-iter.")
        return
    print(f"Best valid route: {route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in route)})")
    print(f"  energy: {energy:.3f}")
    print(f"  measured {route_count}/{total} times "
          f"({route_count/total*100:.1f}% of shots)")
    print(f"Brute-force optimum was: {bf_route}, energy {best_e:.3f}")
    if energy <= best_e + 1e-6:
        print("✓ QAOA found the optimal route.")
    else:
        ratio = energy / best_e if best_e > 0 else float("inf")
        print(f"  Approximation ratio: {ratio:.3f}")

    # Step 7: write cache for the game
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_cache(route, backend_label, energy, valid_frac, total, args.out)


if __name__ == "__main__":
    main()
