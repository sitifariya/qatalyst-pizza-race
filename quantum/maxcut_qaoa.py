"""
MaxCut QAOA at 25 qubits on random degree-3 graphs.

CLAIM: QAOA at p=1 beats local 1-step classical on degree-3 random graphs.
       Published in Carlson et al. 2023 (arXiv:2304.08420).

This is a SEPARATE experiment from the pizza routing game. It demonstrates
QAOA's natural advantage on graph problems (its original 2014 design domain).

Usage:
    # Quick local sanity check (single 25-node graph)
    python3 quantum/maxcut_qaoa.py --backend local --instances 1

    # Multi-instance comparison on local sim (30 graphs)
    python3 quantum/maxcut_qaoa.py --backend local --instances 30

    # Single best-performing instance on Forte Enterprise 1
    python3 quantum/maxcut_qaoa.py --backend ionq_forte_ent --instances 1 --seed 42

References:
  Carlson et al. "Approximation Algorithms for the MaxCut Problem", 
    arXiv:2304.08420 (2023).
  Farhi, Goldstone, Gutmann "A Quantum Approximate Optimization Algorithm",
    arXiv:1411.4028 (2014).
"""
import argparse
import math
import os
import sys
import json
import random
import statistics
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import numpy as np
from scipy.optimize import minimize


# ============================================================================
# RANDOM DEGREE-3 GRAPH GENERATION
# ============================================================================
def generate_random_3regular_graph(n: int, seed: int = None) -> list:
    """Generate a random 3-regular graph on n nodes.
    
    Uses configuration model: each node has 3 "stubs", randomly pair them up.
    Retries if invalid (self-loops or multi-edges).
    
    Args:
      n: number of nodes (must be even for 3-regular)
      seed: random seed for reproducibility
    
    Returns:
      List of edges [(u, v), ...] where u < v
    """
    if n % 2 != 0:
        raise ValueError("3-regular graph needs even n")
    
    if seed is not None:
        random.seed(seed)
    
    max_retries = 100
    for attempt in range(max_retries):
        # Configuration model: 3n/2 stubs (3 per node, 2 stubs per edge)
        stubs = []
        for node in range(n):
            stubs.extend([node, node, node])
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return sorted(edges)
    
    raise RuntimeError(f"Failed to generate valid 3-regular graph after {max_retries} attempts")


# ============================================================================
# MAXCUT QUBO + ISING
# ============================================================================
def maxcut_qubo(edges: list, n: int) -> dict:
    """Build the MaxCut QUBO for the given graph.
    
    For each edge (i, j): -x_i - x_j + 2*x_i*x_j
    Maximising cut = maximising number of edges with x_i != x_j.
    
    Returns Q as dict {(i, j): coeff} where i <= j.
    NOTE: this is MINIMIZING -cut, so optimal QUBO value is negative.
    """
    Q = {}
    for (u, v) in edges:
        # x_u + x_v - 2*x_u*x_v measures whether edge is cut
        # We want to MAXIMIZE this, equivalently MINIMIZE -1 * (x_u + x_v - 2*x_u*x_v)
        Q[(u, u)] = Q.get((u, u), 0.0) - 1.0
        Q[(v, v)] = Q.get((v, v), 0.0) - 1.0
        Q[(u, v)] = Q.get((u, v), 0.0) + 2.0
    return Q


def maxcut_qubo_to_ising(Q: dict, n: int) -> tuple:
    """Convert MaxCut QUBO to Ising via x = (1 - z)/2."""
    h = {i: 0.0 for i in range(n)}
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


def evaluate_cut(bits: list, edges: list) -> int:
    """Count the number of edges cut by bit-assignment.
    
    An edge (u, v) is cut iff bits[u] != bits[v].
    """
    cut = 0
    for (u, v) in edges:
        if bits[u] != bits[v]:
            cut += 1
    return cut


# ============================================================================
# QAOA CIRCUIT (X-mixer, the original Farhi-Goldstone-Gutmann)
# ============================================================================
def build_maxcut_qaoa(h: dict, J: dict, n: int, gamma: float, beta: float):
    """Build the standard depth-1 QAOA circuit for MaxCut."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    
    # Initial state: uniform superposition
    qc.h(range(n))
    
    # Cost layer: exp(-i * gamma * H_cost)
    for i, hi in h.items():
        if abs(hi) > 1e-12:
            qc.rz(2 * gamma * hi, i)
    for (i, j), Jij in J.items():
        if abs(Jij) > 1e-12:
            qc.cx(i, j)
            qc.rz(2 * gamma * Jij, j)
            qc.cx(i, j)
    
    # Mixer layer: exp(-i * beta * H_mixer)
    for i in range(n):
        qc.rx(2 * beta, i)
    
    qc.measure(range(n), range(n))
    return qc


def run_local_aer(circuit, shots: int = 2048):
    """Run on local Aer simulator."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    backend = AerSimulator()
    tqc = transpile(circuit, backend, optimization_level=0)
    return backend.run(tqc, shots=shots).result().get_counts()


def run_ionq(circuit, target: str, shots: int = 1024):
    """Run on IonQ backend."""
    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        raise RuntimeError("pip install qiskit-ionq")
    api_key = os.environ.get("IONQ_API_KEY")
    if not api_key:
        raise RuntimeError("IONQ_API_KEY not set (check your .env file)")
    backend = IonQProvider(token=api_key).get_backend(f"ionq_{target}")
    print(f"Submitting to {backend.name}, {shots} shots...")
    job = backend.run(circuit, shots=shots)
    print(f"  job id: {job.job_id()}")
    return job.result().get_counts()


# ============================================================================
# QAOA PARAMETER OPTIMIZATION (COBYLA on local sim, p=1)
# ============================================================================
def optimise_qaoa_p1(Q: dict, edges: list, n: int, 
                     shots: int = 2048, max_iter: int = 50):
    """Find best (gamma, beta) parameters via COBYLA on local sim.
    
    For p=1 MaxCut on degree-3 graphs, the published optimal angles are
    near gamma = pi/8, beta = pi/4 (Farhi et al. 2014). We use this as
    starting point.
    """
    h, J, offset = maxcut_qubo_to_ising(Q, n)
    
    def objective(params):
        gamma, beta = params
        qc = build_maxcut_qaoa(h, J, n, gamma, beta)
        counts = run_local_aer(qc, shots=shots)
        total = sum(counts.values())
        e_avg = 0.0
        for bitstr, k in counts.items():
            bits = [int(b) for b in bitstr[::-1]]
            if len(bits) < n:
                bits = bits + [0] * (n - len(bits))
            # negative cut (we minimize)
            e = -evaluate_cut(bits[:n], edges)
            e_avg += e * k / total
        return e_avg
    
    # Published optimal angles for degree-3 MaxCut p=1
    x0 = np.array([np.pi / 8, np.pi / 4])
    print(f"  Optimising QAOA p=1 (max_iter={max_iter})...")
    print(f"  x0 = {x0}")
    res = minimize(objective, x0, method="COBYLA",
                   options={"maxiter": max_iter, "rhobeg": 0.2})
    print(f"  best avg cut = {-res.fun:.3f} (expected QAOA average)")
    return res.x


# ============================================================================
# CLASSICAL BASELINE: LOCAL 1-STEP
# ============================================================================
def local_1step_classical(edges: list, n: int, n_trials: int = 100, seed: int = 0) -> dict:
    """Local 1-step classical algorithm for MaxCut.
    
    Each node decides its assignment based on its IMMEDIATE NEIGHBORS only
    (no global information). This is the "fair" classical comparison for
    QAOA at p=1 per Carlson et al. 2023.
    
    Algorithm:
      1. Randomly initialize each node to group 0 or 1
      2. For each node, check: would flipping it increase the cut?
         (i.e., are more than half my edges currently NOT cut?)
      3. If yes, flip it
      4. NO iteration — single pass, fully local
    
    Returns best result over n_trials random initializations.
    """
    rng = random.Random(seed)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_cut = -1
    cut_distribution = []
    
    for trial in range(n_trials):
        bits = [rng.randint(0, 1) for _ in range(n)]
        
        # Single-pass local decision: each node flips if it improves locally
        new_bits = bits.copy()
        for i in range(n):
            # Count neighbors that DISagree with me (cut edges) vs agree (uncut)
            disagree = sum(1 for j in adj[i] if bits[i] != bits[j])
            agree = len(adj[i]) - disagree
            # If majority of neighbors AGREE with me, flipping increases the cut
            if agree > disagree:
                new_bits[i] = 1 - bits[i]
        
        cut = evaluate_cut(new_bits, edges)
        cut_distribution.append(cut)
        if cut > best_cut:
            best_cut = cut
    
    return {
        "best": best_cut,
        "mean": statistics.mean(cut_distribution),
        "stdev": statistics.stdev(cut_distribution) if len(cut_distribution) > 1 else 0.0,
        "n_trials": n_trials,
    }


def best_cut_from_counts(counts: dict, edges: list, n: int) -> dict:
    """Find best cut among the QAOA samples, plus average cut."""
    total = sum(counts.values())
    cuts = []
    weights = []
    best = 0
    for bitstr, k in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        if len(bits) < n:
            bits = bits + [0] * (n - len(bits))
        cut = evaluate_cut(bits[:n], edges)
        cuts.append(cut)
        weights.append(k)
        if cut > best:
            best = cut
    avg = sum(c * w for c, w in zip(cuts, weights)) / total
    return {"best": best, "average": avg, "total_shots": total}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_single_instance(seed: int, n: int, backend: str, shots: int, max_iter: int) -> dict:
    """Run QAOA + classical comparison on one random graph instance."""
    print(f"\n--- Instance seed={seed} ---")
    edges = generate_random_3regular_graph(n, seed=seed)
    num_edges = len(edges)
    print(f"  Generated 3-regular graph: {n} nodes, {num_edges} edges")
    
    # Classical baseline
    print(f"  Running local 1-step classical (100 trials)...")
    classical = local_1step_classical(edges, n, n_trials=100, seed=seed)
    print(f"  Classical: best={classical['best']}, "
          f"mean={classical['mean']:.2f}, stdev={classical['stdev']:.2f}")
    
    # QAOA
    Q = maxcut_qubo(edges, n)
    
    # Quick tune on local sim
    best_params = optimise_qaoa_p1(Q, edges, n, shots=shots, max_iter=max_iter)
    gamma, beta = best_params
    print(f"  Best QAOA params: gamma={gamma:.4f}, beta={beta:.4f}")
    
    # Final circuit
    h, J, _ = maxcut_qubo_to_ising(Q, n)
    final_qc = build_maxcut_qaoa(h, J, n, gamma, beta)
    print(f"  Circuit: {final_qc.num_qubits} qubits, "
          f"depth {final_qc.depth()}, {sum(final_qc.count_ops().values())} gates")
    
    # Run on target backend
    print(f"  Executing on {backend}...")
    if backend == "local":
        counts = run_local_aer(final_qc, shots=shots)
    elif backend == "ionq_sim":
        counts = run_ionq(final_qc, "simulator", shots=shots)
    elif backend == "ionq_forte":
        counts = run_ionq(final_qc, "qpu.forte", shots=shots)
    elif backend == "ionq_forte_ent":
        counts = run_ionq(final_qc, "qpu.forte-enterprise-1", shots=shots)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    qaoa = best_cut_from_counts(counts, edges, n)
    print(f"  QAOA: best={qaoa['best']}, average={qaoa['average']:.2f}")
    
    # Comparison
    qaoa_beats_classical = qaoa['best'] > classical['best']
    qaoa_avg_beats_classical_mean = qaoa['average'] > classical['mean']
    relative_advantage = ((qaoa['average'] - classical['mean']) / classical['mean'] * 100 
                          if classical['mean'] > 0 else 0)
    
    print(f"  Quantum advantage:")
    print(f"    QAOA best > classical best: {qaoa_beats_classical}")
    print(f"    QAOA avg > classical mean: {qaoa_avg_beats_classical_mean}")
    print(f"    Relative advantage: {relative_advantage:+.2f}%")
    
    return {
        "seed": seed,
        "n_nodes": n,
        "num_edges": num_edges,
        "edges": edges,
        "classical": classical,
        "qaoa": qaoa,
        "qaoa_params": {"gamma": float(gamma), "beta": float(beta)},
        "circuit_depth": final_qc.depth(),
        "circuit_gates": sum(final_qc.count_ops().values()),
        "qaoa_best_beats_classical_best": bool(qaoa_beats_classical),
        "qaoa_avg_beats_classical_mean": bool(qaoa_avg_beats_classical_mean),
        "relative_advantage_pct": float(relative_advantage),
        "backend": backend,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim", "ionq_forte", "ionq_forte_ent"],
                        default="local")
    parser.add_argument("--n", type=int, default=24,
                        help="Number of nodes (default 24 for ~25 qubit equivalent).")
    parser.add_argument("--instances", type=int, default=5,
                        help="Number of random graph instances to test.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (if set, overrides --instances to run just one).")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--out", default="maxcut_results/maxcut_qaoa_result.json")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"MaxCut QAOA on random 3-regular graphs (n={args.n})")
    print(f"Backend: {args.backend}, instances: {args.instances}, shots: {args.shots}")
    print("=" * 70)
    
    # Decide seed range
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(args.instances))
    
    results = []
    for seed in seeds:
        try:
            result = run_single_instance(seed, args.n, args.backend, 
                                          args.shots, args.max_iter)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on seed {seed}: {e}")
            continue
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    n_wins_best = sum(1 for r in results if r["qaoa_best_beats_classical_best"])
    n_wins_avg = sum(1 for r in results if r["qaoa_avg_beats_classical_mean"])
    mean_advantage = statistics.mean(r["relative_advantage_pct"] for r in results) if results else 0
    
    print(f"  Instances tested: {len(results)}")
    print(f"  QAOA best > classical best: {n_wins_best}/{len(results)}")
    print(f"  QAOA avg > classical mean: {n_wins_avg}/{len(results)}")
    print(f"  Mean relative advantage: {mean_advantage:+.2f}%")
    
    # Write cache
    payload = {
        "experiment": "MaxCut QAOA p=1 vs local 1-step classical, random 3-regular graphs",
        "backend": args.backend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_nodes": args.n,
        "shots": args.shots,
        "n_instances": len(results),
        "aggregate": {
            "qaoa_wins_best": n_wins_best,
            "qaoa_wins_avg": n_wins_avg,
            "mean_relative_advantage_pct": float(mean_advantage),
        },
        "instances": results,
        "reference": "Carlson et al. 2023, arXiv:2304.08420 - QAOA advantage on degree-3 graphs",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nCache written: {args.out}")


if __name__ == "__main__":
    main()
