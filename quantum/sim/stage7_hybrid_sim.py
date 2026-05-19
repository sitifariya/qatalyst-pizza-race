"""
Stage 7 Hybrid (SIMULATOR variant) — 5+5 customers, 25 qubits per cluster
=========================================================================

This is the laptop-runnable variant of stage7_hybrid.py.

Why this exists
---------------
The main `stage7_hybrid.py` uses 6+6 customers (36 qubits per cluster) which
fits IonQ Forte exactly but exceeds local Aer simulator (~25 qubit ceiling)
and the free IonQ Cloud simulator. To validate the hybrid pipeline end-to-end
without waiting for Forte access, we drop two boundary customers (Eli, Sam)
and run the same algorithm at 5+5 = 25 qubits per cluster.

Hardware fits
-------------
  - Local Aer simulator      ✓
  - IonQ Cloud simulator     ✓
  - IonQ Forte (real HW)     ✓ (well under 36-qubit limit)

Demonstrated result (local simulation, May 2026)
------------------------------------------------
  Hybrid:                       score 550 (7 hot, 3 cold)
  Classical SA, 100 restarts:
    best 550, mean 442, worst 400, stdev 68
  Hybrid wins:                  72/100 trials
  Hybrid matches or beats:      100/100 trials

The win-rate dropped from 100/100 (at 6+6) to 72/100 (at 5+5) because the
smaller problem is easier for classical SA. The hybrid still demonstrably
beats classical's mean by 25% (550 vs 442), which is the defensible claim.

Usage
-----
    # Local: full pipeline runs in ~30 seconds
    python stage7_hybrid_sim.py --backend local

    # IonQ Cloud simulator (free, queue 1-10 min)
    python stage7_hybrid_sim.py --backend ionq_sim

This file lives in quantum/sim/ to keep it separate from the Forte-targeted
scripts at the top of quantum/.
"""
import argparse
import math
import os
import sys
import json
import random
import statistics
from datetime import datetime, timezone
from itertools import permutations

# Look for qubo_builder.py in the parent quantum/ folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, THIS_DIR)

from qubo_builder import (
    distance, build_distance_matrix,
    assignment_penalty, distance_cost, cold_penalty,
    qubo_to_ising, build_qaoa_circuit,
    run_local, run_ionq, optimise_qaoa,
    best_valid_route_from_counts,
)


# ============================================================================
# PROBLEM: 10-customer subset of game's Stage 7
# ============================================================================
SHOP = (350, 200)
CUSTOMERS = [
    # NW cluster (5 customers)
    {"id": 0, "name": "Ria",  "x": 130, "y": 110, "hotBy": 18},
    {"id": 1, "name": "Adam", "x": 180, "y": 85,  "hotBy": 16},
    {"id": 2, "name": "Luna", "x": 225, "y": 140, "hotBy": 20},
    {"id": 3, "name": "Jay",  "x": 150, "y": 170, "hotBy": 22},
    {"id": 4, "name": "Max",  "x": 340, "y": 100, "hotBy": 15},
    # SE cluster (5 customers)
    {"id": 5, "name": "Bee",  "x": 510, "y": 325, "hotBy": 42},
    {"id": 6, "name": "Kai",  "x": 555, "y": 340, "hotBy": 44},
    {"id": 7, "name": "Leo",  "x": 480, "y": 300, "hotBy": 40},
    {"id": 8, "name": "Zara", "x": 545, "y": 280, "hotBy": 38},
    {"id": 9, "name": "Nia",  "x": 355, "y": 315, "hotBy": 36},
]
FUEL_TANK = 130.0
NUM_CUSTOMERS = len(CUSTOMERS)


# ============================================================================
# STEP 1: CLASSICAL CLUSTERING (k-means)
# ============================================================================
def kmeans_partition(customers, max_iter=20):
    n = len(customers)
    centroid_a = (customers[0]["x"], customers[0]["y"])
    centroid_b = (customers[5]["x"], customers[5]["y"])

    for _ in range(max_iter):
        assignment = []
        for c in customers:
            d_a = math.hypot(c["x"] - centroid_a[0], c["y"] - centroid_a[1])
            d_b = math.hypot(c["x"] - centroid_b[0], c["y"] - centroid_b[1])
            assignment.append(0 if d_a < d_b else 1)
        cust_a = [customers[i] for i in range(n) if assignment[i] == 0]
        cust_b = [customers[i] for i in range(n) if assignment[i] == 1]
        if cust_a:
            centroid_a = (sum(c["x"] for c in cust_a) / len(cust_a),
                          sum(c["y"] for c in cust_a) / len(cust_a))
        if cust_b:
            centroid_b = (sum(c["x"] for c in cust_b) / len(cust_b),
                          sum(c["y"] for c in cust_b) / len(cust_b))

    return ([i for i in range(n) if assignment[i] == 0],
            [i for i in range(n) if assignment[i] == 1])


# ============================================================================
# STEP 2: PER-CLUSTER QAOA
# ============================================================================
def build_cluster_qubo(cluster_ids):
    cluster_customers = [CUSTOMERS[i] for i in cluster_ids]
    n = len(cluster_customers)
    if n == 0:
        return {}, 0, []

    D = build_distance_matrix(SHOP, cluster_customers)
    Q = {}
    assignment_strength = float(D.max()) * 4.0
    assignment_penalty(Q, n, assignment_strength)
    distance_cost(Q, D, n)
    cold_weight = float(D.max()) * 2.0
    cold_penalty(Q, cluster_customers, D, n, cold_weight)
    return Q, n * n, cluster_customers


def solve_cluster_qaoa(cluster_ids, backend, p, shots, max_iter):
    n = len(cluster_ids)
    if n == 0:
        return [], 0.0, 1.0
    if n == 1:
        return [cluster_ids[0]], 0.0, 1.0

    Q, num_vars, cluster_customers = build_cluster_qubo(cluster_ids)

    # Tiny clusters: brute-force
    if n <= 4:
        best_perm = None
        best_score = float("-inf")
        for perm in permutations(range(n)):
            t = 0.0; prev = SHOP
            hot = 0; cold = 0
            for li in perm:
                c = cluster_customers[li]
                t += distance(prev, (c["x"], c["y"]))
                if t > c.get("hotBy", 1e9):
                    cold += 1
                else:
                    hot += 1
                prev = (c["x"], c["y"])
            s = hot * 100 - cold * 50
            if s > best_score:
                best_score = s
                best_perm = perm
        route_orig = [cluster_ids[i] for i in best_perm]
        print(f"    n={n} → classical brute-force. "
              f"Route: {'-'.join(CUSTOMERS[i]['name'] for i in route_orig)}")
        return route_orig, 0.0, 1.0

    # n=5: full QAOA pipeline (25 qubits, fits laptop)
    h, J, _ = qubo_to_ising(Q, num_vars)
    print(f"    QUBO: {num_vars} qubits, {len(Q)} terms")
    print(f"    Tuning QAOA params on local simulator…")
    best_params = optimise_qaoa(Q, num_vars, p=p, shots=shots, max_iter=max_iter)

    if p == 1:
        final_qc = build_qaoa_circuit(h, J, num_vars,
                                      best_params[0], best_params[1], p=1)
    else:
        final_qc = build_qaoa_circuit(h, J, num_vars,
                                      best_params[:p], best_params[p:], p=p)
    print(f"    Final circuit: {final_qc.num_qubits} qubits, "
          f"depth {final_qc.depth()}, "
          f"{sum(final_qc.count_ops().values())} gates")

    print(f"    Submitting to {backend}…")
    if backend == "local":
        counts = run_local(final_qc, shots=shots)
    elif backend == "ionq_sim":
        counts = run_ionq(final_qc, "simulator", shots=shots)
    elif backend == "ionq_forte":
        counts = run_ionq(final_qc, "qpu.forte", shots=shots)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    route_local, energy, hits, total, valid_frac = best_valid_route_from_counts(
        counts, Q, num_vars, n
    )

    if route_local is None:
        # Fallback: brute-force the cluster by GAME SCORE (not QUBO energy).
        # The QUBO is an approximation; for small clusters we want the
        # game's actual scoring to drive the choice.
        print(f"    QAOA returned no valid permutation. Falling back to brute force.")
        best_perm = None
        best_game_score = float("-inf")
        for perm in permutations(range(n)):
            t = 0.0; prev = SHOP; hot = 0; cold = 0
            for li in perm:
                c = cluster_customers[li]
                t += distance(prev, (c["x"], c["y"]))
                if t > c.get("hotBy", 1e9):
                    cold += 1
                else:
                    hot += 1
                prev = (c["x"], c["y"])
            game_score_local = hot * 100 - cold * 50
            if game_score_local > best_game_score:
                best_game_score = game_score_local
                best_perm = perm
        route_orig = [cluster_ids[i] for i in best_perm]
        return route_orig, 0.0, 0.0

    route_orig = [cluster_ids[i] for i in route_local]
    print(f"    Valid permutations: {valid_frac*100:.2f}% of {total} shots")
    print(f"    Best route appeared {hits}/{total} times")
    return route_orig, energy, valid_frac


# ============================================================================
# STEP 3: SCORE FULL SOLUTION
# ============================================================================
def score_full_solution(routes):
    vans = []
    for route in routes:
        if not route:
            vans.append({"d": 0, "fuel": 0, "hot": 0, "cold": 0, "overFuel": False})
            continue
        d = 0.0; fuel = 0.0; t = 0.0; cold = 0
        prev = SHOP
        for ci in route:
            c = CUSTOMERS[ci]
            leg = distance(prev, (c["x"], c["y"]))
            d += leg; fuel += leg; t += leg
            if t > c.get("hotBy", 1e9):
                cold += 1
            prev = (c["x"], c["y"])
        back = distance(prev, SHOP)
        d += back; fuel += back
        vans.append({"d": d, "fuel": fuel,
                     "hot": len(route) - cold, "cold": cold,
                     "overFuel": fuel > FUEL_TANK})
    total_hot = sum(v["hot"] for v in vans)
    total_cold = sum(v["cold"] for v in vans)
    score = total_hot * 100 - total_cold * 50
    for v in vans:
        if v["overFuel"]:
            score -= (v["fuel"] - FUEL_TANK) * 5
    return {"score": score, "hot": total_hot, "cold": total_cold, "vans": vans}


# ============================================================================
# STEP 4: CLASSICAL SA BASELINE
# ============================================================================
def classical_sa_baseline(num_restarts=100, iters=2000):
    def order_nn(route_ids):
        if not route_ids:
            return []
        out = []
        remaining = list(route_ids)
        prev = SHOP
        while remaining:
            best = min(remaining,
                       key=lambda j: distance(prev,
                                              (CUSTOMERS[j]["x"], CUSTOMERS[j]["y"])))
            out.append(best)
            remaining.remove(best)
            prev = (CUSTOMERS[best]["x"], CUSTOMERS[best]["y"])
        return out

    def single_sa(seed):
        random.seed(seed)
        assignment = [random.randint(0, 1) for _ in range(NUM_CUSTOMERS)]
        if sum(assignment) == 0: assignment[0] = 1
        if sum(assignment) == NUM_CUSTOMERS: assignment[0] = 0
        r0 = order_nn([i for i in range(NUM_CUSTOMERS) if assignment[i] == 0])
        r1 = order_nn([i for i in range(NUM_CUSTOMERS) if assignment[i] == 1])
        sol = [r0, r1]
        s = score_full_solution(sol)["score"]
        best_s = s
        best_sol = [r0[:], r1[:]]
        T = 80.0
        cooling = (0.5 / T) ** (1 / iters)

        for _ in range(iters):
            if random.random() < 0.5:
                cust = random.randint(0, NUM_CUSTOMERS - 1)
                v_from = 0 if cust in sol[0] else 1
                cand = [sol[0][:], sol[1][:]]
                if v_from == 0:
                    cand[0].remove(cust); cand[1].append(cust)
                else:
                    cand[1].remove(cust); cand[0].append(cust)
            else:
                v = random.randint(0, 1)
                if len(sol[v]) < 2: continue
                i = random.randint(0, len(sol[v]) - 1)
                k = random.randint(0, len(sol[v]) - 1)
                if i > k: i, k = k, i
                new_route = sol[v][:i] + sol[v][i:k+1][::-1] + sol[v][k+1:]
                cand = [sol[0][:], sol[1][:]]; cand[v] = new_route
            cs = score_full_solution(cand)["score"]
            delta = cs - s
            if delta > 0 or random.random() < math.exp(delta / T):
                sol = cand; s = cs
                if s > best_s: best_s = s; best_sol = [sol[0][:], sol[1][:]]
            T *= cooling
        return best_sol, best_s

    scores = []
    best_overall = None
    for seed in range(num_restarts):
        sol, s = single_sa(seed)
        scores.append(s)
        if best_overall is None or s > best_overall[1]:
            best_overall = (sol, s)
    return scores, best_overall


# ============================================================================
# CACHE WRITER
# ============================================================================
def write_cache(out_path, *, routes, hybrid_score, sa_scores, sa_best,
                cluster_a, cluster_b, backend, p, shots,
                qaoa_energies, valid_fractions):
    payload = {
        "stage_id": "7_hybrid_sim",
        "approach": "k-means + per-cluster QAOA (simulator-runnable 5+5 variant)",
        "backend": backend,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "p": int(p),
        "shots_per_cluster": int(shots),
        "clusters": {
            "A": {"customer_ids": [int(i) for i in cluster_a],
                  "size": len(cluster_a), "qubits": len(cluster_a)**2},
            "B": {"customer_ids": [int(i) for i in cluster_b],
                  "size": len(cluster_b), "qubits": len(cluster_b)**2},
        },
        "qaoa_energies": [float(e) for e in qaoa_energies],
        "valid_fractions": [float(v) for v in valid_fractions],
        "hybrid_route": [[int(x) for x in r] for r in routes],
        "hybrid_score": float(hybrid_score["score"]),
        "hybrid_hot": int(hybrid_score["hot"]),
        "hybrid_cold": int(hybrid_score["cold"]),
        "classical_sa": {
            "n_restarts": len(sa_scores),
            "best": float(max(sa_scores)),
            "mean": float(statistics.mean(sa_scores)),
            "worst": float(min(sa_scores)),
            "stdev": float(statistics.stdev(sa_scores)) if len(sa_scores) > 1 else 0.0,
            "best_route": [[int(x) for x in r] for r in sa_best[0]],
        },
        "hybrid_wins_count": sum(1 for s in sa_scores if s < hybrid_score["score"]),
        "hybrid_wins_or_ties": sum(1 for s in sa_scores if s <= hybrid_score["score"]),
        "note": ("This is the laptop/cloud-runnable variant of Stage 7's "
                 "hybrid pipeline. 5+5 customers (instead of 6+6) so 25 "
                 "qubits per cluster fits all available simulators. "
                 "Full 6+6 variant is at quantum/stage7_hybrid.py."),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nCache written: {out_path}")
    return payload


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim", "ionq_forte"],
                        default="local")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--sa-restarts", type=int, default=100)
    parser.add_argument("--sa-iters", type=int, default=2000)
    parser.add_argument("--out", default="stage7_sim_results/stage7_hybrid_sim_result.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 7 Hybrid (SIM variant) — 5+5 customers, 25 qubits/cluster")
    print(f"Backend: {args.backend}, p={args.p}, shots/cluster={args.shots}")
    print("=" * 70)

    # ----- STEP 1 -----
    print("\n[1/4] Classical k-means clustering (k=2)…")
    cluster_a, cluster_b = kmeans_partition(CUSTOMERS)
    print(f"  Cluster A ({len(cluster_a)}): "
          f"{[CUSTOMERS[i]['name'] for i in cluster_a]}")
    print(f"  Cluster B ({len(cluster_b)}): "
          f"{[CUSTOMERS[i]['name'] for i in cluster_b]}")
    print(f"  QUBO sizes: {len(cluster_a)**2} and {len(cluster_b)**2} qubits")

    # ----- STEP 2 -----
    print(f"\n[2/4] Solving each cluster on {args.backend}…")
    print(f"  Cluster A:")
    route_a, energy_a, vf_a = solve_cluster_qaoa(
        cluster_a, args.backend, args.p, args.shots, args.max_iter)
    print(f"  Cluster B:")
    route_b, energy_b, vf_b = solve_cluster_qaoa(
        cluster_b, args.backend, args.p, args.shots, args.max_iter)
    routes = [route_a, route_b]

    # ----- STEP 3 -----
    print("\n[3/4] Hybrid solution:")
    hybrid_eval = score_full_solution(routes)
    print(f"  Van 1: {route_a} ({'-'.join(CUSTOMERS[i]['name'] for i in route_a)})")
    print(f"  Van 2: {route_b} ({'-'.join(CUSTOMERS[i]['name'] for i in route_b)})")
    print(f"  Score: {hybrid_eval['score']}  "
          f"(hot={hybrid_eval['hot']}, cold={hybrid_eval['cold']})")

    # ----- STEP 4 -----
    print(f"\n[4/4] Classical SA baseline "
          f"({args.sa_restarts} restarts)…")
    sa_scores, sa_best = classical_sa_baseline(num_restarts=args.sa_restarts,
                                                iters=args.sa_iters)
    print(f"  best: {max(sa_scores)}, mean: {statistics.mean(sa_scores):.1f}, "
          f"worst: {min(sa_scores)}, stdev: {statistics.stdev(sa_scores):.1f}")

    hybrid_wins = sum(1 for s in sa_scores if s < hybrid_eval["score"])
    hybrid_ties_or_wins = sum(1 for s in sa_scores if s <= hybrid_eval["score"])
    print(f"\n=== COMPARISON ===")
    print(f"  Hybrid:                          {hybrid_eval['score']}")
    print(f"  Classical SA best of {args.sa_restarts}:   {max(sa_scores)}")
    print(f"  Classical SA mean of {args.sa_restarts}:  {statistics.mean(sa_scores):.1f}")
    print(f"  Hybrid strictly beats:           {hybrid_wins}/{args.sa_restarts}")
    print(f"  Hybrid matches or beats:         {hybrid_ties_or_wins}/{args.sa_restarts}")

    write_cache(args.out,
                routes=routes,
                hybrid_score=hybrid_eval,
                sa_scores=sa_scores,
                sa_best=sa_best,
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                backend=args.backend,
                p=args.p,
                shots=args.shots,
                qaoa_energies=[energy_a, energy_b],
                valid_fractions=[vf_a, vf_b])


if __name__ == "__main__":
    main()
