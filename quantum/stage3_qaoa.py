"""
Stage 3 — "The split" — multi-van QAOA pipeline.

What's new vs Stage 2:
  - 6 customers across 2 vans (fleet split problem)
  - 72 qubits total (6 * 6 positions * 2 vans)
  - Combinatorial assignment + per-van TSP simultaneously

Hardware reality:
  72 qubits is well beyond IonQ Forte's 36-qubit limit. This stage runs on
  the IonQ Cloud noiseless simulator only. Real-hardware execution would
  require either:
    (a) Future hardware with 100+ qubits (post-2027), or
    (b) Hybrid decomposition: classical clustering layer feeds two smaller
        per-van QUBOs (each ~18 qubits) to Forte. We don't do (b) here;
        it's a different technique that we'll cover in a separate
        per-stage script later.

Usage:
    python stage3_qaoa.py --backend local
    python stage3_qaoa.py --backend ionq_sim   # only sensible cloud target

Note: do NOT pass --backend ionq_forte for this stage. The submission will
fail (qubit count exceeds the device limit) or hang.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qubo_builder import (
    build_distance_matrix, qubo_to_ising, build_qaoa_circuit,
    run_local, run_ionq, optimise_qaoa, expected_energy,
    write_cache, qubo_energy,
)
from qubo_builder_multivan import (
    assignment_penalty_mv, distance_cost_mv, cold_penalty_mv,
    fuel_overrun_penalty_mv, decode_mv, enumerate_brute_force_mv,
)


# ----------------------------------------------------------------------------
# STAGE 3 PROBLEM (mirrors stages.js Stage 3)
# ----------------------------------------------------------------------------
SHOP = (350, 200)
CUSTOMERS = [
    {"id": 0, "name": "Ria",  "x": 120, "y": 105, "hotBy": 28},
    {"id": 1, "name": "Adam", "x": 590, "y": 110, "hotBy": 32},
    {"id": 2, "name": "Luna", "x": 90,  "y": 300, "hotBy": 42},
    {"id": 3, "name": "Jay",  "x": 600, "y": 300, "hotBy": 34},
    {"id": 4, "name": "Bee",  "x": 260, "y": 330, "hotBy": 22},
    {"id": 5, "name": "Kai",  "x": 460, "y": 335, "hotBy": 24},
]
FUEL_TANK = 28.0
V = 2  # vans
N = len(CUSTOMERS)
NUM_VARS = N * N * V  # 72 qubits


def build_stage3_qubo():
    D = build_distance_matrix(SHOP, CUSTOMERS)
    Q = {}
    assignment_strength = float(D.max()) * 6.0  # stronger than single-van
    assignment_penalty_mv(Q, N, V, assignment_strength)
    distance_cost_mv(Q, D, N, V)
    cold_weight = float(D.max()) * 2.0
    cold_penalty_mv(Q, CUSTOMERS, D, N, V, cold_weight)
    fuel_pen = 5.0
    fuel_overrun_penalty_mv(Q, D, N, V, FUEL_TANK, fuel_pen)
    return Q, assignment_strength, cold_weight, fuel_pen


def best_valid_route_from_counts_mv(counts, Q, num_vars, n, V):
    total = sum(counts.values())
    valid_count = 0
    best = None
    for bitstr, k in counts.items():
        bits = [int(b) for b in bitstr[::-1]]
        if len(bits) < num_vars:
            bits = bits + [0] * (num_vars - len(bits))
        routes, ok = decode_mv(bits[:num_vars], n, V)
        if not ok:
            continue
        # Skip configurations where a van has zero or all customers
        # (those are degenerate, not what Stage 3 is asking for)
        total_assigned = sum(len(r) for r in routes)
        if total_assigned != n:
            continue
        valid_count += k
        e = qubo_energy(Q, bits[:num_vars])
        if best is None or e < best[1]:
            best = (routes, e, k, bitstr)
    valid_frac = valid_count / total
    if best is None:
        return None, None, 0, total, valid_frac
    return best[0], best[1], best[2], total, valid_frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim"],
                        default="local",
                        help="Stage 3 has 72 qubits — only simulator backends are valid.")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--out", default="stage3_results/stage3_qaoa_result.json")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip local parameter optimisation. Uses fixed gamma=0.5, beta=0.4. "
                             "Required for 72-qubit Stage 3 since local statevector sim runs out of memory.")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Stage 3 — The split (6 customers, 2 vans, fuel={FUEL_TANK} each)")
    print(f"Backend: {args.backend}, p={args.p}, shots={args.shots}")
    print(f"** {NUM_VARS} qubits — simulator only **")
    print("=" * 70)

    # Step 1: build QUBO
    Q, asgn_str, cold_w, fuel_pen = build_stage3_qubo()
    print(f"\nQUBO: {NUM_VARS} variables, {len(Q)} terms")
    print(f"  assignment penalty = {asgn_str:.2f}")
    print(f"  cold-pizza penalty = {cold_w:.2f}")
    print(f"  fuel overrun rate  = {fuel_pen:.2f}/unit")

    # Step 2: classical brute force over partitions and orderings
    bf = enumerate_brute_force_mv(CUSTOMERS, SHOP, fuel_tank=FUEL_TANK, V=V)
    print(f"\nBrute-force optimum (over 5040 candidates):")
    print(f"  Van 1: {bf['sol'][0]} ({'-'.join(CUSTOMERS[i]['name'] for i in bf['sol'][0])})")
    print(f"  Van 2: {bf['sol'][1]} ({'-'.join(CUSTOMERS[i]['name'] for i in bf['sol'][1])})")
    print(f"  Score: {bf['score']:.2f} (hot={bf['hot']}/6, cold={bf['cold']}/6)")
    for k, v in enumerate(bf['vans']):
        print(f"  Van {k+1}: dist={v['d']:.1f}, fuel={v['fuel']:.1f}, overFuel={v['overFuel']}")

    # Step 3: tune QAOA params on local simulator (or skip)
    import numpy as np
    if args.skip_tuning:
        # 72 qubits exceeds laptop memory for statevector simulation.
        # Use empirically-reasonable fixed parameters and let the cloud
        # simulator (tensor-network based) execute the final circuit.
        if args.p == 1:
            best_params = np.array([0.5, 0.4])
        else:
            best_params = np.concatenate([
                np.full(args.p, 0.5),  # gammas
                np.full(args.p, 0.4),  # betas
            ])
        print(f"\nSkipping local tuning. Fixed params: {best_params}")
    else:
        print(f"\nNote: 72 qubits at p={args.p} is expensive. Each COBYLA iter takes ~30-90s.")
        best_params = optimise_qaoa(Q, NUM_VARS, p=args.p, shots=args.shots,
                                    max_iter=args.max_iter)
        print(f"  best params: {best_params}")

    # Step 4: build the final circuit
    h, J, _ = qubo_to_ising(Q, NUM_VARS)
    if args.p == 1:
        final_qc = build_qaoa_circuit(h, J, NUM_VARS,
                                      best_params[0], best_params[1], p=1)
    else:
        final_qc = build_qaoa_circuit(h, J, NUM_VARS,
                                      best_params[:args.p],
                                      best_params[args.p:], p=args.p)
    print(f"\nFinal circuit: {final_qc.num_qubits} qubits, "
          f"depth {final_qc.depth()}, "
          f"{sum(final_qc.count_ops().values())} gates")

    # Step 5: execute
    print(f"\nExecuting on {args.backend}…")
    if args.backend == "local":
        counts = run_local(final_qc, shots=args.shots)
        backend_label = "qiskit_aer_local"
    elif args.backend == "ionq_sim":
        counts = run_ionq(final_qc, "simulator", shots=args.shots)
        backend_label = "ionq_simulator"

    # Step 6: decode
    routes, energy, hits, total, valid_frac = best_valid_route_from_counts_mv(
        counts, Q, NUM_VARS, N, V
    )

    print(f"\n--- RESULTS ---")
    print(f"valid multi-van assignments: {valid_frac*100:.2f}% of {total} shots")
    if routes is None:
        print("\nNo valid multi-van assignment surfaced.")
        print("With 72 qubits, the one-hot encoding is sparse. Suggested fixes:")
        print("  1. Increase --shots (try 8192)")
        print("  2. Increase --p (try 3 or 4) at the cost of much longer optimisation")
        print("  3. Raise the assignment penalty in build_stage3_qubo()")
        print("  4. Use hybrid decomposition instead (separate per-van QUBOs)")
        return

    # Evaluate the returned routes using the same scoring as brute-force
    # Quick re-evaluation
    import math
    def dist2(a, b): return math.hypot(a[0]-b[0], a[1]-b[1]) / 10.0
    def eval_van(route):
        if not route: return {"d":0,"fuel":0,"hot":0,"cold":0,"overFuel":False}
        d=0; fuel=0; t=0; cold=0; prev=SHOP
        for ci in route:
            c=CUSTOMERS[ci]; leg=dist2(prev,(c["x"],c["y"]))
            d+=leg; fuel+=leg; t+=leg
            if t>c.get("hotBy",1e9): cold+=1
            prev=(c["x"],c["y"])
        back=dist2(prev,SHOP); d+=back; fuel+=back
        return {"d":d,"fuel":fuel,"hot":len(route)-cold,"cold":cold,"overFuel":fuel>FUEL_TANK}
    vans_eval = [eval_van(r) for r in routes]
    h_tot = sum(v["hot"] for v in vans_eval)
    c_tot = sum(v["cold"] for v in vans_eval)
    game_score = h_tot*100 - c_tot*50
    for v in vans_eval:
        if v["overFuel"]: game_score -= (v["fuel"] - FUEL_TANK) * 5

    print(f"\nQAOA returned:")
    print(f"  Van 1: {routes[0]} ({'-'.join(CUSTOMERS[i]['name'] for i in routes[0])})")
    print(f"  Van 2: {routes[1]} ({'-'.join(CUSTOMERS[i]['name'] for i in routes[1])})")
    print(f"  QUBO energy: {energy:.3f} (measured {hits}/{total} times)")
    print(f"  game score: {game_score:.2f} "
          f"(hot={h_tot}/6, cold={c_tot}/6)")
    for k, v in enumerate(vans_eval):
        print(f"  Van {k+1}: dist={v['d']:.1f}, fuel={v['fuel']:.1f}, overFuel={v['overFuel']}")

    if game_score >= bf['score'] - 0.5:
        print(f"\n✓ Game-score matches brute force ({bf['score']:.2f}).")
    else:
        print(f"\n  Gap from optimum: {bf['score'] - game_score:.2f}")

    # Step 7: write cache
    write_cache(args.out,
                stage_id=3,
                route=[[int(x) for x in r] for r in routes],  # nested for multi-van
                customers=CUSTOMERS,
                backend=backend_label,
                energy=energy,
                valid_fraction=valid_frac,
                total_shots=total,
                num_qubits=NUM_VARS,
                p=args.p,
                qubo_size=len(Q),
                extra={"game_score": game_score,
                       "game_hot": h_tot,
                       "game_cold": c_tot,
                       "vans": V,
                       "fuel_tank": FUEL_TANK,
                       "encoding_note":
                           f"multi-van one-hot, N*N*V variables; brute-force "
                           f"reference score {bf['score']:.2f}"})


if __name__ == "__main__":
    main()
