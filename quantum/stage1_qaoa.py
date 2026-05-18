"""
Stage 1 — "Pizza goes cold" — QAOA pipeline.

What's new vs Stage 0:
  - 4 customers (16 qubits, fits Forte easily)
  - Time-window constraints: each customer has a hotBy deadline.
  - The QUBO includes a linear cold-pizza penalty: assigning customer i to
    position t when even the best-case arrival time at position t exceeds
    customer i's deadline incurs cost `cold_weight`.

Note: the cold penalty is a linear approximation, not the exact game scoring
(which depends on the full route). QAOA biases toward respecting deadlines,
but the final "score" we publish for the cache uses the game's real scoring
to keep the result consistent with what players see.

Usage:
    python stage1_qaoa.py --backend local
    python stage1_qaoa.py --backend ionq_sim
    python stage1_qaoa.py --backend ionq_forte
"""
import argparse
import sys
import os

# Import shared utilities. Works either from this folder or one above.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qubo_builder import (
    build_distance_matrix,
    assignment_penalty,
    distance_cost,
    cold_penalty,
    qubo_to_ising,
    build_qaoa_circuit,
    run_local,
    run_ionq,
    optimise_qaoa,
    best_valid_route_from_counts,
    brute_force_qubo_optimum,
    game_score,
    write_cache,
)


# ----------------------------------------------------------------------------
# STAGE 1 PROBLEM (mirrors stages.js Stage 1)
# ----------------------------------------------------------------------------
SHOP = (350, 200)
CUSTOMERS = [
    {"id": 0, "name": "Ria",  "x": 200, "y": 140, "hotBy": 30},
    {"id": 1, "name": "Adam", "x": 500, "y": 140, "hotBy": 34},
    {"id": 2, "name": "Bee",  "x": 200, "y": 280, "hotBy": 36},
    {"id": 3, "name": "Luna", "x": 500, "y": 280, "hotBy": 40},
]
N = len(CUSTOMERS)
NUM_VARS = N * N  # 16


def build_stage1_qubo():
    """Distance + assignment penalty + cold-pizza linear penalty."""
    D = build_distance_matrix(SHOP, CUSTOMERS)
    Q = {}
    # Strong assignment penalty so QAOA cares about valid permutations first.
    assignment_strength = float(D.max()) * 4.0
    assignment_penalty(Q, N, assignment_strength)
    distance_cost(Q, D, N)
    # Cold-pizza penalty roughly the same order as the "hot pizza" reward in
    # the game (100 game-points). Scale it relative to distance so QUBO terms
    # are commensurate.
    cold_weight = float(D.max()) * 2.0
    cold_penalty(Q, CUSTOMERS, D, N, cold_weight)
    return Q, assignment_strength, cold_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim", "ionq_aria", "ionq_forte"],
                        default="local")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--out", default="stage1_results/stage1_qaoa_result.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Stage 1 — Pizza goes cold (4 customers, time windows)")
    print(f"Backend: {args.backend}, p={args.p}, shots={args.shots}")
    print("=" * 70)

    # Step 1: build QUBO
    Q, asgn_str, cold_w = build_stage1_qubo()
    print(f"\nQUBO: {NUM_VARS} variables, {len(Q)} terms")
    print(f"  assignment penalty = {asgn_str:.2f}")
    print(f"  cold-pizza penalty = {cold_w:.2f}")

    # Step 2: brute-force the QUBO to know the QAOA target
    bf_route, bf_energy = brute_force_qubo_optimum(Q, NUM_VARS, N)
    bf_score = game_score(bf_route, SHOP, CUSTOMERS, constraints=True)
    print(f"\nBrute-force QUBO optimum:")
    print(f"  route: {bf_route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in bf_route)})")
    print(f"  QUBO energy: {bf_energy:.3f}")
    print(f"  game score (real scoring): {bf_score['score']} "
          f"(hot={bf_score['hot']}, cold={bf_score['cold']})")

    # Step 3: tune QAOA params on local simulator
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

    # Step 5: run on requested backend
    print(f"\nExecuting on {args.backend}…")
    if args.backend == "local":
        counts = run_local(final_qc, shots=args.shots)
        backend_label = "qiskit_aer_local"
    elif args.backend == "ionq_sim":
        counts = run_ionq(final_qc, "simulator", shots=args.shots)
        backend_label = "ionq_simulator"
    elif args.backend == "ionq_aria":
        counts = run_ionq(final_qc, "qpu.aria-1", shots=args.shots)
        backend_label = "ionq_aria"
    elif args.backend == "ionq_forte":
        counts = run_ionq(final_qc, "qpu.forte", shots=args.shots)
        backend_label = "ionq_forte"

    # Step 6: decode best valid route from samples
    route, energy, hits, total, valid_frac = best_valid_route_from_counts(
        counts, Q, NUM_VARS, N
    )

    print(f"\n--- RESULTS ---")
    print(f"valid permutations: {valid_frac*100:.1f}% of {total} shots")
    if route is None:
        print("No valid permutation observed. Increase --p, --max-iter, or --shots.")
        return

    score = game_score(route, SHOP, CUSTOMERS, constraints=True)
    print(f"best valid route: {route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in route)})")
    print(f"  QUBO energy: {energy:.3f} (measured {hits}/{total} times)")
    print(f"  game score: {score['score']} "
          f"(hot={score['hot']}, cold={score['cold']})")
    print(f"  arrivals: {[(CUSTOMERS[ci]['name'], round(score['arrivals'][k],1), CUSTOMERS[ci]['hotBy']) for k, ci in enumerate(route)]}")

    if energy <= bf_energy + 1e-6:
        print(f"\n✓ QAOA matched the brute-force optimum.")
    else:
        gap = energy - bf_energy
        print(f"\n  Approximation gap: {gap:.3f} above optimum.")

    # Step 7: write cache for the whitepaper / future game integration
    write_cache(args.out,
                stage_id=1,
                route=route,
                customers=CUSTOMERS,
                backend=backend_label,
                energy=energy,
                valid_fraction=valid_frac,
                total_shots=total,
                num_qubits=NUM_VARS,
                p=args.p,
                qubo_size=len(Q),
                extra={"game_score": score["score"],
                       "game_hot": score["hot"],
                       "game_cold": score["cold"]})


if __name__ == "__main__":
    main()
