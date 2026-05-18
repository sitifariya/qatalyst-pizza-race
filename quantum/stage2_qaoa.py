"""
Stage 2 — "Fuel runs out" — QAOA pipeline.

What's new vs Stage 1:
  - 4 customers in a clean rectangle (16 qubits, same as Stage 1)
  - Deadlines are loose (80 min each) so cold-pizza pressure is mild.
  - The new mechanic is a **fuel tank**. Total route distance (including the
    return-to-shop leg) must stay under fuel_tank. Overruns are penalised
    proportional to the excess.

QUBO additions over Stage 1:
  - The distance cost is already quadratic in the route variables, so total
    fuel is implicit in the existing distance terms. We add an *overrun
    penalty* term that activates only when total distance exceeds the tank.
  - Since QUBO can't directly encode "max(0, fuel - tank)", we use a
    surrogate: amplify distance terms by a soft factor that grows when the
    estimated route length is near or above the tank. This nudges QAOA
    toward shorter routes without changing the optimisation structure.

Usage:
    python stage2_qaoa.py --backend local
    python stage2_qaoa.py --backend ionq_sim
    python stage2_qaoa.py --backend ionq_forte
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qubo_builder import (
    build_distance_matrix,
    assignment_penalty,
    distance_cost,
    add_qubo_term,
    var_index,
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
# STAGE 2 PROBLEM (mirrors stages.js Stage 2)
# ----------------------------------------------------------------------------
SHOP = (350, 200)
CUSTOMERS = [
    {"id": 0, "name": "Ria",  "x": 160, "y": 160, "hotBy": 80},
    {"id": 1, "name": "Adam", "x": 540, "y": 160, "hotBy": 80},
    {"id": 2, "name": "Bee",  "x": 160, "y": 290, "hotBy": 80},
    {"id": 3, "name": "Luna", "x": 540, "y": 290, "hotBy": 80},
]
FUEL_TANK = 135.0
N = len(CUSTOMERS)
NUM_VARS = N * N  # 16


def fuel_overrun_penalty(Q, D, n, fuel_tank, penalty):
    """Add a quadratic penalty term that grows when route distance approaches fuel_tank.

    Approximation: we amplify the distance terms by a small extra factor.
    The exact "ReLU(fuel - tank)" needs slack variables (which add more qubits).
    Here we use a softer approach: each edge whose distance exceeds
    fuel_tank/N gets a small extra penalty, biasing QAOA away from long edges.
    """
    threshold = fuel_tank / n
    # Extra penalty on long edges (depot->i, i->j, i->depot)
    depot = n
    # depot -> first
    for i in range(n):
        if D[depot][i] > threshold:
            extra = (D[depot][i] - threshold) * penalty
            add_qubo_term(Q, var_index(i, 0, n), var_index(i, 0, n), extra)
    # consecutive customers
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j and D[i][j] > threshold:
                    extra = (D[i][j] - threshold) * penalty
                    add_qubo_term(Q, var_index(i, t, n),
                                  var_index(j, t + 1, n), extra)
    # last -> depot
    for i in range(n):
        if D[i][depot] > threshold:
            extra = (D[i][depot] - threshold) * penalty
            add_qubo_term(Q, var_index(i, n - 1, n),
                          var_index(i, n - 1, n), extra)


def build_stage2_qubo():
    """Distance + assignment penalty + fuel-overrun bias."""
    D = build_distance_matrix(SHOP, CUSTOMERS)
    Q = {}
    assignment_strength = float(D.max()) * 4.0
    assignment_penalty(Q, N, assignment_strength)
    distance_cost(Q, D, N)
    # Fuel overrun penalty: amplify long-edge cost when distance approaches tank.
    fuel_pen = 5.0  # matches game's "5 points per unit over fuel" penalty rate
    fuel_overrun_penalty(Q, D, N, FUEL_TANK, fuel_pen)
    return Q, assignment_strength, fuel_pen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim", "ionq_aria", "ionq_forte"],
                        default="local")
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--out", default="stage2_results/stage2_qaoa_result.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Stage 2 — Fuel runs out (4 customers, tank={FUEL_TANK})")
    print(f"Backend: {args.backend}, p={args.p}, shots={args.shots}")
    print("=" * 70)

    # Step 1: build QUBO
    Q, asgn_str, fuel_pen = build_stage2_qubo()
    print(f"\nQUBO: {NUM_VARS} variables, {len(Q)} terms")
    print(f"  assignment penalty = {asgn_str:.2f}")
    print(f"  fuel overrun rate  = {fuel_pen:.2f}/unit")
    print(f"  fuel tank          = {FUEL_TANK}")

    # Step 2: brute-force the QUBO + the real game scoring
    bf_route, bf_energy = brute_force_qubo_optimum(Q, NUM_VARS, N)
    bf_score = game_score(bf_route, SHOP, CUSTOMERS,
                          constraints=True, fuel_tank=FUEL_TANK)
    print(f"\nBrute-force QUBO optimum:")
    print(f"  route: {bf_route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in bf_route)})")
    print(f"  QUBO energy: {bf_energy:.3f}")
    print(f"  game score:  {bf_score['score']} "
          f"(hot={bf_score['hot']}, cold={bf_score['cold']}, "
          f"fuel={bf_score['fuel']:.1f}, overFuel={bf_score['overFuel']})")

    # Step 3: tune QAOA params
    best_params = optimise_qaoa(Q, NUM_VARS, p=args.p, shots=args.shots,
                                max_iter=args.max_iter)
    print(f"  best params: {best_params}")

    # Step 4: build final circuit
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

    # Step 5: run
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

    # Step 6: decode
    route, energy, hits, total, valid_frac = best_valid_route_from_counts(
        counts, Q, NUM_VARS, N
    )

    print(f"\n--- RESULTS ---")
    print(f"valid permutations: {valid_frac*100:.1f}% of {total} shots")
    if route is None:
        print("No valid permutation observed.")
        return

    score = game_score(route, SHOP, CUSTOMERS,
                       constraints=True, fuel_tank=FUEL_TANK)
    print(f"best valid route: {route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in route)})")
    print(f"  QUBO energy: {energy:.3f} (measured {hits}/{total} times)")
    print(f"  game score:  {score['score']} "
          f"(hot={score['hot']}, cold={score['cold']}, "
          f"fuel={score['fuel']:.1f}, overFuel={score['overFuel']})")

    if energy <= bf_energy + 1e-6:
        print(f"\n✓ QAOA matched the brute-force QUBO optimum.")
    else:
        print(f"\n  QUBO gap: {energy - bf_energy:.3f} above optimum.")
    if score['score'] == bf_score['score']:
        print(f"✓ Game-score result matches brute force ({score['score']}).")

    # Step 7: write cache
    write_cache(args.out,
                stage_id=2,
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
                       "game_cold": score["cold"],
                       "fuel_used": round(score["fuel"], 2),
                       "fuel_tank": FUEL_TANK,
                       "over_fuel": score["overFuel"]})


if __name__ == "__main__":
    main()
