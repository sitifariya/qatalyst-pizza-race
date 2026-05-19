"""
Stage 4_q — VIP rush (research companion to game's Stage 4).

This is the Forte-compatible variant of the game's Stage 4.
The game uses N=7 customers for engaging gameplay; this research-grade
companion uses N=5 customers so the encoding fits real quantum hardware.

What's new vs Stage 2:
  - 5 customers (25 qubits, fits IonQ Forte's 36-qubit limit)
  - Two VIP customers (Mayor, Celeb) with the same mechanic as the game:
    - Cold pizza penalty: 50 points (regular) or 150 points ADDITIONAL (VIP)
    - VIP cold = -200 total points (matches the game's 4x penalty)
  - Time windows + fuel tank (combined Stage 1 + Stage 2 mechanics)

QUBO terms added on top of qubo_builder primitives:
  - vip_cold_penalty: scales the linear cold penalty by 4x for VIP customers

Usage:
    python stage4q_qaoa.py --backend local              # ~30 sec
    python stage4q_qaoa.py --backend ionq_sim           # 1-10 min queue
    python stage4q_qaoa.py --backend ionq_forte         # real hardware
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qubo_builder import (
    build_distance_matrix,
    assignment_penalty,
    distance_cost,
    cold_penalty,
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
# STAGE 4_Q PROBLEM (research companion to game's Stage 4)
# ----------------------------------------------------------------------------
SHOP = (350, 200)
CUSTOMERS = [
    {"id": 0, "name": "Mayor", "x": 290, "y": 170, "hotBy": 25, "vip": True},
    {"id": 1, "name": "Adam",  "x": 560, "y": 140, "hotBy": 50},
    {"id": 2, "name": "Celeb", "x": 400, "y": 260, "hotBy": 30, "vip": True},
    {"id": 3, "name": "Bee",   "x": 180, "y": 310, "hotBy": 55},
    {"id": 4, "name": "Ria",   "x": 520, "y": 320, "hotBy": 48},
]
FUEL_TANK = 80.0
N = len(CUSTOMERS)
NUM_VARS = N * N  # 25 qubits


def vip_cold_penalty(Q, customers, D, n, vip_extra_weight):
    """Extra cold-penalty for VIP customers.

    For each VIP customer i and position t, if best-case arrival exceeds
    deadline, add a *3x* additional penalty (on top of the regular cold
    penalty already added by cold_penalty). This makes a VIP cold worth
    4x a regular cold, matching the game's scoring.
    """
    import numpy as np
    depot = n
    avg_step = float(D[np.triu_indices(n + 1, 1)].mean())
    min_first_leg = min(D[depot][k] for k in range(n))
    for i in range(n):
        if not customers[i].get("vip"):
            continue
        deadline = customers[i].get("hotBy", None)
        if deadline is None:
            continue
        for t in range(n):
            best_case = min_first_leg + t * avg_step
            if best_case > deadline:
                # extra weight for VIPs only
                add_qubo_term(Q, var_index(i, t, n), var_index(i, t, n),
                              vip_extra_weight)


def fuel_overrun_penalty(Q, D, n, fuel_tank, penalty):
    """Same as Stage 2: amplify long edges when fuel-pressure is high."""
    threshold = fuel_tank / n
    depot = n
    for i in range(n):
        if D[depot][i] > threshold:
            extra = (D[depot][i] - threshold) * penalty
            add_qubo_term(Q, var_index(i, 0, n), var_index(i, 0, n), extra)
    for t in range(n - 1):
        for i in range(n):
            for j in range(n):
                if i != j and D[i][j] > threshold:
                    extra = (D[i][j] - threshold) * penalty
                    add_qubo_term(Q, var_index(i, t, n),
                                  var_index(j, t + 1, n), extra)
    for i in range(n):
        if D[i][depot] > threshold:
            extra = (D[i][depot] - threshold) * penalty
            add_qubo_term(Q, var_index(i, n - 1, n),
                          var_index(i, n - 1, n), extra)


def build_stage4q_qubo():
    """Distance + assignment + cold + VIP-extra cold + fuel overrun."""
    D = build_distance_matrix(SHOP, CUSTOMERS)
    Q = {}
    assignment_strength = float(D.max()) * 4.0
    assignment_penalty(Q, N, assignment_strength)
    distance_cost(Q, D, N)
    cold_weight = float(D.max()) * 2.0
    cold_penalty(Q, CUSTOMERS, D, N, cold_weight)
    # VIP gets 3x EXTRA on top of regular cold penalty → total 4x for VIPs
    vip_cold_penalty(Q, CUSTOMERS, D, N, cold_weight * 3.0)
    fuel_pen = 5.0
    fuel_overrun_penalty(Q, D, N, FUEL_TANK, fuel_pen)
    return Q, assignment_strength, cold_weight, fuel_pen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=["local", "ionq_sim", "ionq_aria", "ionq_forte"],
                        default="local")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--out", default="stage4q_results/stage4q_qaoa_result.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Stage 4_q — VIP rush research companion (5 cust, 2 VIPs, tank={FUEL_TANK})")
    print(f"Backend: {args.backend}, p={args.p}, shots={args.shots}")
    print(f"** {NUM_VARS} qubits — fits IonQ Forte (36 qubit limit) **")
    print("=" * 70)

    # Step 1: build QUBO
    Q, asgn_str, cold_w, fuel_pen = build_stage4q_qubo()
    print(f"\nQUBO: {NUM_VARS} variables, {len(Q)} terms")
    print(f"  assignment penalty   = {asgn_str:.2f}")
    print(f"  regular cold penalty = {cold_w:.2f}")
    print(f"  VIP extra penalty    = {cold_w * 3.0:.2f}  (total VIP cold = 4x regular)")
    print(f"  fuel overrun rate    = {fuel_pen:.2f}/unit")
    print(f"  fuel tank            = {FUEL_TANK}")

    # Step 2: brute-force QUBO + game scoring
    bf_route, bf_energy = brute_force_qubo_optimum(Q, NUM_VARS, N)
    bf_score = game_score(bf_route, SHOP, CUSTOMERS,
                          constraints=True, fuel_tank=FUEL_TANK)
    print(f"\nBrute-force QUBO optimum:")
    print(f"  route: {bf_route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in bf_route)})")
    print(f"  QUBO energy: {bf_energy:.3f}")
    print(f"  game score:  {bf_score['score']} "
          f"(hot={bf_score['hot']}, cold={bf_score['cold']}, "
          f"VIP cold={bf_score['vipCold']}, fuel={bf_score['fuel']:.1f})")

    # Step 3: tune QAOA params (local Aer can handle 25 qubits fine)
    best_params = optimise_qaoa(Q, NUM_VARS, p=args.p, shots=args.shots,
                                max_iter=args.max_iter)
    print(f"  best params: {best_params}")

    # Step 4: final circuit
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
    print(f"valid permutations: {valid_frac*100:.2f}% of {total} shots")
    if route is None:
        print("No valid permutation found.")
        return

    score = game_score(route, SHOP, CUSTOMERS,
                       constraints=True, fuel_tank=FUEL_TANK)
    print(f"best valid route: {route} "
          f"({'-'.join(CUSTOMERS[i]['name'] for i in route)})")
    print(f"  QUBO energy: {energy:.3f} (measured {hits}/{total} times)")
    print(f"  game score:  {score['score']} "
          f"(hot={score['hot']}, cold={score['cold']}, "
          f"VIP cold={score['vipCold']}, fuel={score['fuel']:.1f})")

    if energy <= bf_energy + 1e-6:
        print(f"\n✓ QAOA matched brute-force QUBO optimum.")
    else:
        print(f"\n  QUBO gap: {energy - bf_energy:.3f}")
    if score['score'] == bf_score['score']:
        print(f"✓ Game-score result matches brute force ({score['score']}).")
    if score['vipCold'] == 0:
        print(f"✓ All VIPs delivered hot.")

    # Step 7: write cache
    write_cache(args.out,
                stage_id="4_q",
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
                       "vip_cold": score["vipCold"],
                       "fuel_used": round(score["fuel"], 2),
                       "fuel_tank": FUEL_TANK,
                       "over_fuel": score["overFuel"],
                       "companion_to": "game Stage 4 (VIP rush, N=7)",
                       "research_note":
                           "N=5 selected to fit IonQ Forte's 36-qubit limit "
                           "while preserving the VIP mechanic (4x cold penalty)."})


if __name__ == "__main__":
    main()
