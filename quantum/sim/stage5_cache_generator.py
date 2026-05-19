"""
stage5_cache_generator.py — Generate cached routes for Stage 5 ("The maybe rush")
================================================================================

Stage 5 is a stochastic routing problem:
  - 6 confirmed customers (definitely order)
  - 3 maybe customers (order with probability 70%, 50%, 30%)

The game shows the player 3 pre-computed route options:
  1. "Cautious"   — skip all maybes, visit only confirmed customers
  2. "Aggressive" — include all maybes, optimized assuming they all order
  3. "Smart"      — include all maybes, ordered to maximize EXPECTED score
                   across many scenario samples (this is what quantum does)

This script computes all three routes and outputs a JSON snippet ready to
paste into stages.js as the stage's cachedRoutes field.

Usage:
    python stage5_cache_generator.py

Output:
    stage5_results/stage5_routes.json
"""
import math
import random
import json
import os
from itertools import permutations
from datetime import datetime, timezone


# ============================================================================
# PROBLEM SETUP
# ============================================================================
SHOP = (350, 200)

CONFIRMED = [
    {"id": 0, "name": "Ria",  "x": 220, "y": 150, "hotBy": 30},
    {"id": 1, "name": "Adam", "x": 480, "y": 150, "hotBy": 30},
    {"id": 2, "name": "Luna", "x": 250, "y": 260, "hotBy": 45},
    {"id": 3, "name": "Jay",  "x": 450, "y": 260, "hotBy": 45},
    {"id": 4, "name": "Bee",  "x": 350, "y": 120, "hotBy": 22},
    {"id": 5, "name": "Kai",  "x": 350, "y": 290, "hotBy": 50},
]

MAYBE = [
    {"id": 6, "name": "Leo",  "x": 120, "y": 200, "hotBy": 45, "prob": 0.7},
    {"id": 7, "name": "Zara", "x": 580, "y": 200, "hotBy": 45, "prob": 0.5},
    {"id": 8, "name": "Nia",  "x": 350, "y": 340, "hotBy": 60, "prob": 0.3},
]

ALL_CUSTOMERS = CONFIRMED + MAYBE
N_CONFIRMED = len(CONFIRMED)
N_TOTAL = len(ALL_CUSTOMERS)

# Scoring constants (per delivery cycle)
HOT_REWARD = 100        # points per hot delivery
COLD_PENALTY = 50       # points per cold delivery
MISSED_PENALTY = 150    # points per maybe customer who ordered but wasn't routed
FUEL_RATE = 0.5         # points per fuel unit

# Number of Monte Carlo scenario samples used to evaluate expected scores
N_SCENARIOS = 5000


# ============================================================================
# GEOMETRY
# ============================================================================
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 10.0


# ============================================================================
# SCORING
# ============================================================================
def evaluate_route(route, scenario_present):
    """Evaluate a route under a specific scenario.

    Args:
        route: list of customer ids in visit order
        scenario_present: dict {customer_id: True/False} for maybe customers
                         (confirmed customers always present)

    Returns:
        dict with hot, cold, missed, fuel
    """
    t = 0.0
    prev = SHOP
    hot = 0
    cold = 0
    fuel = 0.0

    for ci in route:
        c = ALL_CUSTOMERS[ci]
        leg = distance(prev, (c["x"], c["y"]))
        t += leg
        fuel += leg
        prev = (c["x"], c["y"])

        # Confirmed always present, maybe depends on scenario
        is_present = True if c["id"] < N_CONFIRMED else scenario_present.get(c["id"], False)

        if is_present:
            if t > c["hotBy"]:
                cold += 1
            else:
                hot += 1

    # Return to shop
    fuel += distance(prev, SHOP)

    # Missed orders: maybe customers who DID order but aren't in the route
    in_route = set(route)
    missed = sum(
        1 for c in MAYBE
        if scenario_present.get(c["id"], False) and c["id"] not in in_route
    )

    return {"hot": hot, "cold": cold, "missed": missed, "fuel": fuel}


def score_from_result(result):
    """Convert (hot, cold, missed, fuel) into a game score."""
    return (
        result["hot"] * HOT_REWARD
        - result["cold"] * COLD_PENALTY
        - result["missed"] * MISSED_PENALTY
        - result["fuel"] * FUEL_RATE
    )


def expected_score(route, n_samples=N_SCENARIOS, seed=42):
    """Monte Carlo expected score across stochastic scenarios.

    Returns (mean, max, min, std).
    """
    random.seed(seed)
    scores = []
    for _ in range(n_samples):
        scenario = {
            c["id"]: random.random() < c["prob"]
            for c in MAYBE
        }
        result = evaluate_route(route, scenario)
        scores.append(score_from_result(result))

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    return mean, max(scores), min(scores), std


# ============================================================================
# ROUTE OPTIMIZERS
# ============================================================================
def best_order_by_fuel(customer_ids):
    """Find the visit order minimizing total fuel (classical TSP)."""
    if not customer_ids:
        return []
    best = None
    best_fuel = float("inf")
    for perm in permutations(customer_ids):
        t = 0.0
        prev = SHOP
        for ci in perm:
            c = ALL_CUSTOMERS[ci]
            t += distance(prev, (c["x"], c["y"]))
            prev = (c["x"], c["y"])
        t += distance(prev, SHOP)
        if t < best_fuel:
            best_fuel = t
            best = list(perm)
    return best


def best_order_assuming_all_present(customer_ids):
    """Find the visit order maximizing (hot, -fuel) assuming all maybes order.

    This is the "Aggressive" strategy: optimize as if every uncertain customer
    confirmed. Plays well in scenarios where all maybes order, poorly when many
    don't.
    """
    if not customer_ids:
        return []
    best = None
    best_score = (-1, float("inf"))  # (max_hot, min_fuel)
    for perm in permutations(customer_ids):
        t = 0.0
        prev = SHOP
        cold = 0
        fuel = 0.0
        for ci in perm:
            c = ALL_CUSTOMERS[ci]
            leg = distance(prev, (c["x"], c["y"]))
            t += leg
            fuel += leg
            if t > c["hotBy"]:
                cold += 1
            prev = (c["x"], c["y"])
        fuel += distance(prev, SHOP)
        hot = len(perm) - cold
        if hot > best_score[0] or (hot == best_score[0] and fuel < best_score[1]):
            best_score = (hot, fuel)
            best = list(perm)
    return best


def best_order_by_expected_value(customer_ids, n_samples=2000, seed=0):
    """Find the visit order maximizing expected score across scenarios.

    This is the "Smart" strategy. Stand-in for what quantum's superposition
    naturally explores: routes that perform well on average across all
    possible scenario draws.

    Searches over a sample of random permutations (full enumeration would
    be 9! = 362880, manageable but slow; we sample for efficiency).
    """
    if not customer_ids:
        return []

    n = len(customer_ids)

    # For small clusters (≤7), brute force is fine
    if n <= 7:
        candidates = list(permutations(customer_ids))
    else:
        # Sample 10,000 random permutations for n=8,9
        random.seed(seed)
        candidates = []
        for _ in range(10000):
            shuffled = list(customer_ids)
            random.shuffle(shuffled)
            candidates.append(tuple(shuffled))
        # Always include the all-fuel-optimal route as a baseline
        candidates.append(tuple(best_order_by_fuel(customer_ids)))

    # Pre-generate fixed scenarios so the comparison is fair
    random.seed(seed)
    fixed_scenarios = [
        {c["id"]: random.random() < c["prob"] for c in MAYBE}
        for _ in range(n_samples)
    ]

    def quick_expected(route):
        total = 0
        for sc in fixed_scenarios:
            r = evaluate_route(list(route), sc)
            total += score_from_result(r)
        return total / len(fixed_scenarios)

    best = None
    best_e = float("-inf")
    for perm in candidates:
        e = quick_expected(perm)
        if e > best_e:
            best_e = e
            best = list(perm)

    return best


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Stage 5 — 'The maybe rush' — Cache Generator")
    print("=" * 70)
    print(f"\n6 confirmed customers: {[c['name'] for c in CONFIRMED]}")
    maybe_labels = [f"{c['name']} ({int(c['prob']*100)}%)" for c in MAYBE]
    print(f"3 maybe customers:     {maybe_labels}")

    # ----- Strategy 1: CAUTIOUS — skip all maybes -----
    print("\n[1/3] Cautious strategy (confirmed customers only)...")
    cautious_route = best_order_by_fuel([0, 1, 2, 3, 4, 5])
    cautious_exp, cautious_max, cautious_min, cautious_std = expected_score(cautious_route)
    print(f"  Route: {'-'.join(ALL_CUSTOMERS[i]['name'] for i in cautious_route)}")
    print(f"  Expected score: {cautious_exp:.1f}  (range {cautious_min:.0f} to {cautious_max:.0f}, σ={cautious_std:.1f})")

    # ----- Strategy 2: AGGRESSIVE — include all maybes optimistically -----
    print("\n[2/3] Aggressive strategy (include all maybes, optimize assuming all order)...")
    aggressive_route = best_order_assuming_all_present([0, 1, 2, 3, 4, 5, 6, 7, 8])
    aggressive_exp, aggressive_max, aggressive_min, aggressive_std = expected_score(aggressive_route)
    print(f"  Route: {'-'.join(ALL_CUSTOMERS[i]['name'] for i in aggressive_route)}")
    print(f"  Expected score: {aggressive_exp:.1f}  (range {aggressive_min:.0f} to {aggressive_max:.0f}, σ={aggressive_std:.1f})")

    # ----- Strategy 3: SMART — include all maybes, optimize expected value -----
    print("\n[3/3] Smart strategy (include all maybes, optimize expected score across scenarios)...")
    smart_route = best_order_by_expected_value([0, 1, 2, 3, 4, 5, 6, 7, 8])
    smart_exp, smart_max, smart_min, smart_std = expected_score(smart_route)
    print(f"  Route: {'-'.join(ALL_CUSTOMERS[i]['name'] for i in smart_route)}")
    print(f"  Expected score: {smart_exp:.1f}  (range {smart_min:.0f} to {smart_max:.0f}, σ={smart_std:.1f})")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("SUMMARY — Expected scores across 5000 scenario samples:")
    print("=" * 70)
    print(f"  Cautious:   {cautious_exp:>8.1f}")
    print(f"  Aggressive: {aggressive_exp:>8.1f}")
    print(f"  Smart:      {smart_exp:>8.1f}    ← should be highest")

    # ----- Sanity check -----
    if smart_exp < max(cautious_exp, aggressive_exp):
        print("\n⚠️  WARNING: Smart route does not beat Cautious/Aggressive on expectation.")
        print("    Tuning may be needed — review customer geometry or scoring weights.")
    else:
        improvement = smart_exp - max(cautious_exp, aggressive_exp)
        print(f"\n✓ Smart beats next-best by {improvement:.1f} expected points.")

    # ----- Output JSON for stages.js -----
    output = {
        "stage_id": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confirmed_customers": CONFIRMED,
        "maybe_customers": MAYBE,
        "scoring": {
            "hot_reward": HOT_REWARD,
            "cold_penalty": COLD_PENALTY,
            "missed_penalty": MISSED_PENALTY,
            "fuel_rate": FUEL_RATE,
        },
        "scenario_samples": N_SCENARIOS,
        "routes": [
            {
                "label": "Cautious",
                "description": "Visit only confirmed customers. Safe but misses possible orders.",
                "route": cautious_route,
                "expected_score": round(cautious_exp, 1),
                "score_range": [round(cautious_min, 1), round(cautious_max, 1)],
                "stdev": round(cautious_std, 1),
            },
            {
                "label": "Aggressive",
                "description": "Visit everyone, optimize assuming all maybes order. High variance.",
                "route": aggressive_route,
                "expected_score": round(aggressive_exp, 1),
                "score_range": [round(aggressive_min, 1), round(aggressive_max, 1)],
                "stdev": round(aggressive_std, 1),
            },
            {
                "label": "Smart",
                "description": "Visit everyone, ordered to maximize expected score across scenarios.",
                "route": smart_route,
                "expected_score": round(smart_exp, 1),
                "score_range": [round(smart_min, 1), round(smart_max, 1)],
                "stdev": round(smart_std, 1),
                "method_note": (
                    "This route is the expected-value-optimal under stochastic "
                    "customer arrival. It is what a quantum-future stochastic VRP "
                    "solver (Chiew et al., IEEE TQE 2024; Ekstrøm et al., npj QI 2026) "
                    "would naturally return via superposition sampling across scenarios."
                ),
            },
        ],
    }

    os.makedirs("stage5_results", exist_ok=True)
    out_path = "stage5_results/stage5_routes.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten: {out_path}")
    print("\nPaste the 'routes' field into stages.js as the stage's cachedRoutes.")


if __name__ == "__main__":
    main()
