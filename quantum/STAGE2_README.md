# Stage 2 added — quantum/ folder update

Adds Stage 2 ("Fuel runs out") to the pipeline. Reuses the same `qubo_builder.py`
shared module that Stage 1 introduced.

## New file

```
quantum/
├── stage2_qaoa.py         NEW (thin per-stage script for fuel constraint)
└── stage2_results/        NEW (created on first run)
```

## What's new in the QUBO for Stage 2

Stage 1 added a linear cold-pizza penalty. Stage 2 adds a **fuel overrun bias**:

For each edge in the route (depot→first, consecutive customers, last→depot),
if the edge distance exceeds `fuel_tank / N` (the per-stop budget), an extra
penalty proportional to the excess gets added to the QUBO term for that edge.

This biases QAOA toward shorter routes without needing slack variables.

## Run order

```bash
cd quantum
source .venv/bin/activate

# 1. Local test (free, ~90 sec)
python stage2_qaoa.py --backend local

# 2. IonQ cloud simulator (free, queue 1–10 min)
python stage2_qaoa.py --backend ionq_sim

# 3. Real hardware
python stage2_qaoa.py --backend ionq_aria    # ~$30–60
python stage2_qaoa.py --backend ionq_forte   # ~$80–180
```

## What you should see

Brute-force section will print:
```
Brute-force QUBO optimum:
  route: <one of several tied optima>
  QUBO energy: <some negative number>
  game score: 250 (hot=3, cold=1, fuel=~106 to ~134, overFuel=False)
```

**Stage 2's true optimum is 3 hot, 1 cold, no fuel overrun.** The game's
own description says "Deadlines are generous" but actually one customer is
always too far to reach in 80 min. Many routes tie at score 250.

QAOA should return any route with `game score: 250`.

## Two stages of fits-Forte-directly batch are now done

Remaining in the easy batch:
- **Stage 4** (VIP priority, 6 customers, 36 qubits, borderline on Forte)

Then we move to harder stages (3, 5, 6) which need multi-van or stochastic
scaffolding.
