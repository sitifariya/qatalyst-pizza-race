# Stage 3 added — multi-van extension to the QAOA pipeline

Stage 3 ("The split") is the first multi-van stage. 6 customers split across
2 vans, each with its own fuel tank.

## New files

```
quantum/
├── qubo_builder_multivan.py    NEW (multi-van extension to qubo_builder)
├── stage3_qaoa.py              NEW (per-stage script)
└── stage3_results/             NEW (created on first run)
```

## Qubit count: 72

Variables: x[i,t,v] = 1 if customer i visits position t in van v.

6 customers × 6 positions × 2 vans = **72 qubits**.

This is **well beyond IonQ Forte's 36-qubit limit**. Stage 3 runs on:

- The local Aer simulator (free, slow for 72 qubits — minutes per iter)
- The IonQ Cloud noiseless simulator (free, queue 1-15 min)

It will NOT run on Forte. Don't try `--backend ionq_forte` — it will fail.

## Run order

Start small to verify the pipeline:

```bash
cd quantum
source .venv/bin/activate
python stage3_qaoa.py --backend local --p 1 --max-iter 10 --shots 1024
```

That's about 5-10 minutes. If it works:

```bash
python stage3_qaoa.py --backend local --p 2 --max-iter 30 --shots 4096
```

That's 30-60 minutes. Better answers, much longer wait.

Cloud simulator:

```bash
python stage3_qaoa.py --backend ionq_sim --p 2 --max-iter 30 --shots 4096
```

The local tuning still runs locally; only the final circuit gets submitted to IonQ.

## Brute force reference

The script prints the brute-force optimum (over 5040 partition+ordering
candidates). The Stage 3 game has tight fuel — every solution overruns the
28-unit tank. Optimum is:

- Van 1: Bee → Luna → Ria
- Van 2: Kai → Jay → Adam
- Score: -191 (4 hot, 2 cold, both vans over fuel)

QAOA's job is to find this (or another route with similar score).

## Valid-fraction expectations

Stages 0-2 had valid-permutation fractions of 0.5%-3%. At 72 qubits with
multi-van constraints, valid fractions will be **far smaller** (likely
0.001%-0.01%). Expect to need:

- 4096+ shots
- p=2 or higher
- Possibly multiple runs to surface a valid configuration

If no valid assignment surfaces, the script prints specific suggestions.

## Stage 3 is the most expensive stage to optimise

If you find local runs are taking too long, consider:

1. **Lower max_iter:** even max_iter=10 will give a usable result; the
   parameter landscape isn't very sensitive at p=1 for problems this size.
2. **Hybrid decomposition instead** (separate per-van QUBOs of ~18 qubits
   each, both fit Forte). That's a different technique we'd build as a
   separate per-stage script later.

## Honest framing for the whitepaper

Stage 3 demonstrates that multi-van VRP doesn't fit current quantum
hardware. The cached "quantum" result the game shows for Stage 3 is from
classical SA, not real quantum hardware. This stage is in the game to
motivate why decomposition matters for real industrial routing.
