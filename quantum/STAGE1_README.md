# Stage 1 added — quantum/ folder update

This drop adds Stage 1 ("Pizza goes cold") and a shared module so future
stages stop duplicating Stage 0's pipeline code.

## New files

```
quantum/
├── qubo_builder.py        NEW (shared module: QUBO, QAOA, backends, scoring)
├── stage0_qaoa.py         existing (kept as-is; still works standalone)
├── stage1_qaoa.py         NEW (thin per-stage script)
├── stage1_results/        NEW (created on first run)
│   └── stage1_qaoa_result.json
└── ...
```

The shared module is imported by per-stage scripts. Stage 0's existing
standalone script is untouched, so previous results stay reproducible.

## What's new in the QUBO for Stage 1

Stage 0 used only TSP distance + assignment penalty. Stage 1 adds a third
term: a linear cold-pizza penalty.

For each customer i and position t, if the best-case arrival time at
position t already exceeds customer i's `hotBy` deadline, that
(i, t) assignment gets a positive penalty added to the diagonal of the QUBO.

This is a *linear* approximation — the real game score depends on the full
route order, which would require quadratic or cubic QUBO terms. The QAOA
output is *biased toward* respecting deadlines, and we then evaluate the
final returned route against the game's actual scoring before caching.

## Run order

```bash
cd quantum
source .venv/bin/activate

# 1. Local test (free, ~60 sec)
python stage1_qaoa.py --backend local

# 2. IonQ cloud simulator (free, queue 1–10 min)
python stage1_qaoa.py --backend ionq_sim

# 3. Real hardware (costs IonQ credits)
python stage1_qaoa.py --backend ionq_aria    # ~$30–60
python stage1_qaoa.py --backend ionq_forte   # ~$80–180
```

## What you should see

The brute-force section will print something like:

```
Brute-force QUBO optimum:
  route: [0, 2, 1, 3] (Ria-Bee-Adam-Luna)
  QUBO energy: <some negative number>
  game score (real scoring): 100 (hot=2, cold=2)
```

This is correct: **no route can deliver all four hot** with these deadlines.
The optimum is 2 hot, 2 cold, and there are six routes that tie at score 100.
QAOA may surface any of those six.

QAOA matching the brute-force optimum is the success criterion.

## When to stop tuning and submit to hardware

Run `--backend local` first. Look for valid-permutation fraction > 5% and
QAOA matching brute force. If both hold, the pipeline is good for IonQ.
If valid fraction stays under 2% even at `--p 3 --max-iter 80`, raise the
assignment penalty in `build_stage1_qubo()` (`build_distance_matrix(...).max() * 4.0` 
→ multiply by 6.0 or 8.0).

## Next stages after Stage 1 works on Forte

Stage 2 (fuel limit) and Stage 4 (VIP priority) are the next two that fit
Forte directly. Same pattern: one thin script per stage, reusing
`qubo_builder.py`. Stages 3, 5, 6 need multi-van or stochastic scaffolding
that's a larger separate piece of work.
