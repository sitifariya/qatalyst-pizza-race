# Stage 3 — The Split — Evidence and Findings

**Status:** Encoding complete, execution beyond reach of accessible backends.

**Date:** 18 May 2026

## What was attempted

Stage 3 of the Qatalyst Pizza Race is the first multi-van problem in the
game: 6 customers must be split between 2 vans, each with a 28-unit fuel
tank, with time-window deadlines on each delivery.

The natural QUBO encoding uses one-hot variables

```
x[i, t, v] = 1 if customer i is at position t in van v
```

with **N × N × V = 6 × 6 × 2 = 72 logical qubits**.

We built this encoding (see `qubo_builder_multivan.py`) with four penalty
families:

| Family | Purpose | Strength |
|--------|---------|----------|
| `assignment_penalty_mv` | Each customer appears exactly once across all (t, v) slots; each slot holds at most one customer | 320.93 |
| `distance_cost_mv` | TSP distance per van, depot → first → … → last → depot | distance-scaled |
| `cold_penalty_mv` | Linear penalty when best-case arrival exceeds customer's hotBy deadline | 106.98 |
| `fuel_overrun_penalty_mv` | Amplifies long edges proportional to fuel-tank pressure | 5.0 per unit over |

Total QUBO size: **948 quadratic terms over 72 binary variables**.

## Brute-force reference

Exhaustive enumeration over all 5040 (partition × per-van ordering)
candidates confirms the optimum:

- **Van 1:** Bee → Luna → Ria  (distance 77.7, fuel 77.7, over by 49.7)
- **Van 2:** Kai → Jay → Adam  (distance 76.5, fuel 76.5, over by 48.5)
- **Score:** −191.0  (4 hot, 2 cold, both vans over fuel)

Stage 3 is a constraint-violation-required problem: every valid
combinatorial assignment overruns the 28-unit fuel tank. The optimum
minimises the overrun cost rather than satisfying the constraint.

## Execution attempts

### Attempt 1 — Local Aer simulator
```
python stage3_qaoa.py --backend local --p 1 --max-iter 10 --shots 1024
```
Outcome:
```
ERROR: Insufficient memory to run circuit using the statevector simulator.
Required memory: 72057594037927936 MB, max memory: 8192 MB
```
The Aer statevector simulator needs a 2^72-dimensional state vector. Local
hardware (8 GB RAM) cannot allocate this; nor can any laptop, ever.

### Attempt 2 — IonQ Cloud simulator (skip-tuning, fixed γ/β)
```
python stage3_qaoa.py --backend ionq_sim --skip-tuning --p 1 --shots 4096
```

The circuit (72 qubits, depth 160, 2916 gates) was successfully prepared
locally and submitted to IonQ Cloud:

- **Job ID:** `019e3d07-964b-7288-99a0-846fe0fc2822`
- **Submission accepted:** yes
- **Execution result:** `IonQJobFailureError: NotEnoughQubits`

The IonQ Cloud simulator enforces a qubit ceiling below 72.

### Attempt 3 — IonQ Forte (qpu.forte)
Not attempted. Forte has 36 qubits, well below the encoding's 72.

## What this finding means

For the standard one-hot TSP-VRP encoding, **multi-van problems at N = 6
exceed every currently-accessible quantum or quantum-simulating backend**.
This is consistent with the pedagogical framing of the game's Cluster 3
("Quantum tomorrow"), which uses pre-computed classical results because
the hardware to execute multi-van VRP at scale does not yet exist.

Three honest paths exist beyond this finding:

1. **Wait for hardware.** IonQ's roadmap suggests Tempo (100 qubits, late
   2026) and follow-on systems will fit problems at this scale. Stage 3
   could be revisited then.
2. **Encoding tricks.** Tighter encodings (log-encoding, slack reduction,
   permutation matrices) can reduce the qubit count, but typically by
   constant factors rather than orders of magnitude. The structural
   barrier remains.
3. **Hybrid decomposition.** Classical clustering (e.g. Dantzig–Wolfe)
   splits the multi-van problem into two single-van sub-problems, each
   ~18 qubits, each well inside Forte's 36-qubit limit. This is the
   approach used by commercial quantum-routing pipelines.

Option 3 is the realistic next step and is left as future work.

## Code artefacts in this folder

```
quantum/
├── qubo_builder_multivan.py    Multi-van extensions to the shared QUBO module
├── stage3_qaoa.py              Pipeline script (--skip-tuning flag for cloud sub)
├── STAGE3_README.md            User-facing operating instructions
└── STAGE3_EVIDENCE.md          THIS FILE
```

The IonQ job ID (`019e3d07-964b-7288-99a0-846fe0fc2822`) is recorded for
reproducibility; querying the IonQ Cloud Jobs page will return the
`NotEnoughQubits` outcome.

## Citation snippet (for the whitepaper)

> Stage 3 (6 customers, 2 vans, time windows, fuel limit) was encoded as a
> 72-qubit QUBO with 948 quadratic terms. Brute-force enumeration over
> 5040 candidate partition-orderings established a reference optimum of
> −191 game-score (4 hot, 2 cold; both vans over fuel by ≈ 50 units).
> Execution was attempted on the local Aer statevector simulator
> (out-of-memory) and on the IonQ Cloud simulator (rejected with
> `NotEnoughQubits`, job ID `019e3d07-964b-7288-99a0-846fe0fc2822`,
> 18 May 2026). The 72-qubit one-hot encoding exceeds all backends
> currently accessible to this project; multi-van VRP at this scale
> motivates either (a) future devices in the 100+-qubit class or
> (b) hybrid classical-quantum decomposition into single-van sub-problems
> that fit a 36-qubit device such as IonQ Forte.
