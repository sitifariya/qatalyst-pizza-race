# Qatalyst Pizza Race — Quantum Layer (Stage 0)

This folder holds the **real quantum hardware** code for the Pizza Race game.
The game itself plays back cached results in the browser. The pipeline here
produces those cached results by running QAOA on IonQ Forte.

## What this script does

For **Stage 0** (3 customers, 1 van, no constraints) it:

1. Builds a TSP QUBO with 9 binary variables (one per customer-position pair).
2. Brute-force-verifies the QUBO matches the game's distance scoring.
3. Tunes QAOA parameters (gamma, beta) using the local Aer simulator.
4. Submits the final circuit to **local | IonQ simulator | IonQ Forte hardware**.
5. Decodes the best-valid bitstring to a route.
6. Writes a JSON cache file the game can consume.

## Why Stage 0 first?

Stage 0 is the smallest problem in the game: 3 customers means 9 qubits, well
inside IonQ Forte's 36-qubit limit. The optimal route is known (Ria-Luna-Adam,
distance ≈ 9.48). Running this first proves the whole pipeline end-to-end
before scaling to bigger stages.

## Setup

### 1. Python environment

```bash
cd qatalyst-pizza-race/quantum   # wherever you put this folder
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

### 2. IonQ API key

Sign up at https://cloud.ionq.com. Go to Settings → API Keys → Generate.

Copy `.env.example` to `.env` and paste your key in:

```bash
cp .env.example .env
nano .env   # or any editor
```

Or just export the variable in your shell:

```bash
export IONQ_API_KEY="your-actual-key-here"
```

## Run

### Step 1 — Local test (free, 30 seconds)

Verifies the entire pipeline works locally with the Aer simulator before
spending any IonQ credits.

```bash
python stage0_qaoa.py --backend local
```

Expected output: QAOA finds the optimal route (Ria-Luna-Adam) with energy
~9.48. Writes `stage0_results/stage0_qaoa_result.json`.

### Step 2 — IonQ simulator (free, queue time 1–10 min)

Runs on IonQ's cloud-hosted ideal simulator. Same result as local but proves
your IonQ credentials work.

```bash
python stage0_qaoa.py --backend ionq_sim
```

### Step 3 — IonQ Aria QPU (cheap, real hardware)

25-qubit trapped-ion machine. Roughly $20–50 for one full run (1024 shots).
This is the safe-and-cheap first hardware run.

```bash
python stage0_qaoa.py --backend ionq_aria --shots 1024
```

### Step 4 — IonQ Forte QPU (the real target)

36-qubit machine, higher fidelity. Roughly $50–150 for one full run.

```bash
python stage0_qaoa.py --backend ionq_forte --shots 1024
```

### Tuning

- `--p 2` runs QAOA depth 2 (better solution quality, more gates, more cost).
- `--max-iter 50` more parameter-search iterations on the local simulator
  (free — only affects local time).
- `--shots 2048` more shots on hardware (better statistics, costs more).

## Output

After a successful run you get `stage0_results/stage0_qaoa_result.json`:

```json
{
  "stage_id": 0,
  "route": [0, 2, 1],
  "backend": "ionq_forte",
  "energy": 9.479,
  "valid_fraction": 0.31,
  "total_shots": 1024,
  "timestamp_utc": "2026-05-13T20:00:00+00:00",
  "num_qubits": 9,
  "encoding": "TSP one-hot, N*N variables, p=1 QAOA",
  "customer_names": ["Ria", "Adam", "Luna"]
}
```

The `route` field is what feeds back into the game.

## Plugging it back into the game

Once you have a hardware result, update Stage 0 in `js/stages.js`:

```js
0: {
  num:'STAGE 0', title:'First delivery', ...
  // existing fields
  cachedQuantum: [0, 2, 1],  // <-- paste from JSON result.route
  cachedQuantumSource: 'IonQ Forte, 2026-05-13'  // (optional, for the explainer)
},
```

Then for the game to actually surface it, you'd also wire the short-circuit
in `solvers.js` (the same one already used for Cluster 3 stages). Right now
Stage 0 is small enough that the live in-browser brute-force returns the same
answer in ~1 ms, so caching is optional, but having the JSON gives you the
real-hardware-attestation for the whitepaper.

## Cost expectations (May 2026 pricing, approximate)

| Backend         | Per-run cost | Notes |
|-----------------|--------------|-------|
| local           | $0           | Aer simulator |
| ionq_sim        | $0           | Free tier, queue can be slow |
| ionq_aria       | ~$20–50      | 25 qubits, older system |
| ionq_forte      | ~$50–150     | 36 qubits, the headline machine |

Multiply by your number of stages and depth-`p` values for budgeting. Stage 0
alone, run once on Forte, is about the cheapest experiment in the project.

## Next steps after Stage 0 works

1. **Stage 1** (4 customers, time windows): add cold-pizza penalty to the QUBO.
2. **Stage 2** (4 customers, fuel limit): add fuel constraint.
3. **Stages 3–5**: extend the QUBO to multi-van and VIP penalties.
4. **Stage 6** (10 customers, stochastic): scenario-based QUBO; this is the
   "quantum starts to matter" stage in Cluster 2.
5. **Stages 7–8**: hybrid decomposition (cluster geographically, solve each
   sub-cluster on Forte, recombine classically).
6. **Stages 9–11**: future hardware. SA-cached forever, not run on real qubits.

## Troubleshooting

- **"No valid permutation in samples"**: QAOA didn't converge well. Try
  `--p 2 --max-iter 50 --shots 2048` or increase the penalty strength in
  `build_qubo()`.
- **IonQ job hangs**: check https://cloud.ionq.com for queue position and
  device availability.
- **"qiskit-ionq not installed"**: `pip install qiskit-ionq`.
