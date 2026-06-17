# Qatalyst Pizza Race

An open-source benchmark and browser game for comparing classical and quantum optimisation approaches to vehicle routing and graph problems.

**Status:** Live · **Author:** Dr Siti Fariya, Qatalyst Quantum Ltd
**Funding:** Supported by Qollab and IonQ through the Qollab Creative Challenge 2026

**Play it:** [qatalyst-quantum.co.uk/play](https://qatalyst-quantum.co.uk/play)

---

## What is this?

Quantum Courier: Pizza Race is a browser game that turns combinatorial optimisation into a six-stage delivery puzzle. Players plan routes and watch a classical solver and a quantum-inspired solver race through the same problem. The game shows which solver wins, and why.

This repository contains:

- An open benchmark suite of six logistics puzzles with increasing complexity (time windows, fuel limits, multi-van, MaxCut, stochastic demand)
- A classical optimisation engine in Python with brute-force enumeration and simulated annealing solvers
- The browser game itself, with educational explainers in each stage
- A standalone Qiskit script that runs the Stage 4 MaxCut on real IonQ Forte hardware
- Benchmark logging, so every solve contributes to an empirical dataset

## Why this exists

Most quantum optimisation demos either oversell what hardware can do today or hide the cases where classical methods win. Pizza Race is built to show both sides on the same set of problems.

Two results from real IonQ Forte hardware sit at the centre of the game:

- **Stage 4 (MaxCut, 24-node graph):** On this specific instance, QAOA on Forte beat a simple classical local-search baseline. The win does not extend to stronger classical methods. Goemans-Williamson SDP rounding and other mature solvers still outperform quantum hardware on most MaxCut instances today.

- **Stage 3 (25-customer vehicle routing with time windows):** Four QAOA variants were tested on Forte. All four were beaten by classical simulated annealing. Routing solvers are decades old and handle constraints quantum circuits cannot yet express well.

Both results are in the game on purpose. Operators making real decisions need to know where quantum helps and where classical methods still win.

Read the full finding: [qatalyst-quantum.co.uk/news.html](https://qatalyst-quantum.co.uk/news.html)

## The six stages

| Stage | Problem | Who wins | Why |
|---|---|---|---|
| 0 | First delivery (tutorial, 3 customers) | Tie | Trivial scale |
| 1 | Time windows (4 customers) | Tie / classical | Small enough for both to find optimum |
| 2 | Fuel limit (4 customers) | Tie / classical | TSP at small scale |
| 3 | Hybrid routing (10 customers, 2 vans) | Classical | k-means + per-cluster QAOA. Classical SA beat all QAOA variants tested on Forte |
| 4 | Parallel kitchens (24 customers, MaxCut) | **Quantum** | QAOA on Forte beat classical local 1-step on this 24-node graph |
| 5 | Stochastic routing (6 confirmed + 3 maybe) | Classical (today) | Stochastic frontier, where 2027+ quantum hardware is expected to help |

## Repository layout

```
qatalyst-pizza-race/
├── engine/                              Classical solver API (Python + FastAPI)
├── game/                                Pizza Race browser game (HTML + JS)
├── quantum_courier_maxcut_forte.py      Standalone Stage 4 MaxCut on IonQ Forte
├── docs/                                Technical notes, benchmark data
├── LICENSE                              MIT
└── README.md                            This file
```

## Quick start

### Play the game

Live: [qatalyst-quantum.co.uk/play](https://qatalyst-quantum.co.uk/play)

The live game is deployed from a separate marketing site repository. The `game/play.html` in this repository is a reference snapshot included for academic reproducibility under the MIT license.

To run the reference snapshot locally:

```
cd game
python -m http.server 8080
# Open http://localhost:8080/play.html
```

### Run the classical engine locally

```
cd engine
pip install -r requirements.txt
python api.py
# API on http://localhost:8000
```

Health check:

```
curl http://localhost:8000/api/pizza/health
```

See `engine/README.md` for full API documentation.

### Run the Stage 4 MaxCut on real IonQ Forte hardware

Requirements:

```
pip install qiskit qiskit-ionq scipy numpy
export IONQ_API_KEY=your_key_here
```

Then:

```
python quantum_courier_maxcut_forte.py
```

In qBraid Lab the API key is already wired up. Open a notebook with the qiskit-ionq environment and paste the code.

## Forte experiment details

| Field | Value |
|---|---|
| Hardware | IonQ Forte-1 (trapped-ion) |
| Algorithm | QAOA, p=1 |
| Shots | 4,096 |
| Qubits used | 24 (one per order) |
| Graph | 24-node 3-regular, 36 edges |
| Classical baseline | Local 1-step search, ~30 cut edges (avg of 100 random starts) |
| Quantum result | 33 cut edges |
| Advantage on this graph | ~+9% vs local 1-step |
| Job ID | 019e94b0-c3cc-702c-ae74-39e13f799cb9 |

This advantage does not extend to stronger classical methods like Goemans-Williamson, which still outperform quantum hardware on most MaxCut instances at this scale.

## Why structure matters more than qubit count

Graph cutting has a cost function that counts cut edges, which aligns naturally with what a shallow parameterised quantum circuit can express. Vehicle routing does not have that property at scales we can run today, and classical routing heuristics are mature. Same hardware, same team, opposite outcomes.

For QAOA on routing problems with one-hot encoding (Stage 3), valid-output rate on Forte was 4-5% of shots, well below the rate needed to beat strong classical methods. For QAOA on MaxCut (Stage 4), every shot is a valid solution.

## Citing this work

If you use this benchmark suite in academic work, please cite:

> Fariya, S. (2026). Pizza Race: A public benchmark for classical-quantum vehicle routing and graph optimisation across problem scales. Qatalyst Quantum. [Technical report forthcoming.]

## Acknowledgements

This project was made possible by the Qollab Creative Challenge 2026, with direct funding from Qollab and compute credits from IonQ.

## Licence

MIT. See [LICENSE](LICENSE).

## Contact

Dr Siti Fariya, Founder, Qatalyst Quantum
[s.fariya@qatalyst-quantum.co.uk](mailto:s.fariya@qatalyst-quantum.co.uk) · [@Fariyasiti](https://twitter.com/Fariyasiti) · [qatalyst-quantum.co.uk](https://qatalyst-quantum.co.uk)
