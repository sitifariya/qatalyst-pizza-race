# Stage 4_q — VIP Rush (Research Companion)

The Forte-compatible variant of the game's Stage 4. **25 qubits**, fits IonQ
Forte's 36-qubit limit comfortably and runs fine on a laptop's Aer simulator.

## How this differs from the game's Stage 4

| Feature | Game Stage 4 | Stage 4_q (this) |
|---------|--------------|------------------|
| Customers | 7 | 5 |
| Qubits | 49 | 25 |
| VIPs | 2 (Mayor, Celeb) | 2 (Mayor, Celeb) |
| Mechanic | VIP 4x cold penalty | VIP 4x cold penalty (same) |
| Time windows | Yes | Yes |
| Fuel limit | 50 | 80 (relaxed) |
| Forte runnable | No | Yes |
| Local laptop runnable | No | Yes |

The 5-customer subset (Mayor, Adam, Celeb, Bee, Ria) and slightly relaxed
geometry/deadlines were chosen so the brute-force optimum is a meaningful
all-VIPs-hot solution. The game's Stage 4 is calibrated for pacing; the
research companion is calibrated for hardware accessibility.

## Brute-force reference

Optimum: **Mayor → Celeb → Adam → Ria → Bee**, score 32
- 3 hot, 2 cold, **0 VIP cold**
- Fuel: 85.5 (over the 80 tank by 5.5)

The whole point of Stage 4: VIPs delivered first, regulars second. The
optimum delivers both VIPs hot and accepts the cold pizzas + small fuel
overrun as the price.

## Run order

```bash
cd quantum
source .venv/bin/activate

# 1. Local test (free, ~60 sec)
python stage4q_qaoa.py --backend local

# 2. IonQ cloud simulator (free, queue 1-10 min)
python stage4q_qaoa.py --backend ionq_sim

# 3. Real hardware on Forte (paid, ~$80-180)
python stage4q_qaoa.py --backend ionq_forte
```

## What you should see

Brute-force section will print:
```
Brute-force QUBO optimum:
  route: [0, 2, 1, 4, 3] (Mayor-Celeb-Adam-Ria-Bee)
  QUBO energy: <negative number>
  game score: 32 (hot=3, cold=2, VIP cold=0, fuel=85.5)
```

QAOA should return a route with game score 32 and **VIP cold = 0**. The
critical check: VIPs hot, not "lowest QUBO energy".

## Why this is the showcase stage for Forte

- Fits 36-qubit Forte (25 qubits used)
- Tests four QUBO mechanics simultaneously: assignment, distance, time
  windows, VIP-weighted cold, fuel overrun
- Has a clear pedagogy (VIPs first) that survives noise — if QAOA gets
  even close to optimal, the route will likely keep both VIPs hot
- 120 candidate permutations brute-force in milliseconds for verification

This is the strongest single-stage demonstration of the QAOA pipeline on
real quantum hardware.
