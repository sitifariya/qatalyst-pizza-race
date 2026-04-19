# Pizza Race — The Game (reference snapshot)

Single-page browser game. Players plan pizza delivery routes, then watch
classical and quantum solvers tackle the same puzzle.

## Live version

https://qatalyst-quantum.co.uk/play

The live game is deployed from a private marketing-site repository
(`qatalyst-marketing`) as a Render static site. This file (`play.html`) is a
**reference snapshot** included here under MIT license for academic
reproducibility and so contributors can see how the game and engine connect.

**When the live game is updated, this snapshot should be updated too** (copy
the file from the marketing repo into this `game/` folder, commit, push).

## Running locally

```bash
python -m http.server 8080
# Open http://localhost:8080/play.html
```

No build step, no dependencies. Pure HTML + JavaScript.

## What's in here

- `play.html` — the complete game (all HTML, CSS, JS inlined)

That's it. One file. Single-page game.

## Game structure

Nine stages (Cluster 1):

| Stage | Title | Concept taught |
|-------|-------|----------------|
| 0 | First delivery | Tutorial: basic routing |
| 1 | The rush | Time windows (hot-by) and fuel |
| 2 | The split | Multi-van assignment |
| 3 | The rush (practice) | Reinforcement |
| 4 | The split (practice) | Reinforcement |
| 5 | VIP rush | Weighted priorities |
| 6 | Friday night madness | All constraints stacked (boss level) |
| 7 | Charger shuffle | Shared resource contention |
| 8 | The maybe list | Stochastic demand (uncertainty) |

Each stage introduces one new constraint so players learn incrementally.

## Two robots compete with the player

- **The Planner** (classical robot): uses nearest-neighbour heuristic with 2-opt
  improvement. Fast, industry-standard, sometimes suboptimal.
- **The Dreamer** (quantum-flavoured robot): uses brute-force enumeration.
  Slower, but always finds the provable optimum on these small problems. Named
  to suggest how quantum thinks (sampling across possibilities), though on this
  scale it's not actually quantum.

## Integration with the engine API

When the engine is live, the game adds a "Qatalyst engine" panel showing what
the full classical engine would do on the same problem. This panel is hidden
until the API is wired up.

To enable, edit `play.html` and set:

```javascript
const QATALYST_API_URL = "https://qatalyst-pizza-engine.onrender.com";
```

## Deploying updates to the live game

The live game is deployed from the private `qatalyst-marketing` repository on
Render. To update:

1. Edit `play.html` in the `qatalyst-marketing` repo
2. `git push` to trigger Render auto-deploy
3. Copy the updated file into this public repo's `game/play.html` to keep the
   reference snapshot aligned

This two-step is intentional: the live site stays under private control while
the reference snapshot remains open for academic use.

## Licence

MIT. See [../LICENSE](../LICENSE).
