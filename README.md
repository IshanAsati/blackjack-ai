# Blackjack AI

A PyTorch Deep Q-Network (DQN) that learns when to hit or stand by playing blackjack. Two hidden layers, experience replay, a separate target network, Huber loss, and decreasing epsilon-greedy exploration form the initial learning pipeline. Training runs on CPU; no GPU is needed.

## Setup

Python 3.11 or newer. From this directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe -m blackjackai train --episodes 200000 --output runs/dqn
.\.venv\Scripts\python.exe -m blackjackai evaluate runs/dqn/model.pt --episodes 100000
```

## Play in the browser

After training has created `runs/dqn/model.pt`, start the local table:

```powershell
.\.venv\Scripts\python.exe -m blackjackai play
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). When `runs/edge/bet_model.pt` exists, the home page opens the finite-shoe model. Otherwise it opens the original hit/stand game. You can always reach manual hit/stand play through **Play yourself** at `/classic` (keyboard shortcuts: `H` and `S`). Its unlimited-deck rules differ from the finite-shoe experiment.

A shared table for 1-5 seats against one dealer lives at `/multi` (keys `1`-`5` select a seat, `H`/`S` act, `A` takes the model's move). Seats act in turn order under unlimited-deck rules, so each seat's DQN advice stays valid; the dealer plays once after all seats stand or bust.

Got cards in front of you? `/advise` takes the dealer's card and your cards and returns the model's hit/stand call with its values — same endpoint as `POST /api/advise` (`player` list, `dealer` card). Busts and 21s are handled directly; pairs are advised as hit/stand only since the model never splits.

The finite-shoe page offers:

- **Next round**: let the model commit its bet-or-observe choice, then play one full hand.
- **Find next bet**: advance to the first qualifying bet, including its outcome whether it wins or loses; at most 10,000 rounds per click.
- **Start auto / Pause auto**: continuous play with three speed settings. Pausing finishes the current batch; hiding the browser tab pauses auto too.
- **Run 100,000 rounds**: complete exactly that many additional rounds unless explicitly paused, without stopping on a winning streak.
- **Training**: inspect the saved RL and mixed learning curves with a checkpoint slider. Autoencoder and betting results are summaries because their full epoch histories were not saved. Viewing the page does not start a new training job.
- A live profit chart, the same-deal always-bet control, completed-shoe uncertainty, and JSON session downloads. Session state survives a page refresh while the server is running, but is not persisted across server restarts. **Start new session** resets its ledger; the prior session can be downloaded first.

The frozen betting model can be set explicitly with `--bettor runs/edge/bet_model.pt`. Decisions use pre-deal public counts. The network stays frozen in live play; auto mode is inference, not further training. Historical test results appear separately from the current session, which can lose.

## Compare three training approaches

```powershell
.\.venv\Scripts\python.exe -m blackjackai compare --output runs/comparison
```

This produces three distinct artifacts: `rl/` trains a Double DQN from scratch; `unsupervised/` trains an autoencoder from unlabeled game states and has no hit/stand policy because it never receives rewards; `mixed/` initializes a Double DQN with that unsupervised state representation, then fine-tunes it through rewards. `summary.json` identifies the best playable checkpoint by exact expected reward, while each playable run includes a Monte Carlo evaluation. The default runs 200,000 episodes per playable agent. To play a comparison model, supply its path: `python -m blackjackai play runs/comparison/mixed/model.pt --port 8001`.

For a quick pipeline check, use `--episodes 2000 --eval-episodes 1000 --output runs/smoke`. That is too short to judge learning quality. Use a fresh output directory for each run; existing checkpoints are protected from overwrites.

Training produces:

- `model.pt`: network weights and training configuration for inference (not a full resumable optimizer/replay snapshot).
- `training.csv`: exploratory training reward, epsilon, environment steps, and last minibatch loss per reporting interval.
- `evaluation.json`: frozen-policy results for the DQN, random play, and a stand-on-17 heuristic.
- `policy.csv`: network-predicted actions and Q-values by player total, dealer card, and usable ace.
- `config.json`: hyperparameters and seed.

Evaluation uses separate environment instances and seeds, disables exploration, and makes no parameter updates. Each policy gets the same per-hand seeds, but different actions can cause different subsequent draws. Results include win/draw/loss counts, mean reward per hand, and an approximate 95% confidence interval for mean reward. A single run is not proof of convergence: compare multiple training seeds and fresh evaluation seeds. The fixed heuristic is a baseline, not optimal basic strategy.

## Initial game rules

Gymnasium `Blackjack-v1`, explicitly configured with `natural=False, sab=False`:

- Cards drawn with replacement from an unlimited deck; dealer stands on all 17s.
- Observation: player total, dealer's visible card, and whether the player has a usable ace. The dealer's hidden card is never supplied to the model.
- Actions: `0 = stand`, `1 = hit`.
- Rewards: win `+1`, draw `0`, loss `-1`; no premium payout for a natural blackjack.
- No splitting, doubling, surrender, insurance, betting, or card counting in this version.

The observation is encoded as 32 player-total indicators, 10 dealer-card indicators, and one usable-ace flag. The network maps these 43 inputs through 64 and 64 ReLU units to two estimated action values. Terminal hands never bootstrap future reward. The discount is 1 because the objective is undiscounted return within a finite hand.

The objective is to maximize expected reward per hand. Even a strong policy can have negative average reward under these rules; win rate alone does not establish profitability.

## Test whether a betting model can earn a positive return

The original unlimited-deck, even-money, hit/stand game has an exact optimal return of **-0.04655598 units per hand**. Training a larger neural network cannot make that game profitable. The separate finite-shoe experiment tests a different, explicit set of rules:

```powershell
.\.venv\Scripts\python.exe -m blackjackai edge --output runs/edge
```

This simulator uses six decks, 75% penetration, dealer stands on soft 17, 3:2 natural blackjacks, US hole-card peek, and doubling on any first two cards. Splitting, surrender, and insurance are omitted. It is a research ruleset, not a full casino replica. All cards, including hole cards after busts, become visible at round end. No next-round cards or current hidden cards enter a betting decision.

The neural model estimates the return from betting one initial unit using the **pre-deal Hi-Lo true count and fraction of shoe remaining**. Training uses observed simulated rewards, not optimal-action labels. A fixed analytical hit/stand/double strategy plays the hand; the new network learns bet-or-observe decisions. This is Monte Carlo value learning for a contextual bandit, not another DQN player model.

The agent can observe freely and enter on any round. A reference player continues playing during sit-outs, so the shoe keeps advancing. A selected hand follows that player's strategy; doubling adds a second unit. These assumptions matter: an enforced minimum bet on every hand, entry restrictions, time costs, or different payouts could erase the measured edge. There is no bankroll cap in this experiment.

Whole shoes are partitioned into training, validation, and three disjoint test batches. Validation chooses a threshold from a fixed candidate list; the neural model and threshold are frozen before tests. Confidence intervals use shoes as independent clusters rather than treating correlated hands as independent. The always-bet baseline uses the same playing policy and exactly the same test deals. Returns are reported per **initial unit bet**, which includes the gains/losses of doubling, and per observed round including sit-outs. They are not win rates or guaranteed profit.

Outputs include `bet_model.pt`, `validation.json`, `betting_policy.csv`, `config.json`, and `summary.json`. Load a frozen model with `blackjackai.edge.load_bettor`; the browser uses it in finite-shoe mode while manual classic mode uses the separate hit/stand network. A positive lower 95% confidence bound is statistical evidence under these assumptions, not a mathematical proof of profitability in real games.

The strategy audit now compares choices with optimal continuation when counting decision mismatches. Earlier exact policy-return estimates are unchanged; previously saved reports remain historical artifacts.

## Kelly-sized betting (B1)

The flat bet-or-skip system left its edge underused (1.77% participation). The value head's predicted return now feeds a half-Kelly sizing head (`blackjackai/sizing.py`: stake = ½·edge/1.3 × bankroll, 1–4 unit cap, nothing below 0.5% edge, $10,000 roll):

```powershell
.\.venv\Scripts\python.exe -m blackjackai size --output runs/sizing
```

Selection on 10,000 validation shoes (best lower-CI $/100 rounds with trip-halving ≤ 5%), frozen, then tested on 3 × 40,000 fresh shoes. An always-bet same-ramp control on the same deals separates sizing gain from entry gain:

| Test batch | Kelly-sized $/100 rounds | Always-ramp control |
|---|---:|---:|
| 1 | +13.36 | −0.58 |
| 2 | +11.24 | −0.58 |
| 3 | +14.77 | −0.58 |
| Pooled | **+$13.12** | −$0.58 |

(Dollars at $25 units; all three sized batches have lower-CI > 0.) The control losing proves the entry gate does the work — the same ramp without skipping loses. That's ~$13/hour at 100 rounds/hour, ~$26/hour heads-up (200/hr). The live table now stakes real Kelly sizes from its session bankroll and shows them ("Bet 3u (Kelly)").

## Bankroll simulator tab

The finite-shoe page has a **Bankroll sim** tab: hand the model your money and watch. Enter a bankroll ($25–$10M), a $5/$10/$25/$100 unit, and up to 2,000 shoes. It stakes half-Kelly sizes against flat 1-unit betting on the same reserved deals until the roll doubles, busts, or runs out of shoes, then charts both equity curves with terminal/peak/drawdown stats. Same seed replays the identical run. Simulation, not a promise — short runs routinely lose.

## Penetration and table speed (B2)

`ShoeRules` now allows penetration up to 0.90, and `size` accepts `--penetration` and reports $/hour at 60/100/200 rounds/hour. Pilot sweep (same frozen value head, sizing reselected per level, 2 × 3,000-shoe tests):

| Penetration | Pooled $/100 rounds |
|---|---:|
| 0.75 | +13.12 (full 3 × 40k tests) |
| 0.80 | +21.04 |
| 0.85 | +28.79 |

Direction matches the literature (+40–60% EV per +10% pen); magnitudes are pilot-size, not proof. Heads-up vs full-table speed multiplies any of these by ~2× / ~0.6× at fixed edge.

## Splitting (B3)

`Shoe.play_round` now plays pairs: always split aces/8s, never 5s/10s, standard splits otherwise (4,4 only with double-after-split, `ShoeRules.double_after_split`, default on). Split aces get one card each; no resplits in v1; rewards are total profit per initial unit (split rounds can exceed ±2, and the sizing head caps total at risk). Round-paired proof — same cards replayed with and without splitting, 3,000 splittable rounds:

- **+0.229 units per splittable round** (95% CI +0.154 to +0.304), i.e. ≈ **+0.58pp/hand** fleet-wide. Ship bar (lower-CI > 0) met.

The live table plays splits with the analytical pair table (the neural play net covers non-pairs) and renders each hand separately. Insurance, surrender, and resplits are still out — insurance at TC+3 is the documented next highest-ROI line of code.

## Host it yourself (Vercel, no backend)

`webapp/` is a fully static single-page app — the exact same frozen nets run in JavaScript, so there is nothing to host but files. No PyTorch, no Python, no server; sessions live in the browser.

```powershell
.\.venv\Scripts\python.exe -m blackjackai export --checkpoint runs/comparison/rl/model.pt --output webapp
```

This writes `webapp/data/*.json` (network weights, pair/split tables, sizing config, training curves). Re-run it after any retraining; the page refuses to boot with a clear error if data is missing.

Deploy with the Vercel dashboard (import the repo, set **Root Directory to `webapp`**, defaults otherwise) or the CLI (`npx vercel --cwd webapp`). Any static host works the same way.

Parity notes: JS matrix math reproduces torch to <0.001 (pinned cross-check in `tests/engine_smoke.mjs`, full-grid check in `tests/test_export.py`). Card *streams* differ from the Python server (each uses its own seeded PRNG) — mechanics, payouts, counts, and policies are identical. The 69-test Python suite still covers the server; the static app is covered by the export parity tests plus the two Node smoke scripts.

## Count-aware neural play strategy

The live finite-shoe table is now dealt by a neural network, not a lookup table:

```powershell
.\.venv\Scripts\python.exe -m blackjackai count-distill --output runs/count-distilled
.\.venv\Scripts\python.exe -m blackjackai count-rl --hands 200000 --output runs/count
```

`count-distill` clones the analytical hit/stand/double strategy into the same 46-input neural architecture the RL uses (hand one-hot + pre-deal Hi-Lo true count + shoe fraction + double rights). It reaches **100% agreement** on an 11,760-state grid in seconds and plays at fixed-strategy parity (**-0.0067/hand** on 200 fresh shoes). `EdgeService` auto-loads `runs/count-distilled/model.pt` when present and builds the per-round play table from the net at that round's pre-deal count (cached on a grid, so betting still cannot change the card stream: every hand uses the same session policy).

`count-rl` trains that architecture with Double DQN from scratch. Honest negative result, kept in `runs/count/`: **-0.0216/hand** over 2,000 test shoes vs **-0.0067/hand** for the fixed strategy on the same seeds, and RL fine-tuning from the distilled net (`runs/count-tuned/`, **-0.0272/hand**) also regresses. The learnable count-deviation edge for play (~0.1-0.3%/hand) is smaller than DQN approximation error at this budget: the net learns real count sensitivity (e.g. doubling 11v10 only at non-negative counts) but drops basic-strategy spots (e.g. standing 12v4). So the shipped player is the distilled net, reported as supervised cloning rather than RL. The system's edge comes from the neural betting model (+1.525% per initial unit, above), which decides *when* to play; the play network decides *how*. `strategy.csv` in each run shows actions and Q-values by total, dealer, ace, and true count.

## References

- [Gymnasium blackjack rules](https://gymnasium.farama.org/environments/toy_text/blackjack/)
- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Thorp, A Favorable Strategy for Twenty-One (1961)](https://escholarship.org/uc/item/0k1339nz)
