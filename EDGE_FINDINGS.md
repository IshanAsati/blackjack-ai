# Evidence for a positive edge in simulation

The new **neural betting model** produced a positive return on held-out finite-shoe simulations. This is a separate experiment from the browser's original hit/stand game. It does not make the original game profitable or establish that a casino offers these conditions.

## What changed

The original Gymnasium game draws cards with replacement, pays all wins even money, and permits only hit/stand. Its exact optimal expected return is **-0.046555977 units per hand**. No improved policy can exceed that ceiling under those rules.

The new experiment uses six decks, 75% penetration before reshuffling, 3:2 natural blackjack payouts, dealer standing on soft 17, a US hole-card peek, and doubling on the initial two cards. Splitting, surrender, and insurance are omitted. All dealt cards are exposed after the round, including a dealer hole card after a player bust. Cards are never replaced within a shoe.

A fixed analytical strategy chooses hit, stand, or double. A new neural model estimates the return from betting one initial unit from two public observations: the pre-deal Hi-Lo true count and fraction of the shoe remaining. Its alternative action is to observe for zero reward. It learns from simulated terminal returns, not optimal action labels. It cannot inspect undealt cards or the current hole card.

During a sit-out, a reference player continues the same strategy, exposing cards. The observer can mirror that player's next hand by committing a wager before any cards are dealt. Thus participation does not change the card stream. This assumes free observation and unrestricted entry on any round; it is not a model of a vacant table where cards stop moving when the agent stops betting.

## Locked training and evaluation

- Training: 20,000 shoes / 870,967 rounds.
- Validation: 10,000 separate shoes. Selected a bet threshold of 0.020 predicted units from six predefined candidates.
- Testing: three separate batches of 40,000 shoes each, generated only after freezing the network and threshold.
- Random streams are keyed independently by split and shoe; no shoes are shared across partitions.
- Confidence intervals treat whole shoes as independent clusters, accounting for dependence between hands within a shoe.

| Untouched test batch | Bets placed | Return per initial unit | Approximate 95% interval |
|---|---:|---:|---:|
| 1 | 31,010 | +1.527% | +0.336% to +2.718% |
| 2 | 30,659 | +1.543% | +0.324% to +2.762% |
| 3 | 30,979 | +1.504% | +0.328% to +2.681% |
| Combined | 92,648 | **+1.525%** | **+0.834% to +2.215%** |

Across **5,227,040 observed rounds**, the model bet on **1.772%** of rounds and earned **1,412.5 simulated units**. This is **0.000270 units per observed round**, including all sit-outs: about 0.27 units per 1,000 observed rounds. The always-bet control, using exactly the same playing strategy and card streams, lost **1.008% per initial unit** (95% interval: -1.103% to -0.913%).

These returns use initial wagers as the denominator; an allowed double stakes one additional unit. They are not returns on total money wagered, win probabilities, or profits per hour. The model has no bankroll cap, observation costs, or entry restrictions. Each batch's interval excludes zero, providing statistical evidence under the simulator assumptions. That is not mathematical proof or guaranteed future profit.

## What is learned and what is not

The network learns when to participate. It uses Monte Carlo value learning for a contextual bandit; it is not the earlier Double DQN, and the playing policy is analytical rather than learned. This first experiment establishes a plausible source of advantage before combining count-aware play and betting in a larger RL agent.

The betting model skips a fresh shoe at count zero. It also skips many mildly favorable-looking counts to meet its validation-selected threshold. The gain comes from observing depleted shoes and selecting bets, under a ruleset different from the old UI.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m blackjackai edge --output runs/edge-reproduction
```

Run artifacts are in `runs/edge/`: `bet_model.pt`, `config.json`, `validation.json`, `betting_policy.csv`, and `summary.json`. `blackjackai.edge.load_bettor` restores the trained model, threshold, and rules. The browser now exposes this policy through finite-shoe mode with auto play. Manual classic mode at `/classic` still uses the separate hit/stand model; their checkpoints are not interchangeable. Live sessions are separate from the frozen test reported here.

Verification: 22 automated tests passed, including payout and double rules, hidden-information boundaries, no-replacement shoes, separated data streams, confidence-interval calculations, and checkpoint restoration. The strategy audit was also corrected to compare decisions against optimal continuation. That correction does not change previously reported exact policy returns.

## Sources and limits

[Gymnasium's rules](https://gymnasium.farama.org/environments/toy_text/blackjack/) document the original environment's with-replacement dealing. [Thorp's primary paper](https://escholarship.org/uc/item/0k1339nz) establishes the mathematical basis for favorable play in non-independent blackjack deals. All numerical results above are from this project's own experiment, not quotations from either source.

To generalize beyond this result, test altered penetration, payouts, hole-card visibility, entry restrictions, and bankroll limits on newly reserved samples. This experiment compares a specific frozen policy against always betting; it does not show that neural learning is better than a conventional fixed count threshold.
