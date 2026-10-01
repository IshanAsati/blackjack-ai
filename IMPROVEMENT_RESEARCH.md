# Improvement research: where the next edge comes from

Research-only note. No code was modified. Scope is the repo's finite-shoe
ruleset: 6 decks, 75% penetration, S17, 3:2 naturals, US peek, double any-two,
NO split / surrender / insurance (`blackjackai/shoe.py`). All $/hour figures
use **$25 minimum-bet units** and **per 100 observed rounds** (≈1 hour at the
QFIT SCORE convention of 100 hands/hour [S20][S21]) unless stated otherwise.
Heads-up (~150–250 rounds/hour) vs full table (~60–100/hour) is a separate
multiplier; see §A2. Figures tagged **[MEASURED]** are from this repo's frozen
tests; figures tagged **[ESTIMATE]** are scaled from cited literature sims and
are uncertain by roughly ±50% or more where stated.

Coverage of the seven research questions: Q1 §A3–A5; Q2 §A1 (+ plan §B1);
Q3 §A4; Q4 §A6; Q5 §C; Q6 §A7; Q7 §A2.

Baseline reminder [MEASURED] (`EDGE_FINDINGS.md`, `blackjackai/edge.py`):
pooled neural bet-or-skip earns **+1.525% per initial unit bet**
(95% CI +0.834% to +2.215%) but bets on only **1.772%** of observed rounds,
i.e. **0.000270 units per observed round** = **0.027 units per 100 rounds** =
**$0.68 per 100 observed rounds at $25**. Always-bet control on the same
deals loses **−1.008% per initial unit** (CI −1.103% to −0.913%). The
count-aware DQN (`blackjackai/count_rl.py`) reached **−0.0216/hand** vs
**−0.0067/hand** fixed strategy over 2,000 test shoes: a negative result
consistent with a ~0.1–0.3%/hand play-deviation signal buried in ±1/hand
reward noise (per-hand SD ≈ 1.14 [S12]).

## (a) Ranked improvements by expected $/hour at $25 units

| Rank | Improvement | Expected delta vs current $0.68/100 rounds | Type |
|---:|---|---|---|
| 1 | Kelly-sized bet ramp (SIZE, not skip): 1–8 / 1–12 | **+$15 to +$45 per 100 observed rounds** ($15–45/hr at 100 rds/hr) [ESTIMATE] | sizing |
| 2 | Penetration 75% → ~83% + heads-up / wonging (more +counts, more rounds/hour) | **×1.5–3.0 on hourly** once a ramp exists; +$0.3–0.7/hr on the current flat 1u system [ESTIMATE] | environment |
| 3 | Splitting (+ double-after-split, resplit) | **+$8 to +$16 per 100 hands at $25 flat; scales with avg ramp bet** [ESTIMATE] | rules |
| 4 | Insurance deviation at Hi-Lo TC ≥ +3 | **+$2 to +$7 per 100 rounds** at $25-ramp scale; **highest ROI per line of code** [ESTIMATE] | index |
| 5 | Late surrender + Fab 4 | **+$1.5 to +$3 per 100 hands** at $25 flat [ESTIMATE] | rules+index |
| 6 | Remaining Illustrious 18 (ex-insurance, ex-ten-splits) | **+$2 to +$6 per 100 hands** at ramp avg bet [ESTIMATE] | index |
| 7 | Ace side count / level-2 count (Hi-Opt II, Zen, Omega II) | **+$0.2 to +$1 per 100 rounds** at 75% pen shoe [ESTIMATE] | count |
| 8 | Risk-averse / log-utility tuning (incl. fractional Kelly) | **$0 EV; enables larger ramp at fixed RoR** (indirect $/hr) | objective |

Details and citations follow. Relative order of #3 vs #4 depends on regime:
for play-all at flat $25, splitting (#3) exceeds insurance (#4) in absolute
$/hr because it applies to every pair dealt; per hour of implementation effort,
insurance wins (one TC threshold, no payout-engine rewrite).

### A1. Bet sizing is the largest lever (Q2)

Theory (primary): Kelly (1956) maximizes expected log-capital growth [S1];
Thorp brought it to blackjack with *Beat the Dealer* (1962) and formalized
the blackjack case later: optimal fraction ≈ **edge / variance**, and because
blackjack variance is generally > 1 the Kelly fraction is *smaller* than for
an even-money coin toss with the same edge [S2][S3]. Practitioner form:
`f* = advantage / variance`; with variance ≈ 1.3 ([S12] reports 1.295–1.346
across 6-deck rule sets; [S13] uses 1.3) a 1% edge on a $20,000 roll bets
≈ $20,000 × 0.01 / 1.3 ≈ **$154** [S13]. Full Kelly is maximal-growth but
high-volatility: at a 2% edge the simulated P(double before halve) is only
0.67 at full Kelly vs 0.89 at half Kelly, while growth is 75% of max — hence
teams play **1/2 or 1/4 Kelly** [S4][S5]. Trip Risk of Ruin and SCORE, not raw
$/hand, are the correct objectives: SCORE is win rate per 100 hands at a
$10,000 bankroll and 13.5% RoR (relative numbers are what matter: SCORE 20 =
2× SCORE 10) [S20]; RoR tables in Schlesinger's *Blackjack Attack* form underlie
standard calculators [S6].

Spreads and ramps (secondary sims, mark as estimates): conservative 1–4
($10–$40), standard 1–8 ($10–$80: 1/2/4/6/8u at TC ≤0/+1/+2/+3/+4+), aggressive
1–12 (…/12u at TC +5+) [S7]. One worked comparison on a $20,000 roll in a deep
good game: 1–4 ≈ $40/hr at ~2.5% RoR; 1–8 ≈ $70/hr at <2% RoR; 1–12 ≈ $100+/hr
at >3% RoR [S8]. Our rules are worse than that reference (no splits/surrender;
measured always-bet −1.008% vs textbook 6D-S17 ≈ −0.3 to −0.5% [S12]), so scale
down ~30–50%: the **+$15–45/100-rounds** band above assumes a 1–8/1–12 ramp at
$25 min with TC-gated sizing keyed to §A3's edge-per-count (~0.5% per TC
[S13]). The current agent's 1.77% participation at flat 1u is the bottleneck:
even a modest ramp (avg ~2–3u on ~8–10u action per 100 rounds at +0.5–1.0%
edge) yields 0.6–1.8u/100 rounds = $15–45. [ESTIMATE, ±50%+.]

Neural-sizing implication: the net must output **size**, not skip. Minimal
change preserving the current architecture: keep the 2-input value head
E[R | TC, fraction] and add a sizing head `f*(TC, fraction) = clip(E[R]/V̂, 0,
fmax)` with V̂ ≈ 1.3 and bankroll-fraction cap (half-Kelly + spread cap, e.g.
1–8u). See plan §B1. Evaluation must be SCORE/RoR-constrained, not raw
$/hand, or the net will learn overbetting [S2][S4][S20].

### A2. Penetration and table speed (Q7)

Penetration is the dominant *environment* lever. Secondary but consistent
sources: 75%+ is the playability cutoff and 50% multi-deck is Guerlain
unprofitable [S9]; adding 10% penetration adds **40–60% EV**, vs only 10–15%
from betting 10% more at all positive counts [S10]; six-deck SCORE curves
accelerate steeply with penetration (QFIT sims, 2B rounds each, optimal
betting) [S11]. Repo constraint: `ShoeRules.penetration` caps at 0.8, so
83%+ experiments need a config/code change and fresh reserved shoes — do not
reuse test shoes.

Heads-up vs full table is a *throughput* multiplier at fixed edge: heads-up is
faster (more rounds/hour at fixed edge ⇒ more $/hour; faster losses too at
fixed −EV for non-counters), full tables are slower and more social [S22].
SCORE's 100-hands/hour convention [S21] is the numeraire used here; scale
linearly: 200 rounds/hr heads-up ≈ 2× the $/100-rounds figures; 60/hr full ≈
0.6×. Wonging (observe freely, enter only at +counts — exactly the repo's
reference-player assumption) converts penetration into participation: our
1.77% entry rate is a wonging policy with threshold 0.020; lowering the
threshold toward Kelly sizing raises participation and hourly at the cost of
per-bet ROI. Report both metrics (per-bet ROI *and* per-observed-round $),
never one alone.

### A3. Illustrious 18 + Fab 4 values (Q1)

Primary: Schlesinger, *Blackjack Attack* (2nd ed. 1997). Secondary reports
used here (mark as such): the Encyclopedia of Blackjack entry [S14], QFIT
SCORE-by-index charts [S15][S16], and a CVCX-based Wizard-of-Odds sim summary
[S17]. Quantitative claims:

- The 18 capture **~80–85%** of full-index gain in multi-deck shoe games; with
  the Fab 4 surrender indices, **~97%** in one 6D-S17-DAS-surrender/4.5-pen
  1–15-spread CVCX sim (0.834% edge I18+Fab 4 vs 0.862% all indexes) [S17][S18].
  Treat 97% as sim- and rules-specific [ESTIMATE].
- **Insurance (TC ≥ +3, Hi-Lo multi-deck) is #1**, worth **>30–33% of all I18
  gain** alone; the Big 3 (insurance, 16-vs-10, 15-vs-10) are **~60%**; the
  top 12 are **>90%** (4-deck Hi-Lo study context) [S14][S18].
- Perfect count play vs basic strategy: **+66% profit single-deck, +41%
  multi-deck** in one Schlesinger-cited illustration (e.g. $150/hr basic →
  $249 / $211.50) [S18]. Secondary retelling; wide uncertainty — use only as
  an order-of-magnitude cross-check [ESTIMATE].
- QFIT SCORE detail: top-20 indexes (≈ I18 + Fab 4 minus ten-splits) deliver
  nearly all of 100+-index SCORE at poor/medium penetration; at *deep*
  penetration the 100+ tail adds substantially (wide count range late in shoe)
  [S15]. Single-deck behaves differently (tail matters more even at medium
  pen) [S15]. Our 75%-pen 6-deck game sits in the "top-20 ≈ most of it" zone.

Hi-Lo multi-deck S17 indices for the top plays (secondary chart, verify
against *Blackjack Attack* before shipping): insurance TC ≥ +3; 16-vs-10
stand ≥ 0; 15-vs-10 stand ≥ +4; 10,10-vs-5 split ≥ +5; 10,10-vs-6 split ≥ +4;
10-vs-10 double ≥ +4 [S19]. Note the ten-split indices are high-variance and
rare — prime candidates for risk-averse suppression (§A7).

### A4. Missing rules, ranked (Q3)

Rule prices are basic-strategy deltas from published tables (secondary
calculators/sim tables reporting Griffin/Wizard methodology). Baseline: 6D
S17 DAS LS RSA ≈ 0.28% house edge; Wong benchmark (6D S17, no DAS, no
resplit, no surrender) variance 1.28, covariance 0.47 — Wizard's replication
1.295/0.478 [S12].

1. **Splitting (largest).** Enabling DAS alone is **+0.14%** (Wizard: +0.00159
   EV; Appendix 9 exact: −0.4026% with DAS vs −0.5445% without = 0.1419pp)
   [S12]. Resplit-to-4 ≈ +0.05%, resplit-aces ≈ +0.03–0.07%, no-resplit pairs
   ≈ −0.04% [S23]. Composite package (split + DAS + resplit) ≈ **+0.3–0.5%**
   per hand played vs no-split [ESTIMATE]. Cross-check: our measured −1.008%
   vs textbook ≈ −0.3–0.5% leaves ~0.5–0.7pp explained by missing
   splits+surrender+insurance-basic — consistent [MEASURED vs ESTIMATE].
   Exact split EVs and Griffin-approximation accuracy (<±0.001 non-resplit;
   revised formula fixes resplit-tens by >10×) are in Nairn 2019 [S24].
2. **Late surrender.** **+0.07–0.08%** basic-strategy value (Wizard table
   +0.00088 EV with variance *reduction* −0.036 [S12]; house-edge tables
   −0.07 to −0.08pp [S23]). The Fab 4 count-gated surrenders (14-vs-10 +3,
   15-vs-10 0, 15-vs-9 +2, 15-vs-A +1) are worth ≈ I18 entries #10–18 combined
   per secondary summary [S17]. Surrender also *lowers* variance — relevant to
   SCORE/RoR objectives [S12].
3. **Insurance (basic).** At neutral counts insurance is a −7.4% side bet
   (6-deck, P(ten|ace) ≈ 30.9% vs 33.3% break-even) — correctly skipped by
   basic strategy [S25]. Its value is *entirely* as a count-gated deviation
   (§A3): each TC point adds ≈ +2.4pp to insurance ROI from a −7.40% baseline,
   crossing zero at TC ≈ +3 [S25]; a billion-round sim prices count-ramped
   insurance at +$12–22/hr over no-insurance on a $5–$400 spread ($132 →
   $142 standard half-bet → $154 ramped; SCORE 62 → 73 → 81) [S26]. Scale to
   $25 with care (their spread/bankroll differ) — hence the §A4-band $2–7.

Rank: **splitting > surrender > insurance** as *basic-strategy rule additions*;
as *count deviations*, **insurance ≫ any single surrender or split index**
(§A3).

### A5. Edge-per-count and insurance math (supports sizing)

Hi-Lo tags compress Griffin's effects-of-removal (2:+0.38%, 3:+0.44%,
4:+0.55%, 5:+0.69%, 6:+0.46%, 7:+0.28%, 8:0.00%, 9:−0.18%, 10:−0.51%,
A:−0.61%; single-deck benchmark) with betting correlation ≈ 0.97 [S13].
Linear rule of thumb: **edge ≈ (TC − 1) × 0.5%** at ~0.5% base (TC+3 ≈ +1.0%)
[S13]. Use only for ramp shape; freeze final $/hr on sims, not the thumb.

### A6. Count-system upgrades (Q4)

Published correlations (secondary table citing Encyclopedia/QFIT) [S27]:

| System | BC | PE | IC | Ace |
|---|---|---|---|---|
| Hi-Lo | 0.97 | 0.51 | 0.76 | reckoned |
| Hi-Opt I | 0.88 (→~0.97 w/ ASC) | 0.61 | 0.85 | neutral+SC |
| Hi-Opt II | 0.91 (→~0.99 w/ ASC) | 0.67 | 0.91 | neutral+SC |
| Omega II | 0.92 (→~0.99 w/ ASC) | 0.67 | 0.85 | neutral+SC |
| Zen | 0.96 | 0.63 | 0.85 | reckoned (no SC) |
| Wong Halves | 0.99 | 0.57 | 0.72 | reckoned |

Two facts dominate the build decision. First, **betting (~80% of profit)
beats playing (~20%)** in standard lore, so Hi-Lo's BC 0.97 is already the
important part [S28]. Second, QFIT sims show **Hi-Opt II gains minimally over
level-1 at normal penetrations; the ace side count adds only a small further
(cyan) band**, with material level-2+ASC gains confined to deep pen or
single/double-deck pitch games [S16]. A 5.4B-hand 2026 benchmark (secondary,
play-all, I18+Fab 4) still ranks Wong Halves/Zen/REKO within ~1 SCORE point —
execution noise dominates theory gaps [S29]. Verdict: a **second network input
(ace side count) is NOT justified now**; revisit only after sizing + splits +
insurance ship and only with deep-pen shoes. Zen (ace-reckoned, no side count)
is the fallback if level-2 is ever adopted [S27].

### A7. Risk-averse / utility-aware play (Q6)

Max-EV ≠ max-SCORE. Schlesinger's risk-averse indices deliberately deviate
from max-EV (notably ten-splits and thin doubles) to cut variance, raising
SCORE and letting a larger ramp survive the same RoR [S14][S20]. Kelly theory
says the same via fractional betting (half-Kelly keeps 75% growth at 0.89 vs
0.67 double-before-halve odds [S4]). For the agent: gate high-variance
deviations (10,10-vs-5/6 splits; 10-vs-10 / 10-vs-A doubles) on bankroll-scaled
criteria (edge × bet / variance with RoR penalty), and prefer surrender over
marginal hit/stand at short bankrolls. No $/hr is claimed for this row in
isolation; its payoff is enabling ranks #1–#4 at lower RoR.

## (b) Implementation plans for the top 3

Common evaluation harness (all three): fresh shoe streams keyed by
(master, split, index) disjoint from train/validation/test; **shoe-clustered
95% CIs** as in `edge.summarize` (never hand-iid); report per-initial-unit
ROI, per-observed-round $, SCORE-proxy ($/100 rounds at fixed RoR), and
profitable-shoe fraction. Promote only on lower-CI > 0 plus RoR ≤ target on a
$10k-equivalent roll. Reuse `runs/edge` frozen tests as regression gates.

### B1. Kelly-sized betting head (rank #1)

- **Network.** Keep `BetNetwork(2→32→32→1, tanh)` as the *value head*
  V(TC, fraction). Add a *sizing head* with identical trunk or a 2-layer
  16-unit MLP emitting `ê` (edge estimate); bet = `clip(ê / V̂, 0, cap)` ×
  bankroll-unit with V̂ = 1.30 (flat; upgrade to TC-binned V̂ from Wizard
  tables [S12] if calibration shows misfit) and cap = 8u (then 12u variant).
  Scale by bankroll fraction with half-Kelly (×0.5) default [S4]. Two
  thresholds → a ramp: bet 0 below V>0.005, 1u to +1%, linear to cap at +1.5%.
- **State.** Unchanged: pre-deal (TC clipped ±15 / 10, fraction remaining) —
  no hole/current-hand info (preserves the no-lookahead guarantee in
  `edge.collect`).
- **Reward.** Train value head on observed terminal returns exactly as today
  (Monte Carlo bandit); train sizing head on **log-bankroll growth**
  `log(1 + b·R/B)` per shoe-segment (Thorp [S2]) or equivalently penalized
  objective `E[R·b] − λ·Var(R·b)` tuned so RoR ≈ 5% on validation. Never train
  sizing on raw $/hand (learns overbetting).
- **Proof.** Validation picks (cap, λ, thresholds) on lower-CI $/observed-round
  at fixed RoR; frozen triple tested on 3×40k fresh shoes. Ship bar: pooled
  lower-CI $/observed-round > 3× current $0.00675/round at same-or-lower RoR,
  and SCORE-proxy ≥ 2× current. Also report always-bet-with-same-ramp control
  to separate sizing gain from entry gain.

### B2. Penetration + entry/speed protocol (rank #2)

- **Network.** No new play net. Retrain the B1 bet/sizing heads at each pen
  level (0.75 control; 0.80 boundary; 0.83+ only after allowing
  `penetration > 0.8`); entry policy = sizing head with 0-bet allowed
  (wonging). Optional third input: seats/rounds-per-hour so one head prices
  time cost; else fit per-condition heads (cleaner, avoids leakage).
- **State.** (TC, fraction) + condition label (pen, seats). Cards/observations
  identical to today; reference-player-during-sit-out assumption unchanged.
- **Reward.** $/observed-round *minus time cost* (e.g. $/hour at assumed
  rounds/hour) so the head learns to skip marginal +counts at slow tables.
- **Proof.** Pen sweep on reserved shoes with shoe-clustered CIs; claim "deeper
  pen helps" only if pooled $/observed-round lower-CI at deeper pen exceeds
  75% upper-CI at fixed ramp+RoR. Separately A/B heads-up (200/hr) vs full
  (60/hr) pacing on identical deals to isolate the throughput multiplier from
  the edge effect. Expected shape per literature: +40–60% EV per +10% pen
  [S10], accelerating SCORE [S11] — verify, don't assume.

### B3. Splitting support (rank #3)

- **Network.** Extend `CountQNetwork` input with pair flag + rank (or paired
  one-hot) and action set {stand, hit, double, split}; resplit up to 3×
  (4 hands), no drawing to split aces initially (matches Wizard baseline
  [S12]); DAS on/off as a feature flag. Distill first: clone the extended
  analytical table (infinite-deck exact EVs per Appendix-9 style tables) to
  parity before any RL, exactly as `distill_fixed_strategy` did for hit/stand/
  double — supervised, reported as such.
- **State.** Current 46-dim count-aware vector + (is_pair, pair_rank,
  hands_this_round, DAS_allowed). Count/fraction frozen at deal as today.
- **Reward.** Per-round net units across all split hands (negative control:
  same-deal always-hit/stand twin for variance, §C). Stake accounting must
  handle 2–4× initial exposure; sizing head caps split-round commitment
  separately (Kelly on total at risk, covariance 0.47–0.50 [S12]).
- **Proof.** (i) Unit: exact-EV audit vs Appendix-9-style tables (DAS
  −0.4026% vs −0.5445% checkpoint [S12]) and Nairn's Griffin-accuracy bound
  ±0.001 [S24]. (ii) Sim: 2,000-shoe paired test, fixed-strategy + splits vs
  fixed no-split on identical seeds; ship bar: +0.2–0.4pp/hand with lower-CI >
  0 and variance within Wizard envelope (1.30 ± 0.05, cov 0.48 ± 0.03).
  Insurance stays out of this build (separate side-bet node, §A4).

## (c) Principled RL-variance fix: paired common-random-number difference learning

**Diagnosis.** Play deviations are worth ~0.001–0.003/hand [ESTIMATE from §A3]
  against per-hand SD ≈ 1.14 (variance ≈ 1.30 [S12]). Single-sample SNR ≈
  0.002 — Double DQN bootstrapping on raw ±1 rewards cannot resolve it at
  200k hands, exactly the shipped negative result. Unpaired detection of
  0.002/hand at half-width 0.001 needs n ≈ (1.96·1.14/0.001)² ≈ **5M hands**
  [ESTIMATE, illustration]; the fix is to remove shared card noise, not to add
  samples.

**Principled answer (primary theory).** Baselines/control variates reduce
policy-gradient and Monte Carlo variance *without bias* when the baseline is
action-independent given state: Sutton & Barto Ch.13 (policy-gradient with
baseline; advantage A = Q − V) and Sutton et al. 2000 [S30]; the *optimal*
(constant and state) baseline minimizes gradient variance, and the popular
average-reward baseline is provably suboptimal — fit the optimal one instead
(Greensmith et al. 2004/2001 [S31]); same for the optimal reward baseline
(Weaver & Tao 2001 [S32]); trajectory-wise CVs extend this along whole
trajectories and are optimal under stated assumptions (Cheng et al. 2020
[S33]). Multi-agent "difference rewards" (Wolpert & Tumer 2001: Wonderful-Life
/ Aristocrat utilities [S34]) are the same idea from the team-game side:
reward the *difference* the action makes vs a default. Potential-based shaping
(Ng–Harada–Russell 1999) is the reward-shaping member that preserves the
optimal policy — **primary, TO-VERIFY in this pass; re-derive before use**.

**Algorithm (implements §B3 training and any future play-RL).**

1. *Pair every hand.* At each decision state s (hand total/dealer/ace,
   pre-deal TC/fraction, cards dealt so far), record the simulator RNG state.
   Roll the learner action a to termination with RNG stream ω; rewind and roll
   the **fixed `play_table()` action** from the same s with the **same ω**
   (common random numbers). Residual: `d = R_learner(ω) − R_fixed(ω)`.
   Then E[d | s,a] = Q(s,a) − V_fixed(s); argmax_a is preserved because
   V_fixed does not depend on a. This is a control-variate / difference
   estimator, not a shaped hack.
2. *Learn the residual.* Replace DQN target `R + γ max Q` with
   `d + γ max ΔQ` where ΔQ estimates Q − V_fixed; equivalently keep Q-heads but
   train on `d` with a frozen V_fixed lookup (exact infinite-deck EV table or
   Monte-Carlo V̂_fixed(TC, fraction) from the bandit data). Loss: Huber on δ;
   Double-DQN action selection on online Δ-net, evaluation on target Δ-net.
3. *Why variance collapses.* Shared naturals/busts/dealer draws cancel:
   Var(d) ≈ Var of the *decision gap* (typically 0.002–0.05 scale), not 1.30.
   At illustrative Var(d) ≈ 0.01 (SD 0.1), the 0.002/hand test needs
   n ≈ (1.96·0.1/0.001)² ≈ **38k hands (~880 shoes)** instead of ~5M
   [ESTIMATE]. Calibrate Var(d) per decision cluster on validation; size the
   training budget from measured Var(d), not from 1.30.
4. *Pseudocode (adapts `CountTrainer.learn` / `ShoeRunner`).*
   `s, rng = hand.checkpoint(); a = behavior(s); R_L, _ = rollout(s, a, rng.fork()); R_F, _ = rollout(s, fixed(s), rng.fork()); d = R_L − R_F; store (s, a, d, s′, done) with s′ on the learner stream only; TD target on ΔQ.` Keep γ = 1 intra-hand (unchanged objective); never discount across hands.
5. *Evaluation with teeth.* Paired-seed A/B on reserved shoes (same seeds,
   both policies), shoe-clustered CI on mean(d); promote only if lower-CI > 0
   *and* a no-CRN unpaired replication on a second seed block agrees in sign.
   Report Var(R), Var(d), and effective-sample-gain = Var(R)/Var(d).
6. *What NOT to do.* Reward scaling / clipping (changes argmax); γ < 1
   intra-hand (myopic bias); arbitrary shaping bonuses ( Lea to be
   potential-based or drop it); training the baseline on the learner's own
   returns without freezing (feedback loop); hand-iid CIs on correlated
   shoe data (already fixed in `edge.summarize` — keep it).

First demonstration target: 16-vs-10 stand/hit at TC ∈ [−1,+1] and insurance
at TC ∈ [+2,+4] (highest frequency × edge [S14][S26]) — if paired-Δ cannot
show these two, it cannot show anything.

## Assumptions, limits, and uncertainty

- All literature $/hr bands are [ESTIMATE]s transferred across rules/spreads/
  bankrolls; the only [MEASURED] profitability claims are the repo's own
  (+1.525%/u at 1.77% participation; −1.008%/u always-bet; −0.0216 vs −0.0067
  play-RL gap). Do not quote literature bands as repo results.
- Secondary-vs-primary: I18 shares, spread $/hr, correlation tables, and
  insurance-sim $/hr above are secondary reports of primary books/sims;
  primary texts (*Blackjack Attack*, *Theory of Blackjack*, Thorp ch.9, Kelly
  1956, Sutton & Barto, Greensmith, Cheng, Wolpert–Tumer) should be consulted
  before locking indices or sizing constants. Ng shaping is cited TO-VERIFY.
- Simulator gaps vs casino: free observation + unrestricted mid-shoe entry,
  no bankroll cap/time cost/heat/backoff, all cards exposed post-round, no
  splits/surrender/insurance yet. Each inflates sim $/hr vs felt; the RoR and
  time-cost terms in §B1–B2 exist to close part of that gap, not all of it.
- Penetration >0.80, resplit logic, and side-bet plumbing each need
  simulator/reward-accounting changes with new unit tests and fresh shoe
  streams; none are made here.

## Sources

- [S1] J. L. Kelly Jr., "A New Interpretation of Information Rate," *Bell
  Syst. Tech. J.* 35(4):917–926, 1956. DOI 10.1002/j.1538-7305.1956.tb03809.x
  (primary). Max-exponential-growth ⇒ max-E-log-wealth criterion.
- [S2] E. O. Thorp, "The Kelly Criterion in Blackjack, Sports Betting, and
  the Stock Market" (1997 Montreal paper; rev. 1998; Handbook ch.9, 2008)
  (primary). `f* = edge/variance` for blackjack; variance > 1 ⇒ smaller than
  coin-toss fraction; fractional-Kelly practice.
  http://sites.oxy.edu/lengyel/M330/thorp/paper.htm
- [S3] E. O. Thorp, *Beat the Dealer* (1962) (primary, cited via [S2]): first
  practical Kelly sizing for counted blackjack.
- [S4] L. C. MacLean, E. O. Thorp, W. T. Ziemba, "Good and Bad Properties of
  the Kelly Criterion" (primary). Half-Kelly: 0.89 P(double-before-halve) at
  75% growth vs 0.67 at full Kelly (2%-edge table).
  https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
- [S5] W. Chin & M. Ingenoso, "Risk Formulæ for Proportional Betting" (2006)
  (working paper, primary-ish). k-Kelly growth r = (k−k²/2)µ²/σ²; bet
  (kµ/v)·B. https://condor.depaul.edu/wchin/riskpaper.pdf
- [S6] M. Shackleford, "Risk of Ruin in Blackjack," *Wizard of Odds* App.12
  (secondary, Schlesinger-formula tables). Bankroll-unit/RoR mapping.
  https://wizardofodds.com/games/blackjack/risk-of-ruin
- [S7] Blackjack 3000, "Bet Spread Guide (1–4/1–8/1–12)" (secondary, 2026):
  ramp shapes by TC. https://blackjack3000.com/guides/bet-spread-guide
- [S8] Blackjack Apprenticeship, spread/penetration EV examples on $20k roll
  (secondary): 1–4 ≈ $40/hr, 1–8 ≈ $70/hr, 1–12 ≈ $100+/hr in deep good game;
  poor-pen counterexamples. https://www.blackjackapprenticeship.com/bet-spreads-for-hi-lo-card-counting
- [S9] H. Tamburin, "Card Counting 101, Part 2" (2009) (secondary): 75%+
  ideal; 50% multi-deck unprofitable.
  https://www.casinocitytimes.com/henry-tamburin/article/blackjack-card-counting-101-part-2-47280
- [S10] Blackjack Apprenticeship, "How Much Does Penetration Really Matter?"
  (secondary): +10% pen ≈ +40–60% EV vs +10–15% from +10% bets.
  https://www.blackjackapprenticeship.com/how-much-does-penetration-really-matter
- [S11] N. Wattenberger, blackjackincolor penetration charts (secondary sims,
  2B rounds, optimal betting): SCORE accelerates with pen.
  https://blackjackincolor.com/penetration4.htm
- [S12] M. Shackleford, "Variance in Blackjack," *Wizard of Odds* App.4
  (secondary sims, ~10B hands): SD ≈ 1.14 (var 1.303 liberal-strip 6D S17);
  rule deltas S17 +0.00191, DAS +0.00159, surrender +0.00088, RSA +0.00070;
  Appendix 9 DAS −0.4026% vs −0.5445%; Wong benchmark 1.28/0.47 replicated.
  https://wizardofodds.com/games/blackjack/variance
- [S13] ProEdge, "How to Count Cards (Hi-Lo)" (secondary, EoR per Griffin
  [S35]): edge ≈ (TC−1)×0.5%; f* = edge/variance worked example ($154 at 1%
  on $20k). https://www.proedge.tools/resources/card-counting
- [S14] "I is for the Illustrious 18," *Encyclopedia of Blackjack*
  (secondary reporting Schlesinger, *Blackjack Attack*): insurance >30% of
  I18; Big 3 ≈ 60%; top 12 > 90%; indices (insurance +3 multi-deck; 16v10: 0).
  https://www.blackjackreview.com/wp/encyclopedia/i
- [S15] N. Wattenberger, "SCOREs with Different Indexes" (secondary sims):
  top-20 ≈ 100+ at poor/medium pen; tail matters at deep pen / single-deck.
  https://blackjackincolor.com/penetration4.htm
- [S16] N. Wattenberger, "Side Counts — SCOREs" (secondary sims): Hi-Opt II
  minimal gain over level-1 at normal pen; ace-SC small further gain.
  https://blackjackincolor.com/penetration7.htm
- [S17] Lifes a Gambol, "Strategy Deviations" (secondary, CVCX/Wizard sim:
  6D S17 surrender DAS 4.5-pen 1–15): I18+Fab 4 = 0.834% vs 0.862% all-index
  (≈97%); Fab 4 ≈ I18 #10–18 combined. https://lifesagambol.com/strategy-deviations
- [S18] CasinoNewsDaily Illustrious-18 guide (secondary retelling of
  Schlesinger): +66% single-deck / +41% multi-deck perfect-play illustration;
  insurance ≈ 33% of I18. https://www.casinonewsdaily.com/blackjack-guide/illustrious-18
- [S19] ProEdge Illustrious-18/Fab-4 charts (secondary, S17 Hi-Lo index
  table). https://www.proedge.tools/resources/card-counting/deviations
- [S20] N. Wattenberger, *Modern Blackjack*, "Comparison Methodology — SCORE"
  (secondary reporting Schlesinger): SCORE = $/100 hands at $10k roll, 13.5%
  RoR. https://www.qfit.com/book/ModernBlackjackPage99.htm
- [S21] blackjackincolor FAQ (secondary): SCORE assumes 100 hands/hour.
  https://blackjackincolor.com/blackjackfaq.htm
- [S22] *Blackjack Review*, "Heads Up vs Full Table" (2025) (secondary):
  heads-up faster ⇒ more $/hour at +EV (more loss/hour at −EV).
  https://www.blackjackreview.com/wp/2025/01/25/should-you-play-blackjack-heads-up-against-the-dealer-or-at-a-full-table
- [S23] House-edge-by-rules tables (secondary, Wizard methodology): 6:5
  +1.39%, H17 +0.22%, no-DAS +0.14%, no-surrender +0.07–0.08%, RSA −0.06–0.07%.
  https://www.blackjack-gpt.com/blackjack-house-edge-by-rules and
  https://wizardofodds.com/games/blackjack/appendix/9/6ds17r4
- [S24] J. A. Nairn, "Exact Calculation of Expected Values for Splitting
  Pairs in Blackjack," arXiv:1909.13710 (2019) (primary): exact split
  algorithm; Griffin approx ±0.001 non-resplit; revised resplit-tens fix.
- [S25] GamblingCalc insurance-EV derivation (secondary): 2:1 ⇒ break-even
  P(ten) = 33.33% vs ~30.8% fresh shoe (−7.4% 6D); +2.4pp/TC ⇒ flips at +3.
  https://gamblingcalc.com/casino/blackjack-insurance-ev
- [S26] AdvantagePlay insurance sims (secondary, ~1B rounds, Hi-Lo $5–$400):
  $132/hr none → $142 standard → $154 count-ramped (SCORE 62→73→81); TC edge
  table (+3: +1.9% … +8: +16.9%). https://advantageplay.app/table-max-insurance
- [S27] Lifes a Gambol systems table (secondary, via Encyclopedia/QFIT):
  BC/PE/IC for Hi-Lo through Uston APC; ASC lifts Hi-Opt II/Omega II BC to
  ~0.99. https://lifesagambol.com/card-counting-systems
- [S28] Advantage Player systems comparison (secondary): ~80% of profit from
  betting, ~20% from deviations. https://advantage-player.com/blog/comparing-card-counting-systems
- [S29] BlackjackPilot 9-system 5.4B-hand benchmark (2026) (secondary):
  Halves/Zen/REKO within ~1 SCORE point; execution > theory.
  https://blackjackpilot.com/blog/best-card-counting-system-benchmark-2026
- [S30] R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*,
  2nd ed., Ch.13 (primary textbook) + Sutton et al., "Policy Gradient Methods
  with Function Approximation," NeurIPS 1999 (primary): policy-gradient
  theorem; unbiased action-independent baselines; advantage form.
- [S31] E. Greensmith, P. L. Bartlett & J. Baxter, "Variance Reduction
  Techniques for Gradient Estimates in RL," *JMLR* 5:1471–1530 (2004);583–584
  and NeurIPS 2001 version (primary): optimal baseline derivation;
  average-reward baseline suboptimal; actor-critic as control variate.
- [S32] L. Weaver & N. Tao, "The Optimal Reward Baseline for Gradient-Based
  RL," UAI 2001 (primary): optimal constant-baseline theorem (lim b* = mean
  reward).
- [S33] C.-A. Cheng, X. Yan & B. Boots, "Trajectory-wise Control Variates for
  Variance Reduction in Policy Gradient," CoRL 2020 (primary): trajectory CVs
  optimal under stated assumptions.
- [S34] D. H. Wolpert & K. Tumer, "Optimal Payoff Functions for Members of
  Collectives," *Adv. Complex Syst.* 4(2/3):265–279 (2001) (primary):
  difference (Wonderful-Life / Aristocrat) utilities.
- [S35] P. A. Griffin, *The Theory of Blackjack* (1979; 6th ed. 1999)
  (primary book): effects of removal, BC/PE/IC framework, split
  approximations. TO-VERIFY page-level numbers in this pass; EoR values above
  via [S13]'s Griffin table.
