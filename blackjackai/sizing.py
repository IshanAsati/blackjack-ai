"""Kelly-sized betting: turn the value head's edge estimate into a bet SIZE.

The flat bet-or-skip system leaves its edge underused (1.77% participation).
Sizing follows Thorp/Kelly for blackjack: optimal bankroll fraction ~= edge /
variance with variance ~= 1.3, played at half-Kelly by default. This module is
pure functions so the math is unit-testable; evaluation with bankroll paths,
shoe-clustered CIs, and RoR proxies lives in run_sizing below.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

VARIANCE = 1.30  # 6-deck S17 nearly matches Wong/Wizard benchmarks (~1.28-1.35)


@dataclass(frozen=True)
class SizingConfig:
    bankroll_units: float = 400.0  # e.g. $10,000 at $25 units
    kelly_fraction: float = 0.5  # half-Kelly default
    variance: float = VARIANCE
    cap_units: int = 8  # spread cap: 1-8 (12 for the aggressive variant)
    min_edge: float = 0.005  # bet nothing below this predicted edge
    force_bet: bool = False  # always-bet control: same ramp, entry gate open

    def __post_init__(self):
        if self.bankroll_units <= 0 or self.variance <= 0:
            raise ValueError("bankroll and variance must be positive")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.cap_units < 1 or self.min_edge < 0:
            raise ValueError("cap must be >= 1 and min_edge nonnegative")


def kelly_units(edge: float, bankroll_units: float, config: SizingConfig) -> int:
    """Half-Kelly stake in whole units, 0 below min_edge, capped at cap_units."""
    if edge < config.min_edge and not config.force_bet:
        return 0
    fraction = config.kelly_fraction * edge / config.variance
    units = int(fraction * bankroll_units)
    return max(1, min(config.cap_units, units))


def size_bets(edges: np.ndarray, config: SizingConfig) -> np.ndarray:
    """Vectorized stakes for one shoe-stream given frozen per-round edge estimates."""
    bankroll = config.bankroll_units
    units = np.zeros(len(edges), dtype=np.int64)
    for i, edge in enumerate(edges):
        stake = kelly_units(float(edge), bankroll, config)
        units[i] = stake
        # Bankroll evolves inside validation/selection exactly as it would live.
        # Rewards are applied by the caller (needs per-round outcomes); this pass
        # sizes on the pre-round roll only. See simulate_bankroll for the joint loop.
    return units


def simulate_bankroll(edges: np.ndarray, rewards: np.ndarray, shoe_ids: np.ndarray,
                      config: SizingConfig) -> dict:
    """Walk the bankroll round by round in stream order.

    edges: frozen predicted edge per observed round (bet sizing input only).
    rewards: realized profit per INITIAL unit (doubles/splits included).
    Returns terminal growth, worst drawdown, and a trip-RoR proxy: the fraction
    of cyclic shoe-order rotations that ever halve the roll (cheap, labeled).
    """
    edges = np.asarray(edges, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    bankroll, peak, worst = config.bankroll_units, config.bankroll_units, 1.0
    profits = np.zeros(len(edges))
    for i in range(len(edges)):
        stake = kelly_units(edges[i], bankroll, config)
        profit = stake * rewards[i]
        profits[i] = profit
        bankroll += profit
        peak = max(peak, bankroll)
        worst = min(worst, bankroll / config.bankroll_units)
        if bankroll <= 0:
            bankroll = 0.0
            profits[i + 1:] = 0.0
            break
    n_shoes = int(shoe_ids.max()) + 1 if len(shoe_ids) else 0
    per_shoe = np.bincount(shoe_ids, weights=profits, minlength=n_shoes)
    # Trip-RoR proxy: rotate shoe order (preserves within-shoe correlation,
    # varies trip path) and count rotations that ever halve the starting roll.
    halves, rotations = 0, min(n_shoes, 40)
    for shift in range(rotations):
        roll = config.bankroll_units
        ruined = False
        for shoe in np.roll(np.arange(n_shoes), shift):
            roll += per_shoe[shoe]
            if roll <= config.bankroll_units / 2:
                ruined = True
                break
        halves += ruined
    observed = len(edges)
    per_100 = float(profits.sum() / observed * 100) if observed else 0.0
    return {
        "rounds": observed, "profit_units": float(profits.sum()),
        "profit_per_100_rounds": per_100,
        "terminal_bankroll_units": float(bankroll),
        "worst_bankroll_fraction": float(worst),
        "trip_halving_fraction": float(halves / rotations) if rotations else None,
        "per_shoe_profit": per_shoe,
    }


def summarize_sized(per_shoe: np.ndarray, observed: int, unit_dollars: float = 25.0) -> dict:
    """Shoe-clustered CI on sized profit; never treats correlated hands as iid."""
    per_shoe = np.asarray(per_shoe, dtype=float)
    shoes = len(per_shoe)
    total = float(per_shoe.sum())
    per_100 = total / observed * 100
    if shoes > 1:
        se = float(np.std(per_shoe, ddof=1) / np.sqrt(shoes) / (observed / shoes) * 100)
        ci = [per_100 - 1.96 * se, per_100 + 1.96 * se]
    else:
        ci = None
    return {
        "shoes": shoes, "observed_rounds": observed,
        "profit_units": total, "profit_per_100_rounds": per_100,
        "profit_per_100_rounds_95ci": ci,
        "profit_dollars_per_100_rounds": per_100 * unit_dollars,
        "ci_method": "normal approximation with independent shoes as clusters",
    }


def run_sizing(edges: np.ndarray, rewards: np.ndarray, shoe_ids: np.ndarray,
               config: SizingConfig, unit_dollars: float = 25.0) -> dict:
    """Evaluate ONE frozen (value head, sizing cfg) on collected arrays."""
    result = simulate_bankroll(np.asarray(edges, dtype=float), np.asarray(rewards, dtype=float),
                               np.asarray(shoe_ids), config)
    summary = summarize_sized(result["per_shoe_profit"], result["rounds"], unit_dollars)
    return {"sizing": {"kelly_fraction": config.kelly_fraction, "cap_units": config.cap_units,
                       "min_edge": config.min_edge, "bankroll_units": config.bankroll_units,
                       "force_bet": config.force_bet},
            "bankroll": {k: v for k, v in result.items() if k != "per_shoe_profit"},
            "shoe_clustered": summary}


def select_sizing(value_predict, collect_fn, master: int, shoes: int,
                  candidates: list[SizingConfig], max_halving: float = 0.05) -> dict:
    """Pick the sizing config on validation: best lower-CI $/100 with RoR cap."""
    observations, rewards, shoe_ids = collect_fn(shoes, master, "sizing-validation")
    edges = np.asarray(value_predict(observations), dtype=float)
    scored = []
    for config in candidates:
        result = simulate_bankroll(edges, rewards, shoe_ids, config)
        summary = summarize_sized(result["per_shoe_profit"], result["rounds"])
        ci = summary["profit_per_100_rounds_95ci"]
        lower = ci[0] if ci else float("-inf")
        halving = result["trip_halving_fraction"] or 0.0
        scored.append({"config": {"kelly_fraction": config.kelly_fraction, "cap_units": config.cap_units,
                                  "min_edge": config.min_edge, "bankroll_units": config.bankroll_units,
                                  "force_bet": config.force_bet},
                       "profit_per_100_rounds": summary["profit_per_100_rounds"], "lower_ci": lower,
                       "trip_halving_fraction": halving,
                       "eligible": halving <= max_halving})
    eligible = [s for s in scored if s["eligible"]]
    pool = eligible if eligible else scored
    best = max(pool, key=lambda s: s["lower_ci"])
    return {"candidates": scored, "selected": best,
            "selection_rule": f"max lower-95%-CI $/100 rounds subject to trip-halving <= {max_halving}"}
