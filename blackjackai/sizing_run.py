"""Frozen-head Kelly sizing experiment: select stakes on validation, test frozen.

Loads the value head trained by `edge` (its predicted return per initial unit
is the edge estimate), selects (cap, kelly fraction, min edge) on validation
shoes by lower-CI $/100 rounds subject to a trip-halving cap, then freezes and
tests on untouched batches. An always-bet same-ramp control on the same deals
separates sizing gain from entry gain.
"""

import json
from pathlib import Path

import numpy as np

from .edge import collect, load_bettor, predict
from .sizing import SizingConfig, run_sizing, select_sizing, summarize_sized


def run_sizing_experiment(bettor_path: Path, rules, validation_shoes: int, test_shoes: int,
                          test_batches: int, seed: int, rounds_per_hour: int, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    network, _, _ = load_bettor(bettor_path)

    def value_predict(observations):
        return predict(network, observations)

    def collect_fn(shoes, master, split):
        return collect(rules, shoes, master, split)

    print("Selecting Kelly sizing on validation shoes...", flush=True)
    grid = [SizingConfig(kelly_fraction=k, cap_units=c, min_edge=e)
            for c in (4, 8, 12) for k in (0.25, 0.5) for e in (0.005, 0.01, 0.02)]
    selection = select_sizing(value_predict, collect_fn, seed, validation_shoes, grid)
    cfg = selection["selected"]["config"]
    sized = SizingConfig(**cfg)
    print(f"Selected 1-{sized.cap_units} cap, {sized.kelly_fraction}-Kelly, min edge {sized.min_edge} "
          f"(val ${selection['selected'].get('profit_per_100_rounds', 0) * 25:+.2f}/100r)", flush=True)
    control_cfg = SizingConfig(kelly_fraction=sized.kelly_fraction, cap_units=sized.cap_units,
                               min_edge=sized.min_edge, bankroll_units=sized.bankroll_units,
                               force_bet=True)
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    batches, pool_sized, pool_n = [], [], 0
    for batch in range(test_batches):
        observations, rewards, shoe_ids = collect_fn(test_shoes, seed, f"sizing-test-{batch}")
        edges = np.asarray(value_predict(observations), dtype=float)
        neural = run_sizing(edges, rewards, shoe_ids, sized)
        # Control reuses the same deals with the entry gate forced open.
        from .sizing import simulate_bankroll, summarize_sized
        ctrl = simulate_bankroll(edges, rewards, shoe_ids, control_cfg)
        ctrl_summary = summarize_sized(ctrl["per_shoe_profit"], ctrl["rounds"])
        batches.append({"batch": batch,
                        "kelly_sized": neural["shoe_clustered"],
                        "always_bet_same_ramp": ctrl_summary})
        pool_sized.append(neural["shoe_clustered"])
        pool_n += neural["shoe_clustered"]["observed_rounds"]
        print(f"Sizing test {batch + 1}: sized ${neural['shoe_clustered']['profit_dollars_per_100_rounds']:+.2f}/100r "
              f"vs always-ramp ${ctrl_summary['profit_per_100_rounds'] * 25:+.2f}/100r", flush=True)
    pooled_units = sum(b["profit_units"] for b in pool_sized)
    pooled_rounds = sum(b["observed_rounds"] for b in pool_sized)
    per_100 = pooled_units / pooled_rounds * 100
    summary = {
        "status": "sizing_supported" if per_100 > 0 else "sizing_not_supported",
        "bettor": str(bettor_path), "rules": {"decks": rules.decks, "penetration": rules.penetration,
                                              "blackjack_payout": rules.blackjack_payout},
        "selected_sizing": selection["selected"],
        "test_batches": batches,
        "pooled_profit_units_per_100_rounds": per_100,
        "pooled_dollars_per_100_rounds_at_25": per_100 * 25,
        "dollars_per_hour_at_25": {f"{rph}_rounds_per_hour": per_100 * 25 * rph / 100
                                    for rph in (60, rounds_per_hour, 200)},
        "caveat": "Simulator result under free-observation/unrestricted-entry assumptions; "
                  "stakes scale with a $10,000 roll and half-Kelly by default.",
    }
    (output / "sizing.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
