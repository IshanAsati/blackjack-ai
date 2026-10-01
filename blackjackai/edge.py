"""Learn whether to bet from public counts and test on independent shoes.

The neural net estimates Q(count, bet-one-unit) from simulated terminal
returns (Monte Carlo contextual-bandit value learning). Q(skip)=0 exactly.
This is not DQN player strategy training. A fixed, analytical play policy
handles hit/stand/double. One observer may mirror that player's initial
wager or sit out; thus skipping never freezes the shoe or changes the deals.
"""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agent import load_policy
from .oracle import initial_value, optimal_values, policy_report
from .shoe import Shoe, ShoeRules, play_table


@dataclass(frozen=True)
class EdgeConfig:
    train_shoes: int = 20_000
    validation_shoes: int = 10_000
    test_shoes_per_batch: int = 40_000
    test_batches: int = 3
    seed: int = 71623
    epochs: int = 1200

    def __post_init__(self):
        if min(self.train_shoes, self.validation_shoes, self.test_shoes_per_batch, self.epochs) < 1:
            raise ValueError("sample and epoch counts must be positive")
        if self.test_batches < 2:
            raise ValueError("use at least two independent test batches")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


class BetNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observation):
        return self.layers(observation).squeeze(-1)


def load_bettor(path: Path):
    """Load the frozen betting net and decision threshold, separately from DQN."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("kind") != "public_count_monte_carlo_bet_value" or data.get("format_version") != 1:
        raise ValueError("Expected a version-1 public-count betting checkpoint")
    model = BetNetwork()
    model.load_state_dict(data["model"])
    return model.eval(), float(data["threshold"]), ShoeRules(**data["rules"])


def features(observations):
    result = np.asarray(observations, dtype=np.float32).copy()
    result[:, 0] = np.clip(result[:, 0], -15, 15) / 10
    return torch.from_numpy(result)


def shoe_seed(master, split, index):
    # Independently hashed seeds ensure no overlap between split/replicate streams.
    return int.from_bytes(hashlib.sha256(f"{master}:{split}:{index}".encode()).digest()[:8], "little")


def collect(rules, shoes, master, split):
    observations, rewards, shoe_ids = [], [], []
    actions = play_table()
    for index in range(shoes):
        shoe = Shoe(rules, shoe_seed(master, split, index))
        while shoe.can_deal:
            observations.append(shoe.observation())  # Captured BEFORE the deal.
            rewards.append(shoe.play_round(actions))
            shoe_ids.append(index)
    return np.asarray(observations, dtype=np.float32), np.asarray(rewards, dtype=np.float64), np.asarray(shoe_ids, dtype=np.int64)


def predict(network, observations):
    network.eval()
    with torch.no_grad():
        return np.concatenate([network(features(batch)).numpy() for batch in np.array_split(observations, max(1, len(observations) // 8192))])


def summarize(rewards, shoe_ids, selected, shoes):
    """Cluster by shoe: consecutive hands from one shoe are NOT independent."""
    profits = np.bincount(shoe_ids, weights=rewards * selected, minlength=shoes)
    bets = np.bincount(shoe_ids, weights=selected, minlength=shoes)
    observed = np.bincount(shoe_ids, minlength=shoes)
    per_round = float(profits.sum() / observed.sum())
    round_residual = profits - per_round * observed
    round_se = float(np.std(round_residual, ddof=1) / np.sqrt(shoes) / observed.mean()) if shoes > 1 else None
    roi = float(profits.sum() / bets.sum()) if bets.sum() else None
    if roi is not None and shoes > 1:
        roi_se = float(np.std(profits - roi * bets, ddof=1) / np.sqrt(shoes) / bets.mean())
        roi_ci = [roi - 1.96 * roi_se, roi + 1.96 * roi_se]
    else:
        roi_ci = None
    return {
        "shoes": shoes, "observed_rounds": int(observed.sum()), "bets": int(bets.sum()),
        "participation": float(bets.sum() / observed.sum()), "profit_units": float(profits.sum()),
        "profit_per_observed_round": per_round,
        "profit_per_observed_round_95ci": [per_round - 1.96 * round_se, per_round + 1.96 * round_se] if round_se is not None else None,
        "return_per_initial_unit_bet": roi, "return_per_initial_unit_bet_95ci": roi_ci,
        "ci_method": "normal approximation with independent shoes as clusters",
        "profitable_shoe_fraction": float(np.mean(profits > 0)),
        "losing_shoe_fraction": float(np.mean(profits < 0)),
    }


def fit_network(observations, rewards, config):
    """Aggregate repeated features into sufficient statistics for squared loss.

    Half-count/depth bins balance efficient fitting with noisy outcomes. No
    analytical action targets or hidden-card labels are supplied to the learner.
    """
    count_bin = np.floor(np.clip(observations[:, 0], -15, 15) * 2).astype(int)
    depth_bin = np.floor(observations[:, 1] * 10).astype(int)
    cells, inverse, frequency = np.unique(np.column_stack((count_bin, depth_bin)), axis=0, return_inverse=True, return_counts=True)
    grid = np.column_stack(((cells[:, 0] + 0.5) / 2, (cells[:, 1] + 0.5) / 10))
    x = features(grid)
    y = torch.tensor(np.bincount(inverse, weights=rewards) / frequency, dtype=torch.float32)
    weight = torch.tensor(frequency / frequency.sum(), dtype=torch.float32)
    torch.manual_seed(config.seed)
    network = BetNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002, weight_decay=0.0001)
    for _ in range(config.epochs):
        loss = (weight * (network(x) - y).square()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return network, float(loss.item())


def run_edge(config: EdgeConfig, rules: ShoeRules, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    settings = {"experiment": asdict(config), "rules": asdict(rules),
                "fixed_rules": "S17; US peek; double on two cards; no split/surrender/insurance; all cards exposed after round",
                "betting": "one initial unit or skip before deal; double adds one unit; reference player continues when agent skips",
                "selection": "network fit on train shoes; threshold selected on validation; freeze before test",
                "limitations": "free observation and unrestricted mid-shoe entry; no bankroll cap, time costs, or table restrictions"}
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("Collecting training shoes (public pre-deal count only)...", flush=True)
    train_x, train_y, _ = collect(rules, config.train_shoes, config.seed, "train")
    network, loss = fit_network(train_x, train_y, config)
    print(f"Fitted neural betting values from {len(train_y):,} rounds; collecting validation shoes...", flush=True)
    val_x, val_y, val_ids = collect(rules, config.validation_shoes, config.seed, "validation")
    predictions = predict(network, val_x)
    candidates = []
    for threshold in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        report = summarize(val_y, val_ids, predictions > threshold, config.validation_shoes)
        candidates.append({"threshold": threshold, **report})
    # Choose a conservative threshold by lower confidence bound on units per
    # observed round. Tests are never used to tune this decision.
    best = max(candidates, key=lambda item: item["profit_per_observed_round_95ci"][0] if item["bets"] else float("-inf"))
    threshold = best["threshold"]
    validation = {"training_rounds": len(train_y), "weighted_train_loss": loss,
                  "selected_threshold": threshold, "candidates": candidates}
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    torch.save({"format_version": 1, "kind": "public_count_monte_carlo_bet_value",
                "threshold": threshold, "model": network.state_dict(), "rules": asdict(rules),
                "features": "pre-deal true count clipped to [-15,15] divided by 10; shoe fraction remaining"}, output / "bet_model.pt")
    print(f"Frozen bet threshold {threshold:.3f}; starting {config.test_batches} untouched test batches...", flush=True)
    reports, pooled_rewards, pooled_ids, pooled_selection = [], [], [], []
    for batch in range(config.test_batches):
        x, y, ids = collect(rules, config.test_shoes_per_batch, config.seed, f"test-{batch}")
        selected = predict(network, x) > threshold
        result = {"batch": batch, "neural_bet_or_skip": summarize(y, ids, selected, config.test_shoes_per_batch),
                  "always_bet_same_play": summarize(y, ids, np.ones(len(y), dtype=bool), config.test_shoes_per_batch)}
        reports.append(result)
        pooled_rewards.append(y)
        pooled_ids.append(ids + batch * config.test_shoes_per_batch)
        pooled_selection.append(selected)
        print(f"Test {batch + 1}: {result['neural_bet_or_skip']['bets']:,} bets, return/initial unit {result['neural_bet_or_skip']['return_per_initial_unit_bet']}, CI {result['neural_bet_or_skip']['return_per_initial_unit_bet_95ci']}", flush=True)
    all_y, all_ids, all_selected = np.concatenate(pooled_rewards), np.concatenate(pooled_ids), np.concatenate(pooled_selection)
    total_shoes = config.test_batches * config.test_shoes_per_batch
    pooled = summarize(all_y, all_ids, all_selected, total_shoes)
    always = summarize(all_y, all_ids, np.ones(len(all_y), dtype=bool), total_shoes)
    established = pooled["return_per_initial_unit_bet_95ci"] is not None and pooled["return_per_initial_unit_bet_95ci"][0] > 0
    # Negative control: when every round is a fresh shoe, past cards give no edge.
    fresh_decision = bool(predict(network, np.asarray([[0, 1]], dtype=np.float32))[0] > threshold)
    ceiling = {"original_rules_optimal_return": initial_value(optimal_values()), "fresh_shoe_bet_decision": fresh_decision}
    for name, path in {"original_dqn": Path("runs/dqn/model.pt"), "previous_rl": Path("runs/comparison/rl/model.pt")}.items():
        if path.exists():
            ceiling[name] = policy_report(load_policy(path).action)
    summary = {"status": "positive_edge_supported_in_this_simulation" if established else "edge_not_established",
               "scope": "finite-shoe neural bet-or-skip agent with fixed analytical hit/stand/double strategy",
               "original_game": ceiling, "rules_and_assumptions": settings,
               "selected_threshold": threshold, "test_batches": reports,
               "pooled_neural": pooled, "pooled_always_bet": always,
               "caveat": "Statistical evidence under specified simulator assumptions; not a mathematical proof or a guarantee for actual casinos."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "betting_policy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_count", "fraction_remaining", "estimated_return", "decision"])
        for fraction in (1.0, 0.75, 0.5, 0.3):
            states = np.asarray([[count, fraction] for count in range(-10, 16)], dtype=np.float32)
            for state, value in zip(states, predict(network, states)):
                writer.writerow([*state, float(value), "bet" if value > threshold else "observe"])
    return summary
