"""Count-aware neural DQN for finite-shoe blackjack.

The classic unlimited-deck game is provably unbeatable (exact optimal
-0.0466/hand): no neural net can fix that. This module learns where an
edge CAN exist: six-deck shoes with Hi-Lo counting, 3:2 blackjacks,
dealer stands on soft 17, and double on any first two cards.

A single neural Q-network sees the hand (one-hot total/dealer/ace) PLUS
the pre-deal Hi-Lo true count and shoe fraction remaining, and chooses
hit/stand/double. It is trained with Double DQN, experience replay, and
a frozen target network. The fixed analytical play table from shoe.py
is the baseline, not the learner.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agent import encode
from .edge import shoe_seed
from .oracle import add_card
from .shoe import Shoe, ShoeRules, count_tag, play_table

STAND, HIT, DOUBLE = 0, 1, 2
ACTION_NAMES = ("stand", "hit", "double")
COUNT_DIM = 46  # 43 hand one-hot + true-count/10 + fraction + can-double


def count_features(true_count: float, fraction: float) -> tuple[float, float]:
    return float(np.clip(true_count, -15, 15) / 10), float(fraction)


def encode_count_state(total: int, dealer: int, ace: int, true_count: float,
                       fraction: float, can_double: bool) -> np.ndarray:
    hand = encode((total, dealer, ace))
    tc, frac = count_features(true_count, fraction)
    extra = np.asarray([tc, frac, 1.0 if can_double else 0.0], dtype=np.float32)
    return np.concatenate([hand, extra]).astype(np.float32)


class CountQNetwork(nn.Module):
    """Neural Q-function: 46-dim count-aware state -> hit/stand/double values."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.layers = nn.Sequential(
            nn.Linear(COUNT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, states):
        return self.layers(states)


class CountReplay:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, COUNT_DIM), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = self.position = 0

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size):
        indices = self.rng.choice(self.size, size, replace=False)
        return tuple(torch.from_numpy(values[indices]) for values in (
            self.states, self.actions, self.rewards, self.next_states, self.terminated))


class CountHand:
    """One interactive hand inside a running shoe; count frozen at deal."""

    def __init__(self, shoe: Shoe):
        self.shoe = shoe
        cards = shoe.cards
        if len(cards) < 24:
            raise RuntimeError("shoe exhausted; reshuffle before dealing")

        def draw():
            card = cards.pop()
            self.drawn.append(card)
            return card

        self.drawn: list[int] = []
        self.true_count, self.fraction = shoe.observation()
        p1, up, p2, hole = draw(), draw(), draw(), draw()
        self.player = [p1, p2]
        self.dealer = [up, hole]
        self.up = up
        self.stake = 1.0
        self.first = True
        self.done = False
        self.reward = 0.0
        player_total, _ = add_card(*add_card(0, False, p1), p2)
        dealer_total, _ = add_card(*add_card(0, False, up), hole)
        # US peek: round ends immediately on a dealer natural.
        if dealer_total == 21:
            self.reward = 0.0 if player_total == 21 else -1.0
            self.natural = True
            self.done = True
        elif player_total == 21:
            self.reward = shoe.rules.blackjack_payout
            self.natural = True
            self.done = True
        else:
            self.natural = False

    def state_vector(self) -> np.ndarray:
        total, ace = add_card(*add_card(0, False, self.player[0]), self.player[1]) \
            if len(self.player) == 2 else self._total_ace()
        return encode_count_state(total, self.up, int(ace), self.true_count,
                                  self.fraction, self.first)

    def _total_ace(self) -> tuple[int, bool]:
        total, ace = 0, False
        for card in self.player:
            total, ace = add_card(total, ace, card)
        return total, ace

    def legal(self, action: int) -> bool:
        return action != DOUBLE or self.first

    def step(self, action: int) -> float:
        if self.done:
            raise ValueError("hand is already finished")
        if action == DOUBLE and not self.first:
            raise ValueError("double is only legal on the first decision")
        cards = self.shoe.cards

        def draw():
            if not cards:
                raise RuntimeError("shoe exhausted mid-hand")
            card = cards.pop()
            self.drawn.append(card)
            return card

        total, ace = self._total_ace()
        if action == STAND:
            pass
        elif action == HIT:
            card = draw()
            self.player.append(card)
            total, ace = add_card(total, ace, card)
            self.first = False
            if total > 21:
                return self._settle(-self.stake)
            return 0.0  # intermediate: no reward yet
        elif action == DOUBLE:
            card = draw()
            self.player.append(card)
            total, ace = add_card(total, ace, card)
            self.stake = 2.0
            self.first = False
            if total > 21:
                return self._settle(-self.stake)
        else:
            raise ValueError("action must be 0 (stand), 1 (hit), or 2 (double)")
        # Stand or successful double: dealer plays S17.
        dealer_total, dealer_ace = add_card(*add_card(0, False, self.dealer[0]), self.dealer[1])
        while dealer_total < 17:
            card = draw()
            self.dealer.append(card)
            dealer_total, dealer_ace = add_card(dealer_total, dealer_ace, card)
        total, _ = self._total_ace()
        if dealer_total > 21 or total > dealer_total:
            return self._settle(self.stake)
        if total < dealer_total:
            return self._settle(-self.stake)
        return self._settle(0.0)

    def _settle(self, reward: float) -> float:
        self.done = True
        self.reward = float(reward)
        self.shoe.exposed_count += sum(count_tag(c) for c in self.drawn)
        self.shoe.round_count += 1
        return float(reward)


@dataclass
class CountConfig:
    hands: int = 200_000
    seed: int = 917
    hidden: int = 128
    learning_rate: float = 0.0003
    gamma: float = 1.0
    batch_size: int = 128
    capacity: int = 100_000
    warmup: int = 10_000
    train_every: int = 4
    target_every: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_fraction: float = 0.85
    check_every: int = 25_000
    val_shoes: int = 300
    warm_start: str | None = None  # path to a distilled/frozen net to fine-tune

    def epsilon(self, hand: int) -> float:
        start = 0.25 if self.warm_start else self.epsilon_start
        fraction = min(hand / max(1, self.hands * self.decay_fraction), 1.0)
        return start + fraction * (self.epsilon_end - start)


class CountTrainer:
    def __init__(self, config: CountConfig):
        self.config = config
        torch.manual_seed(config.seed)
        self.network = CountQNetwork(config.hidden)
        self.target = deepcopy(self.network).requires_grad_(False)
        self.rng = random.Random(config.seed)
        self.replay = CountReplay(config.capacity, config.seed)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def greedy(self, state: np.ndarray, can_double: bool) -> int:
        with torch.no_grad():
            values = self.network(torch.from_numpy(state)).tolist()
        if not can_double:
            values[DOUBLE] = float("-inf")
        return int(int(np.argmax(values)))

    def action(self, state: np.ndarray, can_double: bool, epsilon: float = 0.0) -> int:
        legal = (STAND, HIT, DOUBLE) if can_double else (STAND, HIT)
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.choice(legal)
        return self.greedy(state, can_double)

    def learn(self) -> float:
        states, actions, rewards, next_states, terminated = self.replay.sample(self.config.batch_size)
        predicted = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
            future = self.target(next_states).gather(1, next_actions).squeeze(1)
            expected = rewards + self.config.gamma * (1.0 - terminated) * future
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())


class ShoeRunner:
    """Deal hands sequentially from fresh shoes keyed by (master, split, index)."""

    def __init__(self, rules: ShoeRules, master: int, split: str):
        self.rules, self.master, self.split = rules, master, split
        self.shoe_index = -1
        self.shoe: Shoe | None = None
        self._next_shoe()

    def _next_shoe(self):
        self.shoe_index += 1
        self.shoe = Shoe(self.rules, shoe_seed(self.master, self.split, self.shoe_index))

    def deal(self) -> CountHand:
        assert self.shoe is not None
        if not self.shoe.can_deal:
            self._next_shoe()
        return CountHand(self.shoe)


def play_shoes(action_for, rules: ShoeRules, shoes: int, master: int, split: str):
    """Play full shoes with a per-decision callback; returns per-shoe profits."""
    runner = ShoeRunner(rules, master, split)
    profits, hands = [], 0
    for _ in range(shoes):
        runner._next_shoe()
        profit = 0.0
        while runner.shoe.can_deal:
            hand = CountHand(runner.shoe)
            if hand.done:
                hand._settle(hand.reward)
                profit += hand.reward
                hands += 1
                continue
            while not hand.done:
                total, ace = hand._total_ace()
                action = action_for(total, hand.up, int(ace), hand.true_count,
                                    hand.fraction, hand.first)
                if action == DOUBLE and not hand.first:
                    action = HIT
                reward = hand.step(action)
            profit += hand.reward
            hands += 1
        profits.append(profit)
    return profits, hands


def summarize_shoes(profits: list[float], hands: int, shoes: int) -> dict:
    arr = np.asarray(profits, dtype=float)
    mean_shoe = float(arr.mean())
    se_shoe = float(arr.std(ddof=1) / math.sqrt(shoes)) if shoes > 1 else None
    per_hand = float(arr.sum() / hands)
    se_hand = float(arr.std(ddof=1) / math.sqrt(shoes) / (hands / shoes)) if shoes > 1 else None
    return {
        "shoes": shoes, "hands": hands,
        "profit_units": float(arr.sum()),
        "profit_per_hand": per_hand,
        "profit_per_hand_95ci": [per_hand - 1.96 * se_hand, per_hand + 1.96 * se_hand] if se_hand else None,
        "profit_per_shoe_95ci": [mean_shoe - 1.96 * se_shoe, mean_shoe + 1.96 * se_shoe] if se_shoe else None,
    }


def fixed_action(actions: dict):
    def choose(total, up, ace, _tc, _frac, first):
        return actions[total, up, bool(ace), first]
    return choose


class CountPolicy:
    """Frozen count-aware neural policy for inference (play + UI advice)."""

    def __init__(self, network: CountQNetwork):
        self.network = network.eval()

    @torch.no_grad()
    def values_for(self, total, dealer, ace, true_count, fraction, can_double):
        vec = encode_count_state(total, dealer, ace, true_count, fraction, can_double)
        return self.network(torch.from_numpy(vec)).tolist()

    @torch.no_grad()
    def action_for(self, total, dealer, ace, true_count, fraction, can_double):
        qs = self.values_for(total, dealer, ace, true_count, fraction, can_double)
        if not can_double:
            qs[DOUBLE] = float("-inf")
        return int(int(np.argmax(qs)))

    def callback(self):
        def choose(total, up, ace, tc, frac, first):
            return self.action_for(total, up, ace, tc, frac, first)
        return choose


def load_count_policy(path: Path) -> CountPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("kind") != "count_aware_double_dqn" or data.get("format_version") != 1:
        raise ValueError("Expected a version-1 count-aware DQN checkpoint")
    network = CountQNetwork(data["metadata"]["hidden"])
    network.load_state_dict(data["model"])
    return CountPolicy(network)


def distill_fixed_strategy(hidden: int, seed: int, output: Path,
                           epochs: int = 300) -> dict:
    """Supervised warm start: teach the SAME neural net the analytical strategy.

    The fixed play table is count-independent, so the student learns correct
    baseline play on a grid of hand states x true counts x shoe fractions.
    Count inputs start near-irrelevant; RL fine-tuning then only needs to
    discover deviations (e.g. doubling more at high counts). This is
    behavior cloning, reported as such - not RL.
    """
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    fixed = play_table()
    states, targets = [], []
    for ace in (0, 1):
        for total in range(12 if ace else 4, 22):
            for dealer in range(1, 11):
                for tc in (-10, -5, -2, 0, 2, 5, 10):
                    for frac in (0.3, 0.6, 1.0):
                        for first in (True, False):
                            states.append(encode_count_state(total, dealer, ace, tc, frac, first))
                            action = fixed[total, dealer, bool(ace), first]
                            targets.append(action if first else min(action, 1))
    x = torch.from_numpy(np.asarray(states))
    y = torch.tensor(targets, dtype=torch.long)
    network = CountQNetwork(hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    for _ in range(epochs):
        loss = nn.functional.cross_entropy(network(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        agreement = (network(x).argmax(dim=1) == y).float().mean().item()
    torch.save({"format_version": 1, "kind": "count_aware_double_dqn",
                "metadata": {"hidden": hidden, "seed": seed,
                             "distilled_agreement": agreement,
                             "note": "supervised clone of the analytical play table; RL fine-tuning starts here"},
                "model": network.state_dict()}, output / "model.pt")
    report = {"kind": "distilled_analytical_strategy", "grid_states": len(states),
              "epochs": epochs, "grid_agreement": agreement}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_count_rl(config: CountConfig, rules: ShoeRules, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    (output / "config.json").write_text(
        json.dumps({"training": asdict(config), "rules": asdict(rules)}, indent=2), encoding="utf-8")
    fixed = play_table()
    trainer = CountTrainer(config)
    if config.warm_start:
        data = torch.load(config.warm_start, map_location="cpu", weights_only=True)
        if data.get("kind") != "count_aware_double_dqn":
            raise ValueError("warm start must be a count-aware DQN checkpoint")
        trainer.network.load_state_dict(data["model"])
        trainer.target.load_state_dict(data["model"])
        print(f"count-rl: warm-started from {config.warm_start}", flush=True)
    runner = ShoeRunner(rules, config.seed, "train")
    steps = 0
    loss = None
    reward_sum = 0.0
    reward_count = 0
    best = {"profit_per_hand": float("-inf")}
    best_state = None
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["hands", "steps", "epsilon", "exploratory_profit_per_hand",
                                                  "loss", "val_profit_per_hand", "fixed_val_profit_per_hand"])
        writer.writeheader()
        for hand_index in range(config.hands):
            epsilon = config.epsilon(hand_index)
            hand = runner.deal()
            if hand.done:
                hand._settle(hand.reward)
                reward_sum += hand.reward
                reward_count += 1
            else:
                state = hand.state_vector()
                first = True
                while not hand.done:
                    action = trainer.action(state, hand.first, epsilon)
                    saved_first = hand.first
                    immediate = hand.step(action)
                    reward_sum += hand.reward if hand.done else 0.0
                    reward_count += 1 if hand.done else 0
                    if hand.done:
                        trainer.replay.add(state, action, immediate,
                                           np.zeros(COUNT_DIM, dtype=np.float32), 1.0)
                    else:
                        nxt = hand.state_vector()
                        trainer.replay.add(state, action, 0.0, nxt, 0.0)
                        state = nxt
                    first = saved_first
                    steps += 1
                    if trainer.replay.size >= max(config.warmup, config.batch_size) and steps % config.train_every == 0:
                        loss = trainer.learn()
                    if steps % config.target_every == 0 and steps:
                        trainer.target.load_state_dict(trainer.network.state_dict())
            done_hands = hand_index + 1
            if done_hands % config.check_every == 0 or done_hands == config.hands:
                val, val_hands = play_shoes(
                    lambda t, u, a, tc, fr, f: trainer.greedy(
                        encode_count_state(t, u, a, tc, fr, f), f),
                    rules, config.val_shoes, config.seed, f"val-{done_hands}")
                fixed_val, _ = play_shoes(fixed_action(fixed), rules, config.val_shoes,
                                          config.seed, f"val-{done_hands}")
                val_mean = float(np.sum(val) / val_hands)
                fixed_mean = float(np.sum(fixed_val) / val_hands)
                writer.writerow({"hands": done_hands, "steps": steps, "epsilon": epsilon,
                                 "exploratory_profit_per_hand": reward_sum / max(1, reward_count),
                                 "loss": loss, "val_profit_per_hand": val_mean,
                                 "fixed_val_profit_per_hand": fixed_mean})
                file.flush()
                print(f"count-rl: hand {done_hands:,}/{config.hands:,} | val {val_mean:+.5f}/hand | fixed {fixed_mean:+.5f}/hand", flush=True)
                if val_mean > best["profit_per_hand"]:
                    best = {"profit_per_hand": val_mean, "hands": done_hands,
                            "fixed_profit_per_hand": fixed_mean}
                    best_state = {k: v.detach().clone() for k, v in trainer.network.state_dict().items()}
                reward_sum = reward_count = 0
    trainer.network.load_state_dict(best_state)
    torch.save({"format_version": 1, "kind": "count_aware_double_dqn",
                "metadata": {"hidden": config.hidden, "seed": config.seed,
                             "best_val_profit_per_hand": best["profit_per_hand"]},
                "model": trainer.network.state_dict()}, output / "model.pt")
    neural_test, test_hands = play_shoes(
        lambda t, u, a, tc, fr, f: trainer.greedy(
            encode_count_state(t, u, a, tc, fr, f), f),
        rules, 2000, config.seed, "test-neural")
    fixed_test, fixed_hands = play_shoes(fixed_action(fixed), rules, 2000, config.seed, "test-fixed")
    report = {"best_validation": best,
              "test_neural": summarize_shoes(neural_test, test_hands, 2000),
              "test_fixed_strategy": summarize_shoes(fixed_test, fixed_hands, 2000)}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "strategy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["player_total", "dealer_card", "usable_ace", "true_count",
                         "can_double", "action", "q_stand", "q_hit", "q_double"])
        trainer.network.eval()
        with torch.no_grad():
            for ace in (0, 1):
                for total in range(12 if ace else 4, 22):
                    for dealer in range(1, 11):
                        for tc in (-5, 0, 2, 5):
                            for first in (True, False):
                                vec = encode_count_state(total, dealer, ace, tc, 0.5, first)
                                qs = trainer.network(torch.from_numpy(vec)).tolist()
                                masked = list(qs)
                                if not first:
                                    masked[DOUBLE] = float("-inf")
                                writer.writerow([total, dealer, ace, tc, int(first),
                                                 ACTION_NAMES[int(np.argmax(masked))], *qs])
    return report
