"""Browser sessions for the frozen public-count betting policy."""

from dataclasses import asdict
import csv
from pathlib import Path
import json
import secrets
import threading
import uuid

import numpy as np
import torch

from .edge import load_bettor, predict, shoe_seed
from .shoe import Shoe, play_table


class LiveBettor:
    def __init__(self, checkpoint: Path):
        self.network, self.threshold, self.rules = load_bettor(checkpoint)
        # Cache neural predictions for all public integer-count/cards-left pairs.
        # This is equivalent to inference on each round and never uses card order.
        self.max_count = 20 * self.rules.decks
        total_cards = self.rules.decks * 52
        counts, remaining = np.meshgrid(np.arange(-self.max_count, self.max_count + 1),
                                        np.arange(1, total_cards + 1), indexing="ij")
        observations = np.column_stack((counts.ravel() / np.maximum(remaining.ravel() / 52, .25),
                                        remaining.ravel() / total_cards))
        self.values = predict(self.network, observations).reshape(counts.shape)
        report_path = checkpoint.with_name("summary.json")
        self.evidence = None
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if "pooled_neural" in report:
                self.evidence = report["pooled_neural"]

    def decision(self, shoe):
        value = float(self.values[shoe.exposed_count + self.max_count, len(shoe.cards) - 1])
        return value, value > self.threshold


class EdgeSession:
    def __init__(self, bettor: LiveBettor, seed: int, service: "EdgeService | None" = None):
        self.bettor, self.seed, self.service = bettor, seed, service
        self.lock = threading.Lock()
        self.shoe_index = 0
        self.shoe = self._new_shoe()
        self.rounds = self.bets = self.wins = self.losses = self.pushes = 0
        self.profit = self.baseline = 0.0
        self.sized_profit = self.sized_baseline = 0.0
        self.bankroll_units = 400.0
        if self.service is not None:
            sizing = getattr(self.service, "sizing", None)
            if sizing is not None:
                self.bankroll_units = sizing.bankroll_units
        self.staked_units = 0
        self.current_profit = self.current_bets = 0
        self.completed = []
        self.last = None
        self.history = [{"round": 0, "profit": 0, "baseline": 0}]

    def _new_shoe(self):
        return Shoe(self.bettor.rules, shoe_seed(self.seed, "live", self.shoe_index))

    def advance(self, count: int, until_bet=False):
        if type(count) is not int or not 1 <= count <= 10_000:
            raise ValueError("round count must be an integer from 1 to 10,000")
        with self.lock:
            for _ in range(count):
                true_count, fraction = self.shoe.observation()
                estimated, _ = self.bettor.decision(self.shoe)
                # Commit selected before the first draw. Never select by outcome.
                # Every hand (bet or observed) uses the same session play
                # policy, so betting never changes the card stream.
                if self.service:
                    actions, pairs = self.service.round_tables(true_count, fraction)
                    stake = self.service.stake_for(estimated, self.bankroll_units)
                    selected = stake > 0
                else:
                    actions, pairs, stake = play_table(), None, 1
                    _, selected = self.bettor.decision(self.shoe)
                reward = self.shoe.play_round(actions, pairs, capture=True)
                profit = reward if selected else 0.0
                sized = stake * reward if selected else 0.0
                self.rounds += 1
                self.bets += int(selected)
                self.profit += profit
                self.baseline += reward
                self.sized_profit += sized
                self.sized_baseline += stake * reward
                self.bankroll_units += sized
                self.staked_units += stake if selected else 0
                self.current_profit += profit
                self.current_bets += int(selected)
                if selected:
                    self.wins += reward > 0
                    self.losses += reward < 0
                    self.pushes += reward == 0
                self.last = {**self.shoe.last_round, "round": self.rounds,
                             "selected": selected, "profit": profit, "stake": stake if selected else 0,
                             "sized_profit": sized,
                             "predeal_true_count": true_count, "predeal_fraction": fraction,
                             "estimated_return": estimated, "shoe": self.shoe_index + 1}
                if self.rounds % 100 == 0:
                    self.history.append({"round": self.rounds, "profit": self.profit, "baseline": self.baseline})
                    if len(self.history) > 400:
                        self.history = self.history[::2]
                if not self.shoe.can_deal:
                    self.completed.append((self.current_profit, self.current_bets))
                    self.current_profit = self.current_bets = 0
                    self.shoe_index += 1
                    self.shoe = self._new_shoe()
                if selected and until_bet:
                    break
            return self.payload()

    def payload(self):
        expected, bet = self.bettor.decision(self.shoe)
        true_count, fraction = self.shoe.observation()
        interval, complete_roi = None, None
        # CI only uses completed shoes. A live-session interval is descriptive,
        # not a fixed-horizon significance test; do not stop when it looks good.
        if len(self.completed) >= 100:
            complete = np.asarray(self.completed, dtype=float)
            profit, bets = complete[:, 0], complete[:, 1]
            if np.count_nonzero(bets) >= 30:
                complete_roi = float(profit.sum() / bets.sum())
                se = float(np.std(profit - complete_roi * bets, ddof=1) / np.sqrt(len(bets)) / bets.mean())
                interval = [complete_roi - 1.96 * se, complete_roi + 1.96 * se]
        end = {"round": self.rounds, "profit": self.profit, "baseline": self.baseline}
        history = self.history if self.history[-1]["round"] == self.rounds else [*self.history, end]
        stake = 0
        if self.service:
            stake = self.service.stake_for(expected, self.bankroll_units) if bet else 0
        return {"seed": self.seed, "rounds": self.rounds, "bets": self.bets,
                "skipped": self.rounds - self.bets, "profit": self.profit, "baseline_profit": self.baseline,
                "sized_profit": self.sized_profit, "sized_baseline_profit": self.sized_baseline,
                "bankroll_units": self.bankroll_units, "staked_units": self.staked_units,
                "wins": self.wins, "losses": self.losses, "pushes": self.pushes,
                "return_per_initial_unit": self.profit / self.bets if self.bets else None,
                "completed_shoes": len(self.completed), "completed_shoes_return": complete_roi,
                "completed_shoes_95ci": interval, "history": history, "last_round": self.last,
                "next": {"action": "bet" if bet else "observe", "estimated_return": expected,
                         "stake_units": stake,
                         "true_count": true_count, "fraction_remaining": fraction,
                         "cards_remaining": len(self.shoe.cards), "shoe": self.shoe_index + 1},
                "rules": asdict(self.bettor.rules), "threshold": self.bettor.threshold,
                "play_policy": self.service.play_policy_name if self.service else "analytical hit/stand/double",
                "sizing": self.service.sizing_name if self.service else "flat 1 unit"}


class EdgeService:
    def __init__(self, checkpoint: Path, player: Path | None = None, sizing: Path | None = None):
        from .shoe import pair_table
        self.bettor = LiveBettor(checkpoint)
        self.fixed = play_table()
        self.pairs = pair_table(self.bettor.rules.double_after_split)
        self.player = None
        self.play_policy_name = "analytical hit/stand/double"
        if player is None:
            candidate = Path("runs/count-distilled/model.pt")
            player = candidate if candidate.is_file() else None
        if player is not None:
            from .count_rl import load_count_policy
            self.player = load_count_policy(player)
            self.play_policy_name = "neural count-aware strategy (distilled analytical play)"
        from .sizing import SizingConfig
        self.sizing = SizingConfig()
        self.sizing_name = "flat 1 unit"
        if sizing is None:
            candidate = Path("runs/sizing/sizing.json")
            sizing = candidate if candidate.is_file() else None
        if sizing is not None:
            try:
                selected = json.loads(Path(sizing).read_text(encoding="utf-8"))["selected_sizing"]["config"]
                self.sizing = SizingConfig(kelly_fraction=selected["kelly_fraction"],
                                           cap_units=selected["cap_units"], min_edge=selected["min_edge"],
                                           bankroll_units=selected.get("bankroll_units", 400.0))
                self.sizing_name = (f"half-Kelly 1-{self.sizing.cap_units} at "
                                    f"{self.sizing.kelly_fraction}-Kelly (validated)")
            except (KeyError, ValueError, OSError):
                pass
        self.sessions = {}
        self.lock = threading.Lock()
        self._table_cache: dict[tuple[float, float], dict] = {}

    def stake_for(self, estimated_edge: float, bankroll_units: float) -> int:
        from .sizing import kelly_units
        if self.sizing_name == "flat 1 unit":
            return 1 if estimated_edge > self.bettor.threshold else 0
        return kelly_units(estimated_edge, bankroll_units, self.sizing)

    def bankroll_sim(self, bankroll_dollars: float, unit_dollars: float,
                     max_shoes: int, seed: int) -> dict:
        """Fun bankroll simulator: stake the frozen models with YOUR money.

        Kelly-sized betting (cap scaled to the roll) vs flat 1-unit betting on
        the same deals. Stops early if the roll doubles or is ruined. Stateless:
        uses fresh reserved shoe streams, never touches live sessions.
        """
        from .edge import collect, predict
        from .sizing import SizingConfig, kelly_units
        if isinstance(bankroll_dollars, bool) or not isinstance(bankroll_dollars, (int, float)):
            raise ValueError("bankroll must be a number between $25 and $10,000,000")
        if not (25 <= bankroll_dollars <= 10_000_000):
            raise ValueError("bankroll must be between $25 and $10,000,000")
        if unit_dollars not in (5, 10, 25, 100):
            raise ValueError("unit must be one of $5, $10, $25, $100")
        if type(max_shoes) is not int or not 1 <= max_shoes <= 2000:
            raise ValueError("shoes must be an integer from 1 to 2,000")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer from 0 to 4,294,967,295")
        start_units = bankroll_dollars / unit_dollars
        base = self.sizing
        cap = max(1, round(base.cap_units * start_units / 400.0))
        cfg = SizingConfig(kelly_fraction=base.kelly_fraction, cap_units=cap,
                           min_edge=base.min_edge, bankroll_units=start_units)
        kelly_roll, flat_roll = start_units, start_units
        kelly_peak, flat_peak = start_units, start_units
        kelly_worst, flat_worst = 1.0, 1.0
        kelly_bets = kelly_staked = kelly_max = 0
        flat_bets = flat_staked = 0
        rounds = shoes = 0
        kelly_curve, flat_curve = [[0, bankroll_dollars]], [[0, bankroll_dollars]]
        outcome, chunk, chunk_shoes = "capped", 0, 250
        while shoes < max_shoes and outcome == "capped":
            take = min(chunk_shoes, max_shoes - shoes)
            observations, rewards, _ = collect(self.bettor.rules, take, seed, f"sim-{seed}-{chunk}")
            edges = predict(self.bettor.network, observations)
            for i in range(len(rewards)):
                stake = kelly_units(float(edges[i]), kelly_roll, cfg)
                if stake:
                    kelly_bets += 1
                    kelly_staked += stake
                    kelly_max = max(kelly_max, stake)
                kelly_roll += stake * float(rewards[i])
                if edges[i] > self.bettor.threshold:
                    flat_bets += 1
                    flat_staked += 1
                    flat_roll += float(rewards[i])
                rounds += 1
                kelly_peak = max(kelly_peak, kelly_roll)
                flat_peak = max(flat_peak, flat_roll)
                kelly_worst = min(kelly_worst, kelly_roll / start_units)
                flat_worst = min(flat_worst, flat_roll / start_units)
                if rounds % 25 == 0:
                    kelly_curve.append([rounds, round(kelly_roll * unit_dollars, 2)])
                    flat_curve.append([rounds, round(flat_roll * unit_dollars, 2)])
                if kelly_roll <= 0:
                    outcome = "ruined"
                    break
            shoes += take
            chunk += 1
            if kelly_roll >= 2 * start_units:
                outcome = "doubled"
        kelly_curve.append([rounds, round(max(kelly_roll, 0) * unit_dollars, 2)])
        flat_curve.append([rounds, round(max(flat_roll, 0) * unit_dollars, 2)])
        dollars = lambda u: round(u * unit_dollars, 2)
        return {
            "seed": seed, "start_dollars": bankroll_dollars, "unit_dollars": unit_dollars,
            "outcome": outcome, "rounds": rounds, "shoes": shoes,
            "sizing": f"half-Kelly 1-{cap} (scaled to your roll)",
            "kelly": {"terminal_dollars": dollars(max(kelly_roll, 0)), "peak_dollars": dollars(kelly_peak),
                      "worst_fraction": round(kelly_worst, 4), "bets": kelly_bets,
                      "staked_units": kelly_staked, "max_stake_units": kelly_max, "curve": kelly_curve},
            "flat": {"terminal_dollars": dollars(max(flat_roll, 0)), "peak_dollars": dollars(flat_peak),
                     "worst_fraction": round(flat_worst, 4), "bets": flat_bets,
                     "staked_units": flat_staked, "curve": flat_curve},
            "note": "Simulation on reserved shoes with the frozen models. Short runs can lose; past sims are not a promise.",
        }

    def round_tables(self, true_count: float, fraction: float) -> tuple[dict, dict]:
        """(hit/stand/double table, pair-split table) for this round's count."""
        return self.actions_for(true_count, fraction), self.pairs

    def actions_for(self, true_count: float, fraction: float) -> dict:
        """Session play table at this round's pre-deal count (cached on a grid)."""
        if self.player is None:
            return self.fixed
        key = (round(float(true_count) * 2) / 2, round(float(fraction) * 20) / 20)
        with self.lock:
            cached = self._table_cache.get(key)
        if cached is not None:
            return cached
        from .count_rl import encode_count_state
        import numpy as np
        grid, keys = [], []
        for total in range(4, 22):
            for up in range(1, 11):
                for ace in (False, True):
                    if ace and (total < 12 or total > 21):
                        continue
                    if not ace and total > 21:
                        continue
                    for first in (True, False):
                        grid.append(encode_count_state(total, up, int(ace), key[0], key[1], first))
                        keys.append((total, up, ace, first))
        with torch.no_grad():
            choices = self.player.network(
                torch.from_numpy(np.asarray(grid))).argmax(dim=1).tolist()
        table = {(t, u, a, f): (action if f else min(action, 1))
                 for (t, u, a, f), action in zip(keys, choices)}
        with self.lock:
            self._table_cache[key] = table
            if len(self._table_cache) > 500:
                self._table_cache.pop(next(iter(self._table_cache)))
        return table

    def new(self, seed=None):
        if seed is None:
            seed = secrets.randbits(32)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer from 0 to 4,294,967,295")
        session = EdgeSession(self.bettor, seed, self)
        identifier = uuid.uuid4().hex
        with self.lock:
            self.sessions[identifier] = session
            if len(self.sessions) > 50:
                self.sessions.pop(next(iter(self.sessions)))
        return {"id": identifier, **session.payload()}

    def get(self, identifier):
        with self.lock:
            session = self.sessions.get(identifier)
        if session is None:
            raise KeyError("Session expired. Start a new session.")
        return session

    def advance(self, identifier, count, until_bet=False):
        return {"id": identifier, **self.get(identifier).advance(count, until_bet)}


def training_history():
    """Read recorded learning curves. Do not synthesize missing epoch histories."""
    runs = []
    for name, directory in (("RL from scratch", "rl"), ("Pretraining + RL", "mixed")):
        root = Path("runs/comparison") / directory
        if not (root / "training.csv").is_file():
            continue
        with (root / "training.csv").open(encoding="utf-8", newline="") as file:
            points = [{key: float(value) if value else None for key, value in row.items()}
                      for row in csv.DictReader(file)]
        report = json.loads((root / "report.json").read_text(encoding="utf-8")) if (root / "report.json").is_file() else None
        runs.append({"id": directory, "name": name, "points": points, "report": report,
                     "status": "complete" if report else "in progress"})
    unsupervised_path = Path("runs/comparison/unsupervised/report.json")
    unsupervised = json.loads(unsupervised_path.read_text(encoding="utf-8")) if unsupervised_path.exists() else None
    validation_path = Path("runs/edge/validation.json")
    betting = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    count = None
    distilled_path = Path("runs/count-distilled/report.json")
    if distilled_path.exists():
        count = {"distilled": json.loads(distilled_path.read_text(encoding="utf-8"))}
        for key, path in (("rl_from_scratch", Path("runs/count/evaluation.json")),
                          ("rl_fine_tuned", Path("runs/count-tuned/evaluation.json"))):
            if path.exists():
                report = json.loads(path.read_text(encoding="utf-8"))
                count[key] = {"best_validation": report.get("best_validation"),
                              "test_neural": report.get("test_neural"),
                              "test_fixed_strategy": report.get("test_fixed_strategy")}
    return {"runs": runs, "unsupervised": unsupervised, "betting": betting, "count": count,
            "optimal_return": -0.0465559770666975,
            "note": "Recorded checkpoints from completed runs, not a new training job. Exact return measures the original unlimited-deck game."}
