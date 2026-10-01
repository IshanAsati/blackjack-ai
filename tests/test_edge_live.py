import numpy as np
import pytest
import torch
from dataclasses import asdict
from pathlib import Path

from blackjackai.count_rl import distill_fixed_strategy
from blackjackai.edge import BetNetwork, predict
from blackjackai.edge_live import EdgeService, EdgeSession, LiveBettor
from blackjackai.shoe import ShoeRules


@pytest.fixture
def bettor(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    net = BetNetwork()
    path = tmp_path / "bet_model.pt"
    torch.save({"kind": "public_count_monte_carlo_bet_value", "format_version": 1,
                "model": net.state_dict(), "threshold": 0.02, "rules": asdict(ShoeRules())}, path)
    return LiveBettor(path)


def test_cached_decisions_match_frozen_network(bettor):
    session = EdgeSession(bettor, 123)
    for _ in range(100):
        direct = float(predict(bettor.network, np.asarray([session.shoe.observation()]))[0])
        cached, selected = bettor.decision(session.shoe)
        assert cached == pytest.approx(direct, abs=2e-7)
        assert selected == (direct > bettor.threshold)
        session.advance(1)


def test_no_bets_keeps_zero_profit_while_shoe_and_baseline_advance(bettor):
    bettor.threshold = float("inf")
    session = EdgeSession(bettor, 123)
    result = session.advance(1000)
    assert result["rounds"] == result["skipped"] == 1000
    assert result["bets"] == result["profit"] == 0
    assert result["completed_shoes"] > 0
    assert result["next"]["shoe"] > 1
    assert result["last_round"]["selected"] is False


def test_all_bets_match_control_and_chunking_preserves_outcomes(bettor):
    bettor.threshold = -float("inf")
    batched, stepped = EdgeSession(bettor, 909), EdgeSession(bettor, 909)
    result = batched.advance(200)
    for _ in range(200):
        other = stepped.advance(1)
    assert result == other
    assert result["profit"] == result["baseline_profit"]
    assert result["bets"] == result["wins"] + result["losses"] + result["pushes"] == 200


def test_last_hand_trace_includes_all_exposed_cards(bettor):
    result = EdgeSession(bettor, 310).advance(1)
    last = result["last_round"]
    assert last["predeal_true_count"] == 0
    assert last["predeal_fraction"] == 1
    assert len(last["player"]) + len(last["dealer"]) == 312 - result["next"]["cards_remaining"]
    assert last["profit"] == (last["reward"] if last["selected"] else 0)


def test_until_bet_stops_on_first_wager_even_if_it_loses(bettor):
    bettor.threshold = -float("inf")
    result = EdgeSession(bettor, 444).advance(10000, until_bet=True)
    assert result["rounds"] == result["bets"] == 1


@pytest.mark.parametrize("count", [0, -1, 10001, True, 1.5, "10"])
def test_invalid_batches_do_not_advance_session(bettor, count):
    session = EdgeSession(bettor, 1)
    with pytest.raises(ValueError):
        session.advance(count)
    assert session.rounds == 0


def test_neural_play_policy_serves_cached_tables(bettor, tmp_path):
    distill_fixed_strategy(32, 11, tmp_path / "player", epochs=150)
    from blackjackai.count_rl import load_count_policy
    from blackjackai.shoe import pair_table, play_table
    player = load_count_policy(tmp_path / "player" / "model.pt")

    class StubService:
        play_policy_name = "test"
        sizing_name = "flat 1 unit"

        def __init__(self):
            import threading
            self.lock = threading.Lock()
            self._table_cache = {}

        def actions_for(self, true_count, fraction):
            key = (round(float(true_count) * 2) / 2, round(float(fraction) * 20) / 20)
            with self.lock:
                cached = self._table_cache.get(key)
            if cached is not None:
                return cached
            from blackjackai.count_rl import encode_count_state
            import numpy as np
            import torch
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
                choices = player.network(torch.from_numpy(np.asarray(grid))).argmax(dim=1).tolist()
            table = {(t, u, a, f): (action if f else min(action, 1))
                     for (t, u, a, f), action in zip(keys, choices)}
            with self.lock:
                self._table_cache[key] = table
            return table

        def round_tables(self, true_count, fraction):
            return self.actions_for(true_count, fraction), pair_table(True)

        def stake_for(self, estimated_edge, bankroll_units):
            return 1 if estimated_edge > bettor.threshold else 0

    service = StubService()
    first = service.actions_for(2.0, 0.7)
    second = service.actions_for(2.04, 0.71)
    assert first is second  # same grid cell: cache hit
    assert service.actions_for(-3.0, 0.4) is not first
    assert first[(16, 10, False, True)] in (0, 1, 2)
    session = EdgeSession(bettor, 77, service)
    result = session.advance(50)
    assert result["rounds"] == 50
    assert result["play_policy"] == "test"
    assert result["last_round"]["moves"]
    assert all(move in ("stand", "hit", "double") for move in result["last_round"]["moves"])
    repeat = EdgeSession(bettor, 77, service).advance(50)
    assert repeat["profit"] == result["profit"]


def test_sized_session_stakes_scale_with_edge(bettor):
    from blackjackai.edge_live import EdgeService
    service = EdgeService.__new__(EdgeService)
    service.bettor = bettor
    from blackjackai.shoe import pair_table, play_table
    service.fixed = play_table()
    service.pairs = pair_table(True)
    from blackjackai.sizing import SizingConfig
    service.sizing = SizingConfig(kelly_fraction=0.5, cap_units=4, min_edge=0.005)
    service.sizing_name = "test-sizing"
    service.player = None
    service.play_policy_name = "test"
    assert service.stake_for(0.001, 400.0) == 0
    assert service.stake_for(0.02, 400.0) >= 1
    assert service.stake_for(0.20, 400.0) == 4
    bettor.threshold = -float("inf")
    session = EdgeSession(bettor, 5150, service)
    result = session.advance(200)
    assert result["sized_profit"] != result["profit"] or result["staked_units"] == result["bets"]
    assert result["bankroll_units"] == 400.0 + result["sized_profit"]
    assert result["last_round"]["stake"] >= 0


def _sim_service(bettor, tmp_path):
    from blackjackai.edge_live import EdgeService
    from blackjackai.edge import BetNetwork
    torch.manual_seed(3)
    path = tmp_path / "sim_bet.pt"
    torch.save({"kind": "public_count_monte_carlo_bet_value", "format_version": 1,
                "model": BetNetwork().state_dict(), "threshold": 0.02,
                "rules": asdict(ShoeRules())}, path)
    return EdgeService(path)


def test_bankroll_sim_is_deterministic_and_accounted(bettor, tmp_path):
    service = _sim_service(bettor, tmp_path)
    first = service.bankroll_sim(10000, 25, 60, 99)
    second = service.bankroll_sim(10000, 25, 60, 99)
    assert first == second
    assert first["outcome"] in ("doubled", "ruined", "capped")
    assert first["rounds"] <= 60 * 60  # sane upper bound on rounds seen
    kelly, flat = first["kelly"], first["flat"]
    assert kelly["curve"][0] == [0, 10000] and flat["curve"][0] == [0, 10000]
    assert kelly["curve"][-1][0] == first["rounds"]
    assert kelly["max_stake_units"] >= 0 and flat["bets"] >= 0
    assert kelly["worst_fraction"] <= 1.0


def test_bankroll_sim_rejects_bad_inputs(bettor, tmp_path):
    service = _sim_service(bettor, tmp_path)
    with pytest.raises(ValueError):
        service.bankroll_sim(10, 25, 60, 99)
    with pytest.raises(ValueError):
        service.bankroll_sim("lots", 25, 60, 99)
    with pytest.raises(ValueError):
        service.bankroll_sim(10000, 7, 60, 99)
    with pytest.raises(ValueError):
        service.bankroll_sim(10000, 25, 0, 99)
    with pytest.raises(ValueError):
        service.bankroll_sim(10000, 25, 5001, 99)
    with pytest.raises(ValueError):
        service.bankroll_sim(10000, 25, 60, -1)
