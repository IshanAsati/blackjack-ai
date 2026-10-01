import numpy as np
import torch

from blackjackai.count_rl import (
    CountHand,
    CountQNetwork,
    ShoeRunner,
    count_features,
    distill_fixed_strategy,
    encode_count_state,
    fixed_action,
    load_count_policy,
    play_shoes,
)
from blackjackai.shoe import ShoeRules, play_table


def test_count_state_encoding_is_46_dim_and_clips():
    vec = encode_count_state(16, 10, 0, 30.0, 0.7, True)
    assert vec.shape == (46,)
    assert vec[43] == 1.5  # clipped true count / 10
    assert vec[44] == 0.7
    assert vec[45] == 1.0
    assert count_features(-30, 1.0) == (-1.5, 1.0)


def test_natural_blackjack_pays_3_to_2_without_decisions():
    runner = ShoeRunner(ShoeRules(), 0, "test")
    seen = False
    for _ in range(200):
        hand = runner.deal()
        if hand.natural and sorted(hand.player) == [1, 10]:
            assert hand.done
            assert hand.reward == 1.5
            seen = True
            break
    assert seen


def test_double_only_legal_first_and_stakes_two():
    runner = ShoeRunner(ShoeRules(), 3, "test")
    for _ in range(200):
        hand = runner.deal()
        if not hand.done:
            total, _ = hand._total_ace()
            if 9 <= total <= 11:
                reward = hand.step(2)
                assert hand.done and hand.stake == 2.0
                assert reward in (-2.0, 0.0, 2.0)
                return
    raise AssertionError("no doubling hand appeared in 200 deals")


def test_double_after_hit_is_rejected():
    runner = ShoeRunner(ShoeRules(), 5, "test")
    for _ in range(200):
        hand = runner.deal()
        if hand.done:
            continue
        hand.step(1)
        if not hand.done:
            try:
                hand.step(2)
            except ValueError:
                return
            raise AssertionError("double after hit should be rejected")
    raise AssertionError("no multi-decision hand appeared in 200 deals")


def test_fixed_strategy_completes_shoes():
    rules = ShoeRules()
    profits, hands = play_shoes(fixed_action(play_table()), rules, 3, 7, "test-fixed")
    assert hands > 30
    assert all(isinstance(p, float) for p in profits)


def test_count_policy_roundtrip(tmp_path):
    network = CountQNetwork(16)
    path = tmp_path / "model.pt"
    torch.save({"format_version": 1, "kind": "count_aware_double_dqn",
                "metadata": {"hidden": 16}, "model": network.state_dict()}, path)
    policy = load_count_policy(path)
    action = policy.action_for(16, 10, 0, 0.0, 0.7, True)
    assert action in (0, 1, 2)
    assert len(policy.values_for(16, 10, 0, 0.0, 0.7, True)) == 3
    # Masking: without double rights the argmax never returns double.
    network.eval()
    with torch.no_grad():
        raw = network(torch.from_numpy(encode_count_state(11, 6, 0, 0.0, 0.7, False))).tolist()
    if int(np.argmax(raw)) == 2:
        assert policy.action_for(11, 6, 0, 0.0, 0.7, False) in (0, 1)


def test_distilled_net_clones_analytical_strategy(tmp_path):
    report = distill_fixed_strategy(32, 11, tmp_path, epochs=150)
    assert report["grid_agreement"] >= 0.99
    policy = load_count_policy(tmp_path / "model.pt")
    fixed = play_table()
    checked = mismatches = 0
    for ace in (0, 1):
        for total in range(12 if ace else 4, 22):
            for dealer in range(1, 11):
                for first in (True, False):
                    expected = fixed[total, dealer, bool(ace), first]
                    if not first:
                        expected = min(expected, 1)
                    got = policy.action_for(total, dealer, ace, 0.0, 0.7, first)
                    checked += 1
                    mismatches += got != expected
    assert mismatches / checked <= 0.01
