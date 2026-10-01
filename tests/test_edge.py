import numpy as np
import pytest
import torch
from dataclasses import asdict

from blackjackai.edge import BetNetwork, EdgeConfig, collect, load_bettor, predict, summarize, shoe_seed
from blackjackai.oracle import optimal_action_values, policy_report
from blackjackai.shoe import DOUBLE, HIT, STAND, Shoe, ShoeRules, count_tag, play_table


def fixture(cards):
    return Shoe(ShoeRules(), 0, cards=cards)


def test_natural_payout_and_dealer_peek_priority():
    assert fixture([1, 10, 10, 1]).play_round() == 0
    assert fixture([1, 9, 10, 7]).play_round() == 1.5
    assert fixture([5, 1, 6, 10, 10]).play_round() == -1


def test_double_risks_two_units_and_takes_one_card():
    assert play_table()[11, 6, False, True] == DOUBLE
    game = fixture([5, 6, 6, 10, 10, 10])
    assert game.play_round() == 2
    assert len(game.cards) == 0
    game = fixture([6, 10, 5, 10, 2])
    assert game.play_round() == -2


def test_soft_seventeen_stands_and_multicard_21_is_not_natural():
    game = fixture([10, 1, 8, 6, 4])
    assert game.play_round() == 1
    assert len(game.cards) == 1
    game = fixture([7, 6, 3, 10, 1, 4])
    # Initial 10 doubles, the added ace gives 21, ordinary double payout = 2.
    assert game.play_round() == 2


def test_hidden_order_does_not_change_predeal_observation():
    left = fixture([10, 2, 3, 4, 5, 6])
    right = fixture([6, 5, 4, 3, 2, 10])
    assert left.observation() == right.observation() == (0, 1)
    left.play_round()
    seen = [10, 2, 3, 4, 5, 6][:6 - len(left.cards)]
    assert left.exposed_count == sum(map(count_tag, seen))


def test_shoe_has_correct_counts_no_replacement_and_cut_card():
    shoe = Shoe(ShoeRules(), 21)
    assert len(shoe.cards) == 312 and shoe.cards.count(10) == 96
    assert sum(map(count_tag, shoe.cards)) == 0
    initial_rounds = 0
    while shoe.can_deal:
        prior = len(shoe.cards)
        shoe.play_round()
        assert len(shoe.cards) < prior
        initial_rounds += 1
    assert 24 <= len(shoe.cards) <= 78
    assert initial_rounds > 1


def test_sample_streams_separate_and_observations_reset_each_shoe():
    assert shoe_seed(1, "train", 0) != shoe_seed(1, "test-0", 0)
    x, y, ids = collect(ShoeRules(), 4, 1, "train")
    again = collect(ShoeRules(), 4, 1, "train")
    for actual, expected in zip((x, y, ids), again):
        np.testing.assert_array_equal(actual, expected)
    for i in range(4):
        np.testing.assert_array_equal(x[np.where(ids == i)[0][0]], [0, 1])


def test_zero_bets_cannot_be_called_positive_edge():
    result = summarize(np.array([1, -1, 1]), np.array([0, 0, 1]), np.zeros(3), 2)
    assert result["profit_units"] == 0
    assert result["return_per_initial_unit_bet_95ci"] is None


def test_cluster_interval_accounts_for_rounds_within_shoes():
    rewards = np.array([1, 1, -1, -1], dtype=float)
    report = summarize(rewards, np.array([0, 0, 1, 1]), np.ones(4), 2)
    assert report["profit_units"] == 0
    assert report["return_per_initial_unit_bet_95ci"] == pytest.approx([-1.96, 1.96])


def test_oracle_reports_true_optimal_action_even_with_bad_future_policy():
    # Always hit loses every hand, but hitting 4 is still the optimal *decision*.
    report = policy_report(lambda _: HIT)
    assert report["exact_mean_reward"] == pytest.approx(-1)
    assert report["action_mismatches"] < 280
    exact = policy_report(lambda state: int(optimal_action_values(state)[1] > optimal_action_values(state)[0]))
    assert exact["action_mismatches"] == 0
    assert exact["optimality_gap"] == pytest.approx(0)


def test_saved_betting_policy_preserves_decisions_and_rules(tmp_path):
    model = BetNetwork().eval()
    checkpoint = tmp_path / "bettor.pt"
    torch.save({"format_version": 1, "kind": "public_count_monte_carlo_bet_value",
                "threshold": 0.02, "rules": asdict(ShoeRules()), "model": model.state_dict()}, checkpoint)
    restored, threshold, rules = load_bettor(checkpoint)
    states = np.asarray([[0, 1], [4, 0.5], [-4, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(predict(restored, states), predict(model, states))
    assert threshold == 0.02 and rules == ShoeRules()
