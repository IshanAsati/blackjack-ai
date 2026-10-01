import pytest

from blackjackai.table import MultiTable


def test_deal_hides_hole_card_and_sets_turn_order():
    table = MultiTable(num_seats=2, forced_cards=[10, 6, 10, 8, 9, 7])
    state = table.public_state()
    assert state["dealer"] == [9, None]
    assert state["current_seat"] == 0
    assert table.observation(0) == (16, 9, 0)
    assert table.observation(1) == (18, 9, 0)


def test_turn_order_is_enforced():
    table = MultiTable(num_seats=2, forced_cards=[10, 6, 10, 8, 9, 7])
    with pytest.raises(ValueError):
        table.hit(1)
    with pytest.raises(ValueError):
        table.stand(5)


def test_bust_and_stand_resolve_against_shared_dealer():
    table = MultiTable(num_seats=2, forced_cards=[10, 6, 10, 8, 9, 7, 10, 10])
    table.hit(0)
    assert table.seat_results[0] == "loss"
    assert table.current_seat == 1
    table.stand(1)
    assert table.stage == "finished"
    # Dealer: 9 + 7 + 10 = 26 bust, so the standing 18 wins.
    assert table.public_state()["dealer"] == [9, 7, 10]
    assert table.seat_results == ["loss", "win"]


def test_all_bust_skips_dealer_draws():
    table = MultiTable(num_seats=1, forced_cards=[10, 6, 9, 7, 10])
    table.hit(0)
    assert table.stage == "finished"
    assert table.public_state()["dealer"] == [9, 7]
    assert table.seat_results == ["loss"]


def test_seat_count_is_validated():
    with pytest.raises(ValueError):
        MultiTable(num_seats=0)
    with pytest.raises(ValueError):
        MultiTable(num_seats=6)
    with pytest.raises(ValueError):
        MultiTable(num_seats=2, forced_cards=[11, 2, 3, 4, 5, 6])
