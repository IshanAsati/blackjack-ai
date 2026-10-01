import pytest

from blackjackai.agent import Agent, Config
from blackjackai.web import GameService


@pytest.fixture
def service(tmp_path):
    checkpoint = tmp_path / "model.pt"
    Agent(Config()).save(checkpoint)
    return GameService(checkpoint), Agent.load(checkpoint)


def test_advice_matches_greedy_policy(service):
    table, agent = service
    result = table.advise([10, 6], 9)
    assert result["player_total"] == 16 and result["usable_ace"] is False
    expected = "hit" if agent.action((16, 9, 0)) == 1 else "stand"
    assert result["advice"]["action"] == expected
    assert result["advice"]["margin"] >= 0


def test_advice_handles_soft_aces_and_blackjack(service):
    table, _ = service
    soft = table.advise([1, 5], 6)
    assert soft["player_total"] == 16 and soft["usable_ace"] is True
    natural = table.advise([1, 10], 9)
    assert natural["status"] == "blackjack"
    assert natural["advice"]["action"] == "stand"


def test_advice_reports_bust_without_action(service):
    table, _ = service
    result = table.advise([10, 10, 5], 9)
    assert result["status"] == "bust" and result["advice"] is None


def test_advice_flags_pairs_as_no_split(service):
    table, _ = service
    result = table.advise([8, 8], 6)
    assert "split" in (result["note"] or "")


@pytest.mark.parametrize("player,dealer", [([10], 9), ([10, "x"], 9), ([10, 11], 9),
                                            ([10, 6], 0), ([10, 6], 11), ([True, 6], 9)])
def test_advice_rejects_bad_cards(service, player, dealer):
    table, _ = service
    with pytest.raises(ValueError):
        table.advise(player, dealer)
