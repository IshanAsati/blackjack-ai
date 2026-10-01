import numpy as np
import pytest
import torch

from blackjackai.agent import Agent, Config, QNetwork, ReplayBuffer, bellman_targets, encode, load_policy, save_policy
from blackjackai.experiment import evaluate_policy, make_env, train
from blackjackai.game import Game, hand_value
from blackjackai.oracle import policy_report


def test_encoding_preserves_observation():
    state = encode((18, 1, 1))
    assert state.shape == (43,)
    assert np.flatnonzero(state).tolist() == [18, 32, 42]
    assert not np.array_equal(state, encode((18, 1, 0)))


def test_terminal_targets_do_not_bootstrap():
    result = bellman_targets(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([100.0, 0.5]), 0.9)
    torch.testing.assert_close(result, torch.tensor([-1.0, 0.45]))


def test_replay_discards_old_transitions():
    replay = ReplayBuffer(2, 1)
    for reward in range(3):
        replay.add(encode((12, 2, 0)), 1, reward, encode((22, 2, 0)), True)
    assert replay.size == 2
    assert sorted(replay.sample(2)[2].tolist()) == [1, 2]


def test_learning_changes_online_but_not_target():
    agent = Agent(Config(batch_size=4, warmup=4, capacity=8))
    for _ in range(4):
        agent.replay.add(encode((21, 2, 0)), 0, 1, encode((21, 2, 0)), True)
    before = [p.clone() for p in agent.online.parameters()]
    assert np.isfinite(agent.learn())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before, agent.target.parameters()))


def test_checkpoint_roundtrip(tmp_path):
    agent = Agent(Config())
    agent.save(tmp_path / "model.pt")
    loaded = Agent.load(tmp_path / "model.pt")
    state = torch.from_numpy(encode((16, 10, 0)))
    torch.testing.assert_close(agent.online(state), loaded.online(state))


def test_comparison_checkpoint_roundtrip(tmp_path):
    source = QNetwork(8)
    save_policy(tmp_path / "model.pt", source, "plain_q", {"hidden": 8})
    loaded = load_policy(tmp_path / "model.pt")
    state = torch.from_numpy(encode((16, 10, 0)))
    torch.testing.assert_close(source(state), loaded.online(state))


def test_exact_stand_on_17_reference_value():
    report = policy_report(lambda observation: int(observation[0] < 17))
    assert report["exact_mean_reward"] == pytest.approx(-0.079295, abs=0.000001)
    assert report["optimal_mean_reward"] == pytest.approx(-0.046556, abs=0.000001)


def test_environment_rules_and_seeded_evaluation():
    env = make_env()
    assert env.unwrapped.natural is False and env.unwrapped.sab is False
    env.close()
    policy = lambda observation: int(observation[0] < 17)
    result = evaluate_policy(policy, 100, 50)
    assert result == evaluate_policy(policy, 100, 50)
    assert result["wins"] + result["draws"] + result["losses"] == 100


def test_training_writes_reloadable_artifacts(tmp_path):
    config = Config(episodes=32, batch_size=4, warmup=4, capacity=32, target_every=8)
    result = train(config, tmp_path, 20)
    assert set(result) == {"dqn", "random", "stand_on_17"}
    assert Agent.load(tmp_path / "model.pt").action((20, 10, 0)) in (0, 1)
    assert all((tmp_path / name).exists() for name in ("training.csv", "policy.csv", "evaluation.json", "config.json"))
    with pytest.raises(FileExistsError):
        train(config, tmp_path, 20)


def test_hand_value_uses_ace_without_busting():
    assert hand_value([1, 7]) == (18, True)
    assert hand_value([1, 7, 8]) == (16, False)


def test_game_hides_hole_card_and_resolves_stand():
    game = Game(forced_cards=[10, 8, 9, 7, 10])
    assert game.observation == (18, 9, 0)
    assert game.public_state()["dealer"] == [9, None]
    game.stand()
    assert game.result == "win"
    assert game.public_state()["dealer"] == [9, 7, 10]


def test_game_resolves_player_bust():
    game = Game(forced_cards=[10, 6, 10, 7, 10])
    game.hit()
    assert game.result == "loss"
    with pytest.raises(ValueError):
        game.hit()
