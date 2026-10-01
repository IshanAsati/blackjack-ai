"""Training and independent, exploration-free evaluation."""

from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random

import gymnasium as gym
import torch

from .agent import Agent, Config, encode


def make_env():
    # Even-money wins, including naturals; dealer stands on soft 17.
    return gym.make("Blackjack-v1", natural=False, sab=False)


def evaluate_policy(policy, episodes: int, seed: int):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = make_env()
    counts = {"wins": 0, "draws": 0, "losses": 0}
    try:
        for episode in range(episodes):
            # Each policy receives the same starting-hand seeds. Its subsequent
            # draws can differ because actions consume different numbers of cards.
            observation, _ = env.reset(seed=seed + episode)
            done = False
            while not done:
                observation, reward, terminated, truncated, _ = env.step(policy(observation))
                done = terminated or truncated
            counts["wins" if reward > 0 else "losses" if reward < 0 else "draws"] += 1
    finally:
        env.close()
    mean = (counts["wins"] - counts["losses"]) / episodes
    second_moment = (counts["wins"] + counts["losses"]) / episodes
    stderr = math.sqrt(max(0, second_moment - mean * mean) / (episodes - 1)) if episodes > 1 else None
    return {"episodes": episodes, "seed": seed, **counts, "mean_reward": mean,
            "win_rate": counts["wins"] / episodes,
            "mean_reward_95ci": [max(-1, mean - 1.96 * stderr), min(1, mean + 1.96 * stderr)] if stderr is not None else None}


def compare(agent: Agent, episodes: int, seed: int):
    rng = random.Random(seed)
    # Freeze the greedy policy into a lookup to make large evaluation runs fast.
    # These actions are network predictions, not a separately learned Q-table.
    actions = {(total, dealer, ace): agent.action((total, dealer, ace))
               for total in range(4, 22) for dealer in range(1, 11) for ace in (0, 1)}
    policies = {"dqn": lambda observation: actions[observation],
                "random": lambda _: rng.randrange(2),
                "stand_on_17": lambda observation: int(observation[0] < 17)}
    return {name: evaluate_policy(policy, episodes, seed) for name, policy in policies.items()}


def export_policy(agent: Agent, path: Path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["player_total", "dealer_card", "usable_ace", "action", "q_stand", "q_hit"])
        with torch.no_grad():
            for ace in (0, 1):
                for total in range(12 if ace else 4, 22):
                    for dealer in range(1, 11):
                        values = agent.online(torch.from_numpy(encode((total, dealer, ace)))).tolist()
                        writer.writerow([total, dealer, ace, "hit" if values[1] > values[0] else "stand", *values])


def train(config: Config, output: Path, eval_episodes: int):
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    if checkpoint.exists():
        raise FileExistsError(f"{checkpoint} already exists; choose a new --output directory")
    torch.set_num_threads(1)
    agent = Agent(config)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    env = make_env()
    steps = 0
    interval = min(10_000, config.episodes)
    reward_sum = count = 0
    last_loss = None
    try:
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "steps", "epsilon", "exploratory_mean_reward", "loss"])
            observation, _ = env.reset(seed=config.seed)
            for episode in range(config.episodes):
                epsilon = config.epsilon(episode)
                done = False
                while not done:
                    action = agent.action(observation, epsilon)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    agent.replay.add(encode(observation), action, reward, encode(next_observation), terminated)
                    steps += 1
                    if agent.replay.size >= max(config.warmup, config.batch_size) and steps % config.train_every == 0:
                        last_loss = agent.learn()
                    if steps % config.target_every == 0:
                        agent.target.load_state_dict(agent.online.state_dict())
                    observation = next_observation
                    reward_sum += reward
                    done = terminated or truncated
                count += 1
                if (episode + 1) % interval == 0 or episode + 1 == config.episodes:
                    writer.writerow([episode + 1, steps, epsilon, reward_sum / count, last_loss])
                    file.flush()
                    print(f"Episode {episode + 1:,}/{config.episodes:,} | epsilon {epsilon:.3f} | training reward {reward_sum / count:+.4f}", flush=True)
                    reward_sum = count = 0
                observation, _ = env.reset()
    finally:
        env.close()
    agent.save(checkpoint)
    export_policy(agent, output / "policy.csv")
    results = compare(agent, eval_episodes, config.seed + 1_000_000)
    (output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
