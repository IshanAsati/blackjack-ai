"""Train pure RL, unsupervised representation, and mixed blackjack experiments."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .agent import EncodedQNetwork, QNetwork, ReplayBuffer, StateAutoencoder, encode, save_policy
from .experiment import evaluate_policy, make_env
from .oracle import policy_report


@dataclass
class ComparisonConfig:
    episodes: int = 200_000
    seed: int = 31415
    hidden: int = 64
    latent: int = 16
    unlabeled_states: int = 100_000
    pretrain_epochs: int = 40
    batch_size: int = 128
    capacity: int = 75_000
    warmup: int = 4_000
    learning_rate: float = 0.0003
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    decay_fraction: float = 0.75
    target_every: int = 2_000
    train_every: int = 4
    check_every: int = 25_000
    eval_episodes: int = 50_000

    def epsilon(self, episode):
        fraction = min(episode / max(1, self.episodes * self.decay_fraction), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class DoubleDQN:
    def __init__(self, network: nn.Module, config: ComparisonConfig, seed: int):
        self.network = network
        self.target = deepcopy(network).requires_grad_(False)
        self.config = config
        self.rng = random.Random(seed)
        self.replay = ReplayBuffer(config.capacity, seed)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def action(self, observation, epsilon=0.0):
        if epsilon and self.rng.random() < epsilon:
            return self.rng.randrange(2)
        return int(self.network(torch.from_numpy(encode(observation))).argmax().item())

    def learn(self):
        states, actions, rewards, next_states, terminated = self.replay.sample(self.config.batch_size)
        predicted = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network chooses; target network evaluates.
            next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
            future = self.target(next_states).gather(1, next_actions).squeeze(1)
            expected = rewards + self.config.gamma * (1.0 - terminated) * future
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())


def collect_unlabeled_states(count: int, seed: int) -> np.ndarray:
    """Collect observations only; actions and rewards are intentionally discarded."""
    env = make_env()
    rng = random.Random(seed)
    states = []
    try:
        observation, _ = env.reset(seed=seed)
        while len(states) < count:
            states.append(encode(observation))
            observation, _, terminated, truncated, _ = env.step(rng.randrange(2))
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    return np.asarray(states, dtype=np.float32)


def train_autoencoder(states: np.ndarray, config: ComparisonConfig, output: Path):
    torch.manual_seed(config.seed)
    model = StateAutoencoder(config.hidden, config.latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    samples = torch.from_numpy(states)
    generator = torch.Generator().manual_seed(config.seed)
    positive_weight = torch.full((43,), 10.0)
    losses = []
    for _ in range(config.pretrain_epochs):
        for indices in torch.randperm(len(samples), generator=generator).split(512):
            batch = samples[indices]
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch), batch, pos_weight=positive_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    with torch.no_grad():
        reconstructed = model(samples)
        player_accuracy = (reconstructed[:, :32].argmax(dim=1) == samples[:, :32].argmax(dim=1)).float().mean().item()
        dealer_accuracy = (reconstructed[:, 32:42].argmax(dim=1) == samples[:, 32:42].argmax(dim=1)).float().mean().item()
        ace_accuracy = ((reconstructed[:, 42] > 0) == (samples[:, 42] > 0.5)).float().mean().item()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "kind": "unsupervised_state_autoencoder",
                "metadata": {"hidden": config.hidden, "latent": config.latent, "unlabeled_states": len(states),
                             "epochs": config.pretrain_epochs}, "model": model.state_dict()}, output / "representation.pt")
    report = {"kind": "unsupervised_state_representation", "unlabeled_states": len(states),
              "epochs": config.pretrain_epochs, "final_loss": losses[-1],
              "player_total_reconstruction_accuracy": player_accuracy,
              "dealer_card_reconstruction_accuracy": dealer_accuracy,
              "usable_ace_reconstruction_accuracy": ace_accuracy,
              "policy_evaluation": None,
              "note": "This run receives no rewards or action labels, so it cannot produce a hit/stand policy by itself."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, report


def train_rl(name: str, network: nn.Module, model_type: str, config: ComparisonConfig, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    trainer = DoubleDQN(network, config, config.seed)
    env = make_env()
    steps = 0
    best = {"exact_mean_reward": float("-inf")}
    best_state = None
    interval_reward = interval_count = 0
    try:
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "steps", "epsilon", "exploratory_mean_reward", "loss", "exact_mean_reward", "action_mismatches"])
            writer.writeheader()
            observation, _ = env.reset(seed=config.seed)
            loss = None
            for episode in range(config.episodes):
                epsilon = config.epsilon(episode)
                done = False
                while not done:
                    action = trainer.action(observation, epsilon)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    trainer.replay.add(encode(observation), action, reward, encode(next_observation), terminated)
                    steps += 1
                    if trainer.replay.size >= max(config.warmup, config.batch_size) and steps % config.train_every == 0:
                        loss = trainer.learn()
                    if steps % config.target_every == 0:
                        trainer.target.load_state_dict(trainer.network.state_dict())
                    observation = next_observation
                    interval_reward += reward
                    done = terminated or truncated
                interval_count += 1
                if (episode + 1) % config.check_every == 0 or episode + 1 == config.episodes:
                    report = policy_report(trainer.action)
                    row = {"episode": episode + 1, "steps": steps, "epsilon": epsilon,
                           "exploratory_mean_reward": interval_reward / interval_count, "loss": loss,
                           "exact_mean_reward": report["exact_mean_reward"], "action_mismatches": report["action_mismatches"]}
                    writer.writerow(row)
                    file.flush()
                    print(f"{name}: episode {episode + 1:,}/{config.episodes:,} | exact {report['exact_mean_reward']:+.5f} | mismatches {report['action_mismatches']}", flush=True)
                    if report["exact_mean_reward"] > best["exact_mean_reward"]:
                        best, best_state = report, {key: value.detach().clone() for key, value in trainer.network.state_dict().items()}
                    interval_reward = interval_count = 0
                observation, _ = env.reset()
    finally:
        env.close()
    trainer.network.load_state_dict(best_state)
    metadata = {"hidden": config.hidden, "seed": config.seed, "best_exact_mean_reward": best["exact_mean_reward"]}
    if model_type == "encoded_q":
        metadata.update({"encoder_hidden": config.hidden, "latent": config.latent})
    save_policy(output / "model.pt", trainer.network, model_type, metadata)
    monte_carlo = evaluate_policy(trainer.action, config.eval_episodes, config.seed + 1_000_000)
    final = {"kind": name, "best_checkpoint": best, "monte_carlo": monte_carlo, "model": "model.pt"}
    (output / "report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def run_comparison(config: ComparisonConfig, output: Path):
    if config.episodes <= 0 or config.unlabeled_states <= 0 or config.pretrain_epochs <= 0 or config.check_every <= 0:
        raise ValueError("episode, state, epoch, and checkpoint counts must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains artifacts; choose a new --output directory")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    states = collect_unlabeled_states(config.unlabeled_states, config.seed)
    representation, unsupervised = train_autoencoder(states, config, output / "unsupervised")
    torch.manual_seed(config.seed)
    rl = train_rl("reinforcement_learning", QNetwork(config.hidden), "plain_q", config, output / "rl")
    torch.manual_seed(config.seed)
    mixed_network = EncodedQNetwork(deepcopy(representation.encoder), config.hidden)
    mixed = train_rl("mixed_unsupervised_pretraining_plus_rl", mixed_network, "encoded_q", config, output / "mixed")
    candidates = {"rl": rl, "mixed": mixed}
    best_name = max(candidates, key=lambda name: candidates[name]["best_checkpoint"]["exact_mean_reward"])
    summary = {"comparison": {"reinforcement_learning": rl, "unsupervised": unsupervised, "mixed": mixed},
               "best_playable_model": best_name, "best_model_path": f"{best_name}/model.pt"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
