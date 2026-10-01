"""DQN with experience replay and a periodically synchronized target network."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

Observation = tuple[int, int, int]


def encode(observation: Observation) -> np.ndarray:
    """One-hot total and dealer card, plus usable ace; never expose the hole card."""
    total, dealer, ace = observation
    result = np.zeros(43, dtype=np.float32)
    result[total] = 1.0
    result[32 + dealer - 1] = 1.0
    result[42] = float(ace)
    return result


@dataclass
class Config:
    episodes: int = 200_000
    seed: int = 42
    hidden: int = 64
    learning_rate: float = 0.0003
    gamma: float = 1.0
    batch_size: int = 128
    capacity: int = 50_000
    warmup: int = 2_000
    train_every: int = 4
    target_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_fraction: float = 0.8

    def __post_init__(self):
        for name in ("episodes", "hidden", "batch_size", "capacity", "warmup", "train_every", "target_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.capacity < max(self.batch_size, self.warmup):
            raise ValueError("capacity must cover batch_size and warmup")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if not 0 < self.decay_fraction <= 1 or not 0 <= self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid decay_fraction, gamma, or learning_rate")

    def epsilon(self, episode: int) -> float:
        fraction = min(episode / max(1, self.episodes * self.decay_fraction), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class QNetwork(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(43, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2),
        )

    def forward(self, states):
        return self.layers(states)


class StateEncoder(nn.Module):
    """A compact state representation used by the unsupervised and mixed runs."""
    def __init__(self, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(43, hidden), nn.ReLU(), nn.Linear(hidden, latent))

    def forward(self, states):
        return self.layers(states)


class StateAutoencoder(nn.Module):
    def __init__(self, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.hidden, self.latent = hidden, latent
        self.encoder = StateEncoder(hidden, latent)
        self.decoder = nn.Sequential(nn.ReLU(), nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 43))

    def forward(self, states):
        return self.decoder(self.encoder(states))


class EncodedQNetwork(nn.Module):
    """Q-network initialized from an unlabeled blackjack-state representation."""
    def __init__(self, encoder: StateEncoder, hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        self.hidden = hidden
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(encoder.layers[-1].out_features, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, states):
        return self.head(self.encoder(states))


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, 43), dtype=np.float32)
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
            self.states, self.actions, self.rewards, self.next_states, self.terminated
        ))


def bellman_targets(rewards, terminated, next_values, gamma):
    # A completed hand has no future reward. Time-limit truncation is separate.
    return rewards + gamma * (1.0 - terminated) * next_values


class Agent:
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(config.seed)
        self.rng = random.Random(config.seed)
        self.online = QNetwork(config.hidden)
        self.target = QNetwork(config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.capacity, config.seed)

    @torch.no_grad()
    def values(self, observation):
        return self.online(torch.from_numpy(encode(observation)))

    @torch.no_grad()
    def action(self, observation, epsilon=0.0):
        if epsilon > 0 and self.rng.random() < epsilon:
            return self.rng.randrange(2)
        return int(self.values(observation).argmax().item())

    def learn(self):
        states, actions, rewards, next_states, terminated = self.replay.sample(self.config.batch_size)
        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            future = self.target(next_states).max(dim=1).values
            expected = bellman_targets(rewards, terminated, future, self.config.gamma)
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": asdict(self.config),
                    "model": self.online.state_dict()}, path)

    @classmethod
    def load(cls, path: Path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        agent = cls(Config(**data["config"]))
        agent.online.load_state_dict(data["model"])
        agent.target.load_state_dict(data["model"])
        agent.online.eval()
        return agent


class FrozenPolicy:
    """Inference-only policy for checkpoints produced by comparison experiments."""
    def __init__(self, network: nn.Module):
        self.online = network.eval()

    @torch.no_grad()
    def values(self, observation):
        return self.online(torch.from_numpy(encode(observation)))

    @torch.no_grad()
    def action(self, observation, epsilon=0.0):
        if epsilon:
            raise ValueError("frozen policies do not support exploration")
        return int(self.values(observation).argmax().item())


def save_policy(path: Path, network: nn.Module, model_type: str, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "model_type": model_type, "metadata": metadata,
                "model": network.state_dict()}, path)


def load_policy(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") == 1:
        return Agent.load(path)
    if data.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format")
    metadata = data["metadata"]
    if data["model_type"] == "plain_q":
        network = QNetwork(metadata["hidden"])
    elif data["model_type"] == "encoded_q":
        network = EncodedQNetwork(StateEncoder(metadata["encoder_hidden"], metadata["latent"]), metadata["hidden"])
    else:
        raise ValueError("Unsupported comparison model type")
    network.load_state_dict(data["model"])
    return FrozenPolicy(network)
