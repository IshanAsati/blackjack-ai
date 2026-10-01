"""Freeze trained models into a static webapp bundle (no backend needed).

Exports neural-network WEIGHTS as JSON plus a tiny MLP forward spec, so the
browser runs the exact same nets (classic DQN, count-aware strategist, betting
value head) with plain JavaScript matrix math. Also exports pair tables,
sizing config, rules, evaluation summaries, and training curves for the UI.

Layout: <output>/data/*.json consumed by the static SPA in <output>.
"""

import csv
import json
from pathlib import Path

import torch


def mlp_json(network) -> dict:
    """Ordered Linear layers (weight rows-out x cols-in) + activations."""
    linears, activations = [], []
    for module in network.layers:
        if isinstance(module, torch.nn.Linear):
            linears.append({"weights": module.weight.detach().numpy().tolist(),
                            "bias": module.bias.detach().numpy().tolist()})
        elif isinstance(module, torch.nn.ReLU):
            activations.append("relu")
        elif isinstance(module, torch.nn.Tanh):
            activations.append("tanh")
    assert len(linears) == len(activations) + 1
    return {"linears": linears, "activations": activations,
            "input_dim": len(linears[0]["weights"][0]),
            "output_dim": len(linears[-1]["bias"])}


def export_webapp(checkpoint: Path, bettor: Path | None, player: Path | None,
                  sizing: Path | None, output: Path) -> dict:
    from .agent import load_policy
    from .shoe import ShoeRules, pair_table

    data = output / "data"
    if data.exists() and any(data.iterdir()):
        raise FileExistsError(f"{data} already contains a bundle; remove it first")
    data.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_by": "blackjackai export", "files": []}

    def write(name: str, payload) -> None:
        (data / name).write_text(json.dumps(payload), encoding="utf-8")
        manifest["files"].append(name)

    # Classic hit/stand DQN.
    policy = load_policy(checkpoint)
    write("classic_q.json", {"kind": "mlp", "network": mlp_json(policy.online),
                             "observation": "total 4-21 one-hot[0:32], dealer 1-10 one-hot[32:42], usable-ace flag[42]",
                             "actions": ["stand", "hit"]})

    # Count-aware strategist (hit/stand/double + count inputs).
    if player is None:
        candidate = Path("runs/count-distilled/model.pt")
        player = candidate if candidate.is_file() else None
    if player is not None:
        from .count_rl import load_count_policy
        strategist = load_count_policy(player)
        write("count_q.json", {"kind": "mlp", "network": mlp_json(strategist.network),
                               "observation": "classic 43 + clipped-true-count/10 + fraction-remaining + can-double flag",
                               "actions": ["stand", "hit", "double"]})
    manifest["count_strategy"] = player is not None

    # Betting value head + threshold + rules + sizing.
    if bettor is None:
        candidate = Path("runs/edge/bet_model.pt")
        bettor = candidate if candidate.is_file() else None
    if bettor is not None:
        from .edge import load_bettor
        network, threshold, rules = load_bettor(bettor)
        write("bet_value.json", {"kind": "mlp", "network": mlp_json(network),
                                 "observation": "[clipped-true-count/10, fraction-remaining]",
                                 "threshold": threshold, "rules": {"decks": rules.decks,
                                 "penetration": rules.penetration,
                                 "blackjack_payout": rules.blackjack_payout,
                                 "double_after_split": rules.double_after_split}})
    manifest["betting"] = bettor is not None
    write("pairs.json", {"splits": [[rank, up] for rank in range(1, 11) for up in range(1, 11)
                                    if pair_table(True)[(rank, up)] == 3],
                         "note": "basic-strategy splits (S17, double-any-two, DAS); else play the hand normally"})
    from .shoe import play_table as _fixed_table
    _fixed = _fixed_table()
    write("fixed_play.json", {
        "entries": [[total, up, int(ace), int(first), _fixed[total, up, ace, first]]
                    for total in range(4, 22) for up in range(1, 11)
                    for ace in (False, True) if (not ace or 12 <= total <= 21)
                    for first in (True, False)],
        "actions": ["stand", "hit", "double"],
        "note": "analytical infinite-deck hit/stand/double table; pairs handled by pairs.json"})

    sizing_cfg = None
    if sizing is None:
        candidate = Path("runs/sizing/sizing.json")
        sizing = candidate if candidate.is_file() else None
    if sizing is not None:
        report = json.loads(Path(sizing).read_text(encoding="utf-8"))
        sizing_cfg = report["selected_sizing"]["config"]
        write("sizing.json", {"selected": sizing_cfg,
                              "pooled_units_per_100": report.get("pooled_profit_units_per_100_rounds")})
    manifest["sizing"] = sizing_cfg is not None

    # Training curves + summaries for the Training tab (best-effort includes).
    training = {"runs": []}
    for name, directory in (("RL from scratch", "rl"), ("Pretraining + RL", "mixed")):
        root = Path("runs/comparison") / directory
        if (root / "training.csv").is_file():
            with (root / "training.csv").open(encoding="utf-8", newline="") as file:
                training["runs"].append({"name": name, "points": list(csv.DictReader(file))})
    for key, path in (("unsupervised", Path("runs/comparison/unsupervised/report.json")),
                      ("betting_validation", Path("runs/edge/validation.json")),
                      ("distilled", Path("runs/count-distilled/report.json"))):
        if path.is_file():
            training[key] = json.loads(path.read_text(encoding="utf-8"))
    write("training.json", training)
    manifest["optimal_return"] = -0.0465559770666975
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
