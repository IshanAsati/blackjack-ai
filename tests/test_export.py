"""Export parity: the static bundle must reproduce the Python models exactly."""

import json

import numpy as np
import pytest
import torch

from blackjackai.agent import encode, load_policy
from blackjackai.count_rl import encode_count_state, load_count_policy
from blackjackai.edge import load_bettor
from blackjackai.export import export_webapp, mlp_json


def numpy_forward(spec, x):
    acts = {"relu": lambda v: np.maximum(v, 0), "tanh": np.tanh}
    for i, layer in enumerate(spec["linears"]):
        x = np.asarray(layer["weights"], dtype=np.float64) @ np.asarray(x, dtype=np.float64)
        x = x + np.asarray(layer["bias"], dtype=np.float64)
        if i < len(spec["activations"]):
            x = acts[spec["activations"][i]](x)
    return x


def test_mlp_json_matches_torch_on_all_three_nets():
    rng = np.random.default_rng(0)
    policy = load_policy("runs/comparison/rl/model.pt")
    spec = mlp_json(policy.online)
    for _ in range(5):
        x = rng.normal(size=43).astype(np.float32)
        expected = policy.online(torch.from_numpy(x)).detach().numpy()
        assert np.allclose(numpy_forward(spec, x), expected, atol=1e-5)
    strategist = load_count_policy("runs/count-distilled/model.pt")
    spec = mlp_json(strategist.network)
    for _ in range(5):
        x = rng.normal(size=46).astype(np.float32)
        expected = strategist.network(torch.from_numpy(x)).detach().numpy()
        assert np.allclose(numpy_forward(spec, x), expected, atol=1e-5)
    network, _, _ = load_bettor("runs/edge/bet_model.pt")
    spec = mlp_json(network)
    for _ in range(5):
        x = rng.normal(size=2).astype(np.float32)
        expected = network(torch.from_numpy(x)).detach().numpy()
        assert np.allclose(numpy_forward(spec, x), expected, atol=1e-5)


def test_exported_classic_policy_matches_checkpoint(tmp_path):
    export_webapp("runs/comparison/rl/model.pt", None, None, None, tmp_path / "data")
    bundle = json.loads((tmp_path / "data" / "data" / "classic_q.json").read_text(encoding="utf-8"))
    policy = load_policy("runs/comparison/rl/model.pt")
    mismatches = checked = 0
    for total in range(4, 22):
        for dealer in range(1, 11):
            for ace in (0, 1):
                x = encode((total, dealer, ace)).astype(np.float64)
                action = int(np.argmax(numpy_forward(bundle["network"], x)))
                with torch.no_grad():
                    expected = int(policy.online(torch.from_numpy(encode((total, dealer, ace)))).argmax().item())
                checked += 1
                mismatches += action != expected
    assert checked == 360 and mismatches == 0


def test_bundle_layout_and_tables(tmp_path):
    manifest = export_webapp("runs/comparison/rl/model.pt", None, None, None, tmp_path / "data")
    assert manifest["betting"] and manifest["sizing"] and manifest["count_strategy"]
    data = tmp_path / "data" / "data"
    fixed = json.loads((data / "fixed_play.json").read_text(encoding="utf-8"))
    assert len(fixed["entries"]) == 560
    pairs = json.loads((data / "pairs.json").read_text(encoding="utf-8"))
    assert [1, 6] in pairs["splits"] and [10, 6] not in pairs["splits"]
    training = json.loads((data / "training.json").read_text(encoding="utf-8"))
    assert len(training["runs"]) == 2 and training["distilled"]["grid_agreement"] == 1.0
    with pytest.raises(FileExistsError):
        export_webapp("runs/comparison/rl/model.pt", None, None, None, tmp_path / "data")
