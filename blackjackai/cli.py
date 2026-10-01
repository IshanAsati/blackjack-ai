import argparse
import json
from pathlib import Path

import torch

from .agent import Agent, Config, load_policy
from .experiment import compare, train


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a blackjack DQN")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--episodes", type=positive, default=200_000)
    training.add_argument("--eval-episodes", type=positive, default=50_000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", type=Path, default=Path("runs/dqn"))
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--episodes", type=positive, default=100_000)
    evaluation.add_argument("--seed", type=int, default=2_000_042)
    playing = commands.add_parser("play", help="open a local table with DQN advice")
    playing.add_argument("checkpoint", type=Path, nargs="?", default=Path("runs/dqn/model.pt"))
    playing.add_argument("--host", default="127.0.0.1")
    playing.add_argument("--port", type=positive, default=8000)
    playing.add_argument("--bettor", type=Path, help="finite-shoe betting checkpoint (auto-detected when available)")
    comparison = commands.add_parser("compare", help="run RL, unsupervised, and mixed training experiments")
    comparison.add_argument("--episodes", type=positive, default=200_000)
    comparison.add_argument("--unlabeled-states", type=positive, default=100_000)
    comparison.add_argument("--pretrain-epochs", type=positive, default=40)
    comparison.add_argument("--eval-episodes", type=positive, default=50_000)
    comparison.add_argument("--seed", type=int, default=31415)
    comparison.add_argument("--output", type=Path, default=Path("runs/comparison"))
    edge = commands.add_parser("edge", help="test a public-count neural betting agent on finite shoes")
    edge.add_argument("--train-shoes", type=positive, default=20_000)
    edge.add_argument("--validation-shoes", type=positive, default=10_000)
    edge.add_argument("--test-shoes", type=positive, default=40_000)
    edge.add_argument("--test-batches", type=positive, default=3)
    edge.add_argument("--seed", type=int, default=71623)
    edge.add_argument("--output", type=Path, default=Path("runs/edge"))
    counting = commands.add_parser("count-rl", help="train a count-aware neural play strategy on finite shoes")
    counting.add_argument("--hands", type=positive, default=200_000)
    counting.add_argument("--seed", type=int, default=917)
    counting.add_argument("--val-shoes", type=positive, default=300)
    counting.add_argument("--warm-start", type=Path, default=None,
                          help="distilled checkpoint to fine-tune with RL")
    counting.add_argument("--output", type=Path, default=Path("runs/count"))
    distilling = commands.add_parser("count-distill", help="clone the analytical strategy into the count-aware net (supervised)")
    distilling.add_argument("--seed", type=int, default=917)
    distilling.add_argument("--output", type=Path, default=Path("runs/count-distilled"))
    exporting = commands.add_parser("export", help="freeze models into a static webapp bundle")
    exporting.add_argument("--checkpoint", type=Path, default=Path("runs/dqn/model.pt"))
    exporting.add_argument("--bettor", type=Path, default=None)
    exporting.add_argument("--player", type=Path, default=None)
    exporting.add_argument("--sizing", type=Path, default=None)
    exporting.add_argument("--output", type=Path, default=Path("webapp"))
    sizing = commands.add_parser("size", help="Kelly-size the frozen betting value head and test on fresh shoes")
    sizing.add_argument("--bettor", type=Path, default=Path("runs/edge/bet_model.pt"))
    sizing.add_argument("--penetration", type=float, default=0.75)
    sizing.add_argument("--validation-shoes", type=positive, default=10_000)
    sizing.add_argument("--test-shoes", type=positive, default=40_000)
    sizing.add_argument("--test-batches", type=positive, default=3)
    sizing.add_argument("--seed", type=int, default=71623)
    sizing.add_argument("--rounds-per-hour", type=positive, default=100)
    sizing.add_argument("--output", type=Path, default=Path("runs/sizing"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        result = train(Config(episodes=args.episodes, seed=args.seed), args.output, args.eval_episodes)
    elif args.command == "evaluate":
        result = compare(load_policy(args.checkpoint), args.episodes, args.seed)
    elif args.command == "compare":
        from .comparison import ComparisonConfig, run_comparison
        result = run_comparison(ComparisonConfig(episodes=args.episodes, unlabeled_states=args.unlabeled_states,
                                pretrain_epochs=args.pretrain_epochs, eval_episodes=args.eval_episodes, seed=args.seed), args.output)
    elif args.command == "edge":
        from .edge import EdgeConfig, run_edge
        from .shoe import ShoeRules
        result = run_edge(EdgeConfig(train_shoes=args.train_shoes, validation_shoes=args.validation_shoes,
                         test_shoes_per_batch=args.test_shoes, test_batches=args.test_batches, seed=args.seed),
                         ShoeRules(), args.output)
    elif args.command == "count-rl":
        from .count_rl import CountConfig, train_count_rl
        from .shoe import ShoeRules
        result = train_count_rl(CountConfig(hands=args.hands, seed=args.seed, val_shoes=args.val_shoes,
                                            warm_start=str(args.warm_start) if args.warm_start else None),
                                ShoeRules(), args.output)
    elif args.command == "count-distill":
        from .count_rl import distill_fixed_strategy
        result = distill_fixed_strategy(128, args.seed, args.output)
    elif args.command == "export":
        from .export import export_webapp
        result = export_webapp(args.checkpoint, args.bettor, args.player, args.sizing, args.output)
    elif args.command == "size":
        from .sizing_run import run_sizing_experiment
        from .shoe import ShoeRules
        result = run_sizing_experiment(args.bettor, ShoeRules(penetration=args.penetration),
                                       args.validation_shoes, args.test_shoes, args.test_batches,
                                       args.seed, args.rounds_per_hour, args.output)
    else:
        from .web import serve
        serve(args.checkpoint, args.host, args.port, args.bettor)
        return
    print(json.dumps(result, indent=2))
