"""Deterministic diagnostic: score a learned policy against exact blackjack values."""

from pathlib import Path

from blackjackai.agent import Agent
from blackjackai.oracle import policy_report


agent = Agent.load(Path("runs/dqn/model.pt"))
dqn = policy_report(agent.action)
stand = policy_report(lambda state: int(state[0] < 17))
print(f"exact optimal value: {dqn['optimal_mean_reward']:+.6f}")
print(f"exact dqn value:     {dqn['exact_mean_reward']:+.6f}")
print(f"exact stand-17:      {stand['exact_mean_reward']:+.6f}")
print(f"dqn-optimal gap:     {dqn['optimality_gap']:+.6f}")
print(f"action mismatches:   {dqn['action_mismatches']} / {dqn['reachable_states']}")
for state in dqn["largest_mismatches"]:
    print(f"  total={state['player_total']}, dealer={state['dealer_card']}, usable_ace={state['usable_ace']}: optimal {state['optimal_action']}, learned {state['policy_action']}, decision cost {state['decision_cost']:.3f}")

# The actual symptom check: the trained saved policy must beat the simple
# stand-on-17 policy under exact, non-sampled evaluation.
assert dqn["exact_mean_reward"] > stand["exact_mean_reward"], "DQN policy is worse than the stand-on-17 baseline"
