"""Exact policy evaluation for the infinite-deck blackjack rules used here."""

from functools import cache

CARDS = tuple(range(1, 11))
CARD_PROBABILITY = {card: (4 / 13 if card == 10 else 1 / 13) for card in CARDS}
# A hit always increases hard total, so this order supports dynamic programming.
PLAYER_STATES = tuple(sorted(
    [(total, False) for total in range(4, 22)] + [(total, True) for total in range(12, 22)],
    key=lambda state: state[0] - 10 * state[1], reverse=True,
))


def add_card(total: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    if card == 1 and not usable_ace and total + 11 <= 21:
        return total + 11, True
    total += 1 if card == 1 else card
    if total > 21 and usable_ace:
        return total - 10, False
    return total, usable_ace


@cache
def dealer_distribution(total: int, usable_ace: bool) -> dict[int, float]:
    """Distribution of dealer totals; 22 represents a dealer bust."""
    if total >= 17:
        return {total: 1.0}
    result: dict[int, float] = {}
    for card, probability in CARD_PROBABILITY.items():
        next_total, next_ace = add_card(total, usable_ace, card)
        if next_total > 21:
            result[22] = result.get(22, 0.0) + probability
        else:
            for final_total, final_probability in dealer_distribution(next_total, next_ace).items():
                result[final_total] = result.get(final_total, 0.0) + probability * final_probability
    return result


@cache
def stand_value(player_total: int, dealer_upcard: int) -> float:
    value = 0.0
    for hole_card, probability in CARD_PROBABILITY.items():
        dealer_total, dealer_ace = add_card(0, False, dealer_upcard)
        dealer_total, dealer_ace = add_card(dealer_total, dealer_ace, hole_card)
        for final_total, final_probability in dealer_distribution(dealer_total, dealer_ace).items():
            reward = 1 if final_total > 21 or player_total > final_total else -1 if player_total < final_total else 0
            value += probability * final_probability * reward
    return value


@cache
def optimal_values() -> dict[tuple[int, int, bool], float]:
    values = {}
    for total, ace in PLAYER_STATES:
        for dealer in range(1, 11):
            hit = sum(
                probability * (-1 if (next_state := add_card(total, ace, card))[0] > 21 else values[(next_state[0], dealer, next_state[1])])
                for card, probability in CARD_PROBABILITY.items()
            )
            values[(total, dealer, ace)] = max(stand_value(total, dealer), hit)
    return values


def state_values(action_for) -> tuple[dict[tuple[int, int, bool], float], list[dict]]:
    """Evaluate any deterministic hit/stand policy exactly, including its mistakes."""
    values = {}
    mismatches = []
    for total, ace in PLAYER_STATES:
        for dealer in range(1, 11):
            hit = sum(
                probability * (-1 if (next_state := add_card(total, ace, card))[0] > 21 else values[(next_state[0], dealer, next_state[1])])
                for card, probability in CARD_PROBABILITY.items()
            )
            stand = stand_value(total, dealer)
            action = action_for((total, dealer, int(ace)))
            if action not in (0, 1):
                raise ValueError("policy must return 0 (stand) or 1 (hit)")
            values[(total, dealer, ace)] = (stand, hit)[action]
            # Compare with optimal continuation, not continuation under the
            # evaluated policy (which can itself contain later mistakes).
            optimal_stand, optimal_hit = optimal_action_values((total, dealer, int(ace)))
            optimal_action = int(optimal_hit > optimal_stand)
            if action != optimal_action:
                mismatches.append({
                    "player_total": total, "dealer_card": dealer, "usable_ace": int(ace),
                    "optimal_action": "hit" if optimal_action else "stand",
                    "policy_action": "hit" if action else "stand", "decision_cost": abs(optimal_hit - optimal_stand),
                })
    return values, mismatches


def optimal_action_values(observation) -> tuple[float, float]:
    total, dealer, ace = observation
    optimal = optimal_values()
    hit = 0.0
    for card, probability in CARD_PROBABILITY.items():
        next_total, next_ace = add_card(total, bool(ace), card)
        hit += probability * (-1 if next_total > 21 else optimal[(next_total, dealer, next_ace)])
    return stand_value(total, dealer), hit


def initial_value(values: dict[tuple[int, int, bool], float]) -> float:
    value = 0.0
    for first in CARDS:
        for second in CARDS:
            total, ace = add_card(0, False, first)
            total, ace = add_card(total, ace, second)
            for dealer in CARDS:
                value += CARD_PROBABILITY[first] * CARD_PROBABILITY[second] * CARD_PROBABILITY[dealer] * values[(total, dealer, ace)]
    return value


def policy_report(action_for) -> dict:
    values, mismatches = state_values(action_for)
    optimal = initial_value(optimal_values())
    score = initial_value(values)
    return {
        "exact_mean_reward": score,
        "optimal_mean_reward": optimal,
        "optimality_gap": optimal - score,
        "action_mismatches": len(mismatches),
        "reachable_states": len(PLAYER_STATES) * 10,
        "largest_mismatches": sorted(mismatches, key=lambda item: item["decision_cost"], reverse=True)[:10],
    }
