"""Finite-shoe research environment; separate from Gymnasium and the live UI.

Six decks, 75% penetration, S17, 3:2 naturals, US hole-card peek, double
on any first two cards, no splits/surrender/insurance. All cards are exposed
at round end, including dealer hole cards after a player bust. Betting is
decided BEFORE any cards for the new round are dealt. During a sit-out a
reference player continues the same play policy, exposing cards normally.
"""

from dataclasses import dataclass
from functools import cache
import random

from .oracle import CARD_PROBABILITY, PLAYER_STATES, add_card, dealer_distribution

STAND, HIT, DOUBLE, SPLIT = 0, 1, 2, 3


def count_tag(card: int) -> int:
    return 1 if 2 <= card <= 6 else -1 if card in (1, 10) else 0


@dataclass(frozen=True)
class ShoeRules:
    decks: int = 6
    penetration: float = 0.75
    blackjack_payout: float = 1.5
    double_after_split: bool = True

    def __post_init__(self):
        if not isinstance(self.decks, int) or self.decks < 1:
            raise ValueError("decks must be a positive integer")
        if not 0 < self.penetration <= 0.9:
            raise ValueError("penetration must be in (0, 0.9]")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack payout must be positive")


def pair_action(rank: int, up: int, das: bool = True) -> bool:
    """Basic-strategy pair split for S17, double-any-two, no-surrender rules.

    Aces and 8s always split; 10s and 5s never do; the rest split against weak
    up-cards (4,4 only with double-after-split). Returns True to split.
    """
    if rank in (1, 8):
        return True
    if rank in (5, 10):
        return False
    if rank == 9:
        return up in (2, 3, 4, 5, 6, 8, 9)
    if rank == 7:
        return 2 <= up <= 7
    if rank == 6:
        return 2 <= up <= 6
    if rank == 4:
        return das and up in (5, 6)
    return 2 <= up <= 7  # ranks 2 and 3


def pair_table(das: bool = True) -> dict:
    """Split/playing decisions for every pair rank vs dealer up-card."""
    return {(rank, up): SPLIT if pair_action(rank, up, das) else HIT
            for rank in range(1, 11) for up in range(1, 11)}


@cache
def play_table() -> dict:
    """Fixed infinite-deck strategy for these rules, conditional on a failed peek.

    This is an analytical baseline, not the newly learned betting model.
    No shoe composition or hidden information is used to choose player actions.
    """
    table = {}
    for up in range(1, 11):
        forbidden = 10 if up == 1 else 1 if up == 10 else None
        normalizer = 1 - CARD_PROBABILITY.get(forbidden, 0)
        outcomes = {}
        for hole, probability in CARD_PROBABILITY.items():
            if hole == forbidden:
                continue
            total, ace = add_card(*add_card(0, False, up), hole)
            for final, chance in dealer_distribution(total, ace).items():
                outcomes[final] = outcomes.get(final, 0) + probability / normalizer * chance

        def stand(total):
            if total > 21:
                return -1.0
            return sum(chance * (1 if dealer > 21 or total > dealer else -1 if total < dealer else 0)
                       for dealer, chance in outcomes.items())

        continuation = {}
        for total, ace in PLAYER_STATES:
            hit, double = 0.0, 0.0
            for card, probability in CARD_PROBABILITY.items():
                new_total, new_ace = add_card(total, ace, card)
                hit += probability * (-1 if new_total > 21 else continuation[new_total, new_ace])
                double += 2 * probability * stand(new_total)
            stay = stand(total)
            continuation[total, ace] = max(stay, hit)
            table[total, up, ace, False] = max((STAND, HIT), key=lambda a: (stay, hit)[a])
            table[total, up, ace, True] = max((STAND, HIT, DOUBLE), key=lambda a: (stay, hit, double)[a])
    return table


class Shoe:
    def __init__(self, rules: ShoeRules, seed: int, cards=None):
        self.rules = rules
        if cards is None:
            self.cards = list((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10) * (4 * rules.decks))
            random.Random(seed).shuffle(self.cards)
        else:
            # Deterministic fixtures are consumed from left to right.
            self.cards = list(reversed(cards))
        self.initial_size = len(self.cards)
        self.exposed_count = 0
        self.round_count = 0
        self.last_round = None

    @property
    def can_deal(self):
        return len(self.cards) > max(24, self.initial_size * (1 - self.rules.penetration))

    def observation(self):
        """Only public information at the betting boundary; no card-order access."""
        decks_left = len(self.cards) / 52
        return self.exposed_count / max(decks_left, 0.25), len(self.cards) / self.initial_size

    def play_round(self, actions=None, pair_actions=None, capture=False) -> float:
        """Play one round; reward is total profit per 1 initial table unit.

        Splits add one unit of exposure per extra hand (doubles add one more),
        so split-round rewards can exceed +-2. Sizing heads cap total at risk
        separately; see sizing.py.
        """
        if actions is None:
            actions = play_table()
        if pair_actions is None:
            pair_actions = pair_table(self.rules.double_after_split)
        drawn = []

        def draw():
            if not self.cards:
                raise RuntimeError("shoe exhausted; do not reshuffle inside a round")
            card = self.cards.pop()
            drawn.append(card)
            return card

        p1, up, p2, hole = draw(), draw(), draw(), draw()
        player_cards, dealer_cards, moves = [p1, p2], [up, hole], []
        stake = 1.0
        player, soft = add_card(*add_card(0, False, p1), p2)
        dealer, dealer_soft = add_card(*add_card(0, False, up), hole)
        player_natural, dealer_natural = player == 21, dealer == 21
        split_hands = None
        if dealer_natural:
            reward = 0.0 if player_natural else -1.0
        elif player_natural:
            reward = self.rules.blackjack_payout
        elif p1 == p2 and pair_actions.get((p1, up), HIT) == SPLIT:
            reward, stake, split_hands, dealer = self._play_split(
                p1, up, dealer, dealer_soft, dealer_cards, actions, draw, capture)
            player_cards, moves = split_hands[0]["cards"], split_hands[0]["moves"]
            player = split_hands[0]["total"]
        else:
            first, stake = True, 1.0
            while player <= 21:
                action = actions[player, up, soft, first]
                if capture:
                    moves.append(("stand", "hit", "double")[action])
                if action == STAND:
                    break
                card = draw()
                if capture:
                    player_cards.append(card)
                player, soft = add_card(player, soft, card)
                if action == DOUBLE:
                    if not first:
                        raise ValueError("double is only legal on the initial two cards")
                    stake = 2.0
                    break
                first = False
            if player > 21:
                reward = -stake
            else:
                while dealer < 17:
                    card = draw()
                    if capture:
                        dealer_cards.append(card)
                    dealer, dealer_soft = add_card(dealer, dealer_soft, card)
                reward = stake * (1 if dealer > 21 or player > dealer else -1 if player < dealer else 0)
        # These cards become available for NEXT round's bet decision only.
        self.exposed_count += sum(count_tag(card) for card in drawn)
        self.round_count += 1
        if capture:
            self.last_round = {"player": player_cards, "dealer": dealer_cards,
                               "player_total": player, "dealer_total": dealer,
                               "moves": moves, "stake": stake,
                               "natural": player_natural, "reward": float(reward)}
            if split_hands is not None:
                self.last_round["split"] = True
                self.last_round["hands"] = split_hands
        return float(reward)

    def _play_split(self, rank: int, up: int, dealer: int, dealer_soft: bool,
                    dealer_cards: list, actions: dict, draw, capture: bool):
        """Two (or split-ace) hands from a pair; no resplits in v1."""
        hands = []
        for _ in range(2):
            card = draw()
            cards = [rank, card]
            total, ace = add_card(*add_card(0, False, rank), card)
            hands.append({"cards": cards, "total": total, "ace": ace, "moves": [],
                          "stake": 1.0, "bust": False})
        if rank == 1:
            # Split aces receive exactly one card each and stand.
            for hand in hands:
                if capture:
                    hand["moves"].append("stand")
        else:
            for hand in hands:
                total, ace, first = hand["total"], hand["ace"], True
                can_double = self.rules.double_after_split
                while total <= 21:
                    action = actions[total, up, ace, first]
                    if action == DOUBLE and (not first or not can_double):
                        action = HIT if total != 21 else STAND
                    if capture:
                        hand["moves"].append(("stand", "hit", "double")[action])
                    if action == STAND:
                        break
                    card = draw()
                    if capture:
                        hand["cards"].append(card)
                    total, ace = add_card(total, ace, card)
                    if action == DOUBLE:
                        hand["stake"] = 2.0
                        break
                    first = False
                hand["total"], hand["ace"] = total, ace
        live = [hand for hand in hands if hand["total"] <= 21]
        for hand in hands:
            hand["bust"] = hand["total"] > 21
        if live:
            while dealer < 17:
                card = draw()
                if capture:
                    dealer_cards.append(card)
                dealer, dealer_soft = add_card(dealer, dealer_soft, card)
        reward, stake = 0.0, 0.0
        for hand in hands:
            stake += hand["stake"]
            if hand["total"] > 21:
                reward -= hand["stake"]
            else:
                reward += hand["stake"] * (1 if dealer > 21 or hand["total"] > dealer
                                           else -1 if hand["total"] < dealer else 0)
        return reward, stake, hands, dealer


def twin_seed(master: int, split: str, index: int) -> int:
    from .edge import shoe_seed
    return shoe_seed(master, split, index)


def compare_split(rules: ShoeRules, pair_rounds: int, master: int, split: str) -> dict:
    """Round-paired test: split vs play-as-normal from IDENTICAL shoe states.

    For each round dealt a splittable pair, the shoe is snapshotted, the round
    is played with splits, then the snapshot is restored and the same round is
    replayed without splitting. Same cards, same order: only pair handling
    differs, so the per-round delta has far less noise than shoe-level twins
    (whose streams diverge after the first split). Ship bar: mean delta lower-CI
    > 0. Counts as the B3 proof; shoe-level fleet numbers come from sizing runs.
    """
    import math as _math

    import numpy as np

    fixed = play_table()
    with_split = pair_table(rules.double_after_split)
    no_split: dict = {}
    delta, shoes_used, rounds_seen = [], 0, 0
    shoe = Shoe(rules, twin_seed(master, split, 0))
    while len(delta) < pair_rounds:
        if not shoe.can_deal:
            shoes_used += 1
            shoe = Shoe(rules, twin_seed(master, split, shoes_used))
            continue
        # Peek: deal a probe round to find splittable pairs without consuming.
        probe_cards = list(shoe.cards)
        p1 = probe_cards[-1]
        up = probe_cards[-2]
        p2 = probe_cards[-3]
        if not (p1 == p2 and with_split.get((p1, up), HIT) == SPLIT):
            shoe.play_round(fixed, with_split)
            rounds_seen += 1
            continue
        snapshot = (list(shoe.cards), shoe.exposed_count, shoe.round_count)
        split_reward = shoe.play_round(fixed, with_split)
        shoe.cards, shoe.exposed_count, shoe.round_count = ([*snapshot[0]], snapshot[1], snapshot[2])
        plain_reward = shoe.play_round(fixed, no_split)
        delta.append(split_reward - plain_reward)
        rounds_seen += 1
    delta = np.asarray(delta, dtype=float)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / _math.sqrt(len(delta))) if len(delta) > 1 else None
    return {
        "pair_rounds": len(delta), "rounds_seen": rounds_seen, "shoes_used": shoes_used + 1,
        "mean_delta_profit_per_splittable_round": mean,
        "mean_delta_95ci": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
    }
