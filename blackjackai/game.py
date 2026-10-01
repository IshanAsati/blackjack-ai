"""Interactive blackjack game state using the same rules as the training environment."""

from collections import deque
from dataclasses import dataclass, field
import random


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Return the best non-busting total when possible, plus usable-ace status."""
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21
    return total + 10 if usable_ace else total, usable_ace


@dataclass
class Game:
    """One hand of hit/stand blackjack with dealer play on stand."""

    seed: int | None = None
    forced_cards: list[int] | None = None
    player: list[int] = field(default_factory=list, init=False)
    dealer: list[int] = field(default_factory=list, init=False)
    status: str = field(default="playing", init=False)
    result: str | None = field(default=None, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._forced = deque(self.forced_cards or [])
        # Card identity beyond value is irrelevant under the training rules.
        self.player = [self._draw(), self._draw()]
        self.dealer = [self._draw(), self._draw()]

    def _draw(self) -> int:
        if self._forced:
            card = self._forced.popleft()
            if not 1 <= card <= 10:
                raise ValueError("cards must be between 1 (ace) and 10")
            return card
        return self._rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10))

    @property
    def observation(self) -> tuple[int, int, int]:
        total, usable_ace = hand_value(self.player)
        return total, self.dealer[0], int(usable_ace)

    def hit(self) -> None:
        self._require_playing()
        self.player.append(self._draw())
        if hand_value(self.player)[0] > 21:
            self.status, self.result = "finished", "loss"

    def stand(self) -> None:
        self._require_playing()
        while hand_value(self.dealer)[0] < 17:
            self.dealer.append(self._draw())
        player_total = hand_value(self.player)[0]
        dealer_total = hand_value(self.dealer)[0]
        self.status = "finished"
        if dealer_total > 21 or player_total > dealer_total:
            self.result = "win"
        elif player_total < dealer_total:
            self.result = "loss"
        else:
            self.result = "draw"

    def _require_playing(self) -> None:
        if self.status != "playing":
            raise ValueError("this hand is already finished")

    def public_state(self) -> dict:
        player_total, _ = hand_value(self.player)
        reveal = self.status == "finished"
        state = {
            "player": self.player,
            "player_total": player_total,
            "dealer": self.dealer if reveal else [self.dealer[0], None],
            "dealer_total": hand_value(self.dealer)[0] if reveal else self.dealer[0],
            "status": self.status,
            "result": self.result,
        }
        return state
