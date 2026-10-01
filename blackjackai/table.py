"""Multiplayer blackjack table using the same rules as the single-hand trainer.

Unlimited deck (draws with replacement), dealer stands on all 17s,
even-money rewards, no naturals premium. Each seat is an independent
hand against the same dealer; draws do not deplete a shoe so the
single-agent DQN advice stays valid per seat.
"""

from collections import deque
from dataclasses import dataclass, field
import random

from .game import hand_value


@dataclass
class MultiTable:
    """One multiplayer hand: 1-5 seats vs a single dealer."""

    num_seats: int = 2
    seed: int | None = None
    forced_cards: list[int] | None = None
    hands: list[list[int]] = field(default_factory=list, init=False)
    dealer: list[int] = field(default_factory=list, init=False)
    seat_status: list[str] = field(default_factory=list, init=False)
    seat_results: list[str | None] = field(default_factory=list, init=False)
    stage: str = field(default="playing", init=False)
    current_seat: int | None = field(default=None, init=False)

    def __post_init__(self):
        if not 1 <= self.num_seats <= 5:
            raise ValueError("seat count must be between 1 and 5")
        self._rng = random.Random(self.seed)
        self._forced = deque(self.forced_cards or [])
        self.hands = [[self._draw(), self._draw()] for _ in range(self.num_seats)]
        self.dealer = [self._draw(), self._draw()]
        self.seat_status = []
        self.seat_results = []
        for hand in self.hands:
            # A dealt 21 still requires an explicit stand to match the
            # single-hand trainer (no natural shortcut, even money).
            total, _ = hand_value(hand)
            self.seat_status.append("playing" if total <= 21 else "bust")
            self.seat_results.append("loss" if total > 21 else None)
        self.stage = "playing"
        self._advance_current()
        if self.current_seat is None:
            self._finish()

    def _draw(self) -> int:
        if self._forced:
            card = self._forced.popleft()
            if not 1 <= card <= 10:
                raise ValueError("cards must be between 1 (ace) and 10")
            return card
        return self._rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10))

    def observation(self, seat: int) -> tuple[int, int, int]:
        """Model-visible state for one seat; never exposes the hole card."""
        self._require_seat(seat)
        total, usable_ace = hand_value(self.hands[seat])
        return total, self.dealer[0], int(usable_ace)

    def hit(self, seat: int) -> None:
        self._require_action(seat)
        self.hands[seat].append(self._draw())
        if hand_value(self.hands[seat])[0] > 21:
            self.seat_status[seat] = "bust"
            self.seat_results[seat] = "loss"
        self._advance_current()
        if self.current_seat is None:
            self._finish()

    def stand(self, seat: int) -> None:
        self._require_action(seat)
        self.seat_status[seat] = "stood"
        self._advance_current()
        if self.current_seat is None:
            self._finish()

    def _advance_current(self) -> None:
        self.current_seat = next(
            (i for i, status in enumerate(self.seat_status) if status == "playing"),
            None,
        )

    def _finish(self) -> None:
        self.stage = "finished"
        self.current_seat = None
        active = [i for i, s in enumerate(self.seat_status) if s == "stood"]
        if active:
            while hand_value(self.dealer)[0] < 17:
                self.dealer.append(self._draw())
        dealer_total = hand_value(self.dealer)[0]
        dealer_bust = dealer_total > 21
        for i in active:
            player_total = hand_value(self.hands[i])[0]
            if dealer_bust or player_total > dealer_total:
                self.seat_results[i] = "win"
            elif player_total < dealer_total:
                self.seat_results[i] = "loss"
            else:
                self.seat_results[i] = "draw"
        self.seat_status = [
            "finished" if s in ("stood", "bust") else s for s in self.seat_status
        ]

    def _require_seat(self, seat: int) -> None:
        if not isinstance(seat, int) or not 0 <= seat < self.num_seats:
            raise ValueError(f"seat must be an integer from 0 to {self.num_seats - 1}")

    def _require_action(self, seat: int) -> None:
        if self.stage != "playing":
            raise ValueError("this hand is already finished")
        self._require_seat(seat)
        if seat != self.current_seat:
            raise ValueError(f"it is seat {self.current_seat}'s turn, not seat {seat}")
        if self.seat_status[seat] != "playing":
            raise ValueError(f"seat {seat} is already {self.seat_status[seat]}")

    def public_state(self) -> dict:
        reveal = self.stage == "finished"
        seats = []
        for i, hand in enumerate(self.hands):
            total, _ = hand_value(hand)
            seats.append(
                {
                    "seat": i,
                    "cards": list(hand),
                    "total": total,
                    "status": self.seat_status[i],
                    "result": self.seat_results[i],
                    "active": i == self.current_seat,
                }
            )
        return {
            "seats": seats,
            "num_seats": self.num_seats,
            "dealer": list(self.dealer) if reveal else [self.dealer[0], None],
            "dealer_total": hand_value(self.dealer)[0] if reveal else self.dealer[0],
            "stage": self.stage,
            "current_seat": self.current_seat,
        }
