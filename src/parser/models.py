"""Data containers for a single parsed hand history."""

from dataclasses import dataclass, field


@dataclass
class Seat:
    seat_no: int
    player: str
    stack: float
    position: str = ""


@dataclass
class Action:
    street: str
    player: str
    action: str  # fold, check, call, bet, raise, allin
    amount: float = 0.0


@dataclass
class Hand:
    hand_id: str
    table_name: str
    max_seats: int
    small_blind: float
    big_blind: float
    button_seat: int
    seats: list[Seat] = field(default_factory=list)
    hole_cards: dict[str, list[str]] = field(default_factory=dict)
    board: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    pot: float = 0.0
    rake: float = 0.0
    winners: dict[str, float] = field(default_factory=dict)
    showdown: dict[str, list[str]] = field(default_factory=dict)
