"""Assign table positions relative to the button, including short-handed tables."""

from src.config import POSITIONS_6MAX, POSITIONS_8MAX
from src.parser.models import Hand

_TABLES_BY_SEATS = {
    6: POSITIONS_6MAX,
    7: ("BTN", "SB", "BB", "UTG", "MP", "MP+1", "CO"),
    8: POSITIONS_8MAX,
}


def assign_positions(hand: Hand) -> None:
    """Mutates hand.seats in place, setting `.position` for each active seat.

    Incomplete (short-handed) tables are handled by mapping the N active
    seats onto the last N labels of the full-ring template, so BTN/SB/BB stay
    fixed and the missing seats are early/middle positions -- consistent with
    how short-handed ranges widen from the back of the order forward.
    """
    active = sorted(hand.seats, key=lambda s: s.seat_no)
    n = len(active)
    if n < 2:
        return

    button_idx = next((i for i, s in enumerate(active) if s.seat_no == hand.button_seat), 0)
    order = active[button_idx:] + active[:button_idx]

    labels = _TABLES_BY_SEATS.get(n) or _collapse_labels(n)

    for seat, label in zip(order, labels):
        seat.position = label


def _collapse_labels(n: int) -> tuple[str, ...]:
    """Fallback for short-handed tables outside the predefined 6/7/8 templates
    (2-5 active seats). BTN/SB/BB stay fixed and CO stays fixed as the seat
    right before the button; only the early/middle positions collapse.
    """
    if n == 2:
        # Heads-up: the button posts the small blind itself, so the two
        # positions are BTN and BB -- there is no separate SB seat.
        return ("BTN", "BB")

    base = _TABLES_BY_SEATS[6]  # ("BTN", "SB", "BB", "UTG", "MP", "CO")
    head, middle_pool, tail = base[:3], base[3:-1], base[-1:]

    if n <= 3:
        return head[:n]

    n_middle = n - len(head) - len(tail)
    middle = middle_pool[-n_middle:] if n_middle > 0 else ()
    return head + middle + tail
