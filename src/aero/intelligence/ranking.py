"""Bradley-Terry ranking from pairwise outcomes (Bradley & Terry 1952; MM algorithm, Hunter 2004).

Used to rank partners from LLM pairwise judgements (ADR 0008, H8) instead of a weighted formula. Each judgement
is one game. A tie, or a position-inconsistent pair of judgements (A wins as "A", B wins as "A"), contributes
half a win to each side. Every item also plays one virtual tie against a fixed reference of strength 1: this is
the standard regularisation that keeps an undefeated item's strength finite. It does not change who beat whom.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Game:
    a: int
    b: int
    score_a: float  # 1 = a won, 0 = b won, 0.5 = tie

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("an item cannot play itself")
        if self.score_a not in (0.0, 0.5, 1.0):
            raise ValueError("score_a must be 0, 0.5 or 1")


def bradley_terry(n_items: int, games: Sequence[Game], max_iter: int = 10_000, tol: float = 1e-10) -> list[float]:
    """Log-strengths (natural log, mean zero) maximising the regularised Bradley-Terry likelihood."""
    if n_items == 0:
        return []
    wins = np.full(n_items, 0.5)  # the virtual tie against the reference
    pairs = np.zeros((n_items, n_items))
    for g in games:
        if not (0 <= g.a < n_items and 0 <= g.b < n_items):
            raise ValueError(f"game {g} refers to an unknown item")
        wins[g.a] += g.score_a
        wins[g.b] += 1.0 - g.score_a
        pairs[g.a, g.b] += 1
        pairs[g.b, g.a] += 1
    p = np.ones(n_items)
    for _ in range(max_iter):
        denom = (pairs / (p[:, None] + p[None, :])).sum(axis=1) + 1.0 / (p + 1.0)  # + reference game
        new = wins / denom
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    logs = np.log(p)
    return [float(x) for x in logs - logs.mean()]


def win_probability(log_strength_a: float, log_strength_b: float) -> float:
    """P(a beats b) under the fitted model."""
    return 1.0 / (1.0 + math.exp(log_strength_b - log_strength_a))
