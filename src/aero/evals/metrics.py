"""Agreement, classification, ranking and calibration metrics (ADR 0008 release gates). Pure numpy, tested.

* ``cohen_kappa``: chance-corrected agreement between a judge and a human (nominal, or weighted for ordinals).
* ``prf``: precision / recall / F1 for set matching (extracted vs gold signals).
* ``kendall_tau_b``: rank agreement with ties (partner rankings vs analysts).
* ``expected_calibration_error`` and ``fit_isotonic``: when a probability may be shown (plan §5).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable] | None = None,
                weights: Literal["none", "linear", "quadratic"] = "none") -> float:
    """Cohen's kappa. For ordinal scales pass ``labels`` in order and ``weights='quadratic'``."""
    if len(a) != len(b):
        raise ValueError("rater sequences differ in length")
    if not a:
        raise ValueError("no items")
    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[idx[x], idx[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights == "none":
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    else:
        w = ((i - j) / max(k - 1, 1)) ** 2
    denom = float((w * expected).sum())
    if denom == 0:
        return 1.0  # both raters used one identical category throughout
    return 1.0 - float((w * observed).sum()) / denom


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf(predicted: AbstractSet[Hashable], gold: AbstractSet[Hashable]) -> PRF:
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(round(p, 4), round(r, 4), round(f, 4), tp, fp, fn)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two score vectors over the same items (handles ties)."""
    if len(x) != len(y):
        raise ValueError("length mismatch")
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denom) if denom else 0.0


def ranking_tau(system: Sequence[str], gold: Sequence[str]) -> float:
    """tau-b between two orderings, over the items both contain (rank 1 = best)."""
    common = [g for g in gold if g in system]
    if len(common) < 2:
        return 0.0
    return kendall_tau_b([-gold.index(c) for c in common], [-list(system).index(c) for c in common])


def expected_calibration_error(probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum over bins of |accuracy - confidence| weighted by bin size."""
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if p.shape != o.shape or p.size == 0:
        raise ValueError("probs and outcomes must be non-empty and the same length")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(o[mask].mean() - p[mask].mean())
    return float(ece)


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone step function fitted by pool-adjacent-violators (Zadrozny & Elkan 2002)."""

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, x: float) -> float:
        i = int(np.searchsorted(self.thresholds, x, side="right")) - 1
        return self.values[max(i, 0)]


def fit_isotonic(x: Sequence[float], y: Sequence[int]) -> IsotonicCalibrator:
    order = np.argsort(np.asarray(x, dtype=float), kind="mergesort")
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    blocks: list[list[float]] = []  # [sum_y, count, start_x]
    for xi, yi in zip(xs, ys, strict=True):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, _ = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    return IsotonicCalibrator(tuple(float(b[2]) for b in blocks), tuple(float(b[0] / b[1]) for b in blocks))


ICD203 = (
    (0.05, "almost no chance"), (0.20, "very unlikely"), (0.45, "unlikely"), (0.55, "roughly even chance"),
    (0.80, "likely"), (0.95, "very likely"), (1.00, "almost certain"),
)


def icd203_term(probability: float) -> str:
    """ICD 203 estimative term for a *calibrated* probability (plan §3). Never call it on an uncalibrated score."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return next(term for upper, term in ICD203 if probability <= upper)
