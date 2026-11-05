"""Per-track ripeness voting and line-crossing logic.

Pure Python (no torch / OpenCV) so it can be unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNCERTAIN = "uncertain"


def category(label: str) -> str:
    """Colour/sorting category for a model class name."""
    n = label.lower()
    if "unripe" in n:
        return "unripe"
    if "ripe" in n:
        return "ripe"
    return n


@dataclass
class Grade:
    label: str           # a model class name, or "uncertain"
    confidence: float    # mean detector confidence of the winning class
    vote_share: float    # winning class share of the weighted votes
    observations: int
    votes: dict[str, float]
    reason: str = ""
    predicted_label: str | None = None


@dataclass
class TrackVotes:
    weights: dict[str, float] = field(default_factory=dict)
    conf_sum: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    observations: int = 0

    def add(self, label: str, conf: float, certain_conf: float) -> None:
        # Low-confidence frames still count, but only a quarter as much.
        w = conf if conf >= certain_conf else conf * 0.25
        self.weights[label] = self.weights.get(label, 0.0) + w
        self.conf_sum[label] = self.conf_sum.get(label, 0.0) + conf
        self.counts[label] = self.counts.get(label, 0) + 1
        self.observations += 1

    def finalize(self, min_observations: int, min_vote_share: float, certain_conf: float,
                 require_certainty: bool = True) -> Grade:
        total = sum(self.weights.values())
        votes = {k: round(v / total, 3) for k, v in self.weights.items()} if total else {}
        if not self.weights:
            return Grade(UNCERTAIN, 0.0, 0.0, 0, {}, "no observations")
        top = max(self.weights, key=self.weights.get)
        share = self.weights[top] / total
        mean_conf = self.conf_sum[top] / self.counts[top]
        base = dict(confidence=round(mean_conf, 3), vote_share=round(share, 3),
                    observations=self.observations, votes=votes, predicted_label=top)
        if self.observations < min_observations:
            return Grade(UNCERTAIN, reason=f"Too few frames: {self.observations} / {min_observations} required", **base)
        if not require_certainty and sum(abs(weight - self.weights[top]) < 1e-9 for weight in self.weights.values()) > 1:
            return Grade(UNCERTAIN, reason="Conflicting class predictions: no winning class", **base)
        if require_certainty and share < min_vote_share:
            return Grade(UNCERTAIN, reason=f"classes disagree: {share:.0%} agreement / {min_vote_share:.0%} required", **base)
        if require_certainty and mean_conf < certain_conf:
            return Grade(UNCERTAIN, reason=f"low confidence: {mean_conf:.0%} / {certain_conf:.0%} required", **base)
        return Grade(top, **base)


def side_of_line(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """>0 on the 'positive' side, <0 on the negative side (image coordinates, y down)."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def within_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float, margin: float = 0.05) -> bool:
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return False
    t = ((px - x1) * dx + (py - y1) * dy) / length2
    return -margin <= t <= 1 + margin


def crossed(prev_side: float, side: float, direction: str) -> bool:
    if prev_side == 0 or side == 0 or (prev_side > 0) == (side > 0):
        return False
    if direction == "both":
        return True
    forward = prev_side < 0 < side
    return forward if direction == "forward" else not forward
