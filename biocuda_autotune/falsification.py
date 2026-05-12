"""Falsification and model-test utilities from the notebook."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FalsifyResult:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TierMResult:
    name: str
    passed: bool
    measured: float | None
    predicted: float | None
    tolerance: float | None
    method: str
    notes: str = ""
    skipped: bool = False

    def as_dict(self):
        return asdict(self)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    n = len(pairs)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 0.0


def hill_r2(x: Iterable[float], y: Iterable[float], v: float, k: float, n: float) -> float:
    xs = [float(vv) for vv in x]
    ys = [float(vv) for vv in y]
    pred = []
    for xx in xs:
        xxn = max(xx, 0.0) ** n
        pred.append(v * xxn / (k**n + xxn + 1e-12))
    mean_y = sum(ys) / max(len(ys), 1)
    ss_tot = sum((yy - mean_y) ** 2 for yy in ys)
    ss_res = sum((yy - pp) ** 2 for yy, pp in zip(ys, pred))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def exp3_regret_bound(t: int, k: int, eta: float, eps_corr_max: float = 0.0) -> float:
    return math.sqrt(2.0 * t * k * math.log(max(k, 2))) + t * eta * abs(eps_corr_max)
