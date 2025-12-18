import numpy as np
import numpy.typing as npt
from typing import Literal


def continuous_segments(array: np.ndarray, value: int) -> np.ndarray:
    a = np.asarray(array)
    if a.ndim != 1:
        a = a.ravel()

    eq = (a == value)
    if eq.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    start = eq.copy()
    start[1:] &= ~eq[:-1]
    starts = np.flatnonzero(start)

    end = np.empty_like(eq)
    end[:-1] = eq[:-1] & ~eq[1:]
    end[-1] = eq[-1]
    ends = np.flatnonzero(end) + 1

    out = np.empty((starts.size, 2), dtype=np.int64)
    out[:, 0] = starts
    out[:, 1] = ends
    return out



def compute_events_fast(targets: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    t = np.asarray(targets)
    if t.ndim != 1:
        t = t.ravel()
    n = t.size
    if n == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Use your fast segments extractor for anomalies: targets > 0
    segs = continuous_segments(t > 0, True)  # (M,2)
    M = segs.shape[0]
    if M == 0:
        return np.empty((0, 4), dtype=np.int64)

    out = np.empty((M, 4), dtype=np.int64)
    out[:, 0] = segs[:, 0]
    out[:, 3] = segs[:, 1]

    for i, (s, e) in enumerate(segs):
        w = t[s:e]

        m1 = (w > 1)
        out[i, 1] = (s + int(m1.argmax())) if m1.any() else e

        m2 = (w > 2)
        out[i, 2] = (s + int(m2.argmax())) if m2.any() else e

    return out



def compute_events(targets: npt.NDArray[np.integer]) -> npt.NDArray[np.int64]:
    t = np.asarray(targets)
    if t.ndim != 1:
        t = t.ravel()
    n = t.size
    if n == 0:
        return np.empty((0, 4), dtype=np.int64)

    segs = continuous_segments(t > 0, True)  # (M,2)
    M = segs.shape[0]
    if M == 0:
        return np.empty((0, 4), dtype=np.int64)

    starts = segs[:, 0]
    ends   = segs[:, 1]
    coverage = int(np.sum(ends - starts))

    # Tune these for your workload:
    FEW_M = 32
    FEW_COVERAGE = 4096

    if M <= FEW_M and coverage <= FEW_COVERAGE:
        out = np.empty((M, 4), dtype=np.int64)
        out[:, 0] = starts
        out[:, 3] = ends
        for i, (s, e) in enumerate(segs):
            w = t[s:e]

            # first index where w > 1, else end
            m1 = (w > 1)
            t1 = (s + int(m1.argmax())) if m1.any() else e

            # first index where w > 2, else end
            m2 = (w > 2)
            t2 = (s + int(m2.argmax())) if m2.any() else e

            out[i, 1] = t1
            out[i, 2] = t2
        return out

    # Otherwise use the O(n) vectorized version
    return compute_events_fast(t)


def _edl(predictions: np.ndarray, targets: np.ndarray, mode: Literal["early", "late"]) -> float:
    preds = np.asarray(predictions)
    t = np.asarray(targets)
    if preds.ndim != 1:
        preds = preds.ravel()
    if t.ndim != 1:
        t = t.ravel()

    ev = compute_events(t)  # (M,4): [start, t1, t2, end]
    if ev.shape[0] == 0:
        return 0.0

    preds_b = (preds != 0)
    p = np.empty(preds_b.size + 1, dtype=np.int32)
    p[0] = 0
    np.cumsum(preds_b, out=p[1:])

    s = ev[:, 0]
    e = ev[:, 3]
    detected = (p[e] - p[s]) != 0
    if not np.any(detected):
        return 0.0

    ev = ev[detected]
    s = ev[:, 0]
    t1 = ev[:, 1]

    early = (p[t1] - p[s]) != 0
    if mode == "early":
        return float(np.mean(early))
    else:
        # late = detected but not early (matches your LDF)
        return float(np.mean(~early))
