#!/usr/bin/env python3
"""
description_agent/fusion.py
============================
Implements the confidence-weighted multi-view caption fusion of
Section 6.5, Eq. 23:

    y_hat = argmax_y  sum_i w_i * p_BLIP2(y | I_i),   w_i = conf_i / sum_k conf_k

Rather than sampling over the full caption space (intractable), we realize
this — exactly as the paper describes in the paragraph following Eq. 23 —
"by conditioning the description agent's generation on the highest-
confidence view while incorporating auxiliary detail from secondary
views." Concretely:

  1. Compute normalized weights w_i from each view's perception-agent
     confidence (Eq. 21's confidence score).
  2. Generate the primary caption from the highest-confidence frame.
  3. Generate secondary captions from the remaining views.
  4. Merge: keep the primary caption as the report's backbone and append
     any additional entities/details present in secondary captions but
     absent from the primary one, weighted by their w_i.

This module is intentionally decoupled from the model-loading code in
inference.py so it can be unit-tested with plain caption strings.
"""
import re
import sys
from dataclasses import dataclass
from typing import Callable, List

import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.utils import get_logger  # noqa: E402

logger = get_logger("description_agent.fusion")

# A small vocabulary of scene entities worth surfacing from secondary
# views even if the primary caption omits them (e.g. one drone sees fire,
# another sees the flipped vehicle).
SALIENT_TERMS = [
    "fire", "flames", "smoke", "overturned", "flipped", "rollover",
    "pedestrian", "motorcycle", "truck", "multiple vehicles", "debris",
    "injured", "blocked lane", "guardrail",
]


@dataclass
class View:
    """One drone's observation of the same incident."""
    drone_id: str
    confidence: float          # conf_i from Eq. 21
    caption: str = ""          # p_BLIP2(y | I_i), materialized as text


def normalized_weights(views: List[View]) -> List[float]:
    """w_i = conf_i / sum_k conf_k, per Eq. 23."""
    total = sum(v.confidence for v in views)
    if total <= 0:
        return [1.0 / len(views)] * len(views)
    return [v.confidence / total for v in views]


def _extract_missing_salient_terms(primary_caption: str, other_caption: str) -> List[str]:
    primary_lower = primary_caption.lower()
    other_lower = other_caption.lower()
    missing = []
    for term in SALIENT_TERMS:
        if term in other_lower and term not in primary_lower:
            missing.append(term)
    return missing


def fuse_captions(views: List[View]) -> dict:
    """Fuses multiple per-drone captions of the same incident into a single
    confidence-weighted summary (Eq. 23).

    Returns a dict with the fused text, the chosen primary view, and the
    per-view weights (useful for the dispatch agent's severity score,
    Eq. 24, which needs `conf_i` and `p_BLIP2(y_hat | I)`).
    """
    if not views:
        raise ValueError("fuse_captions requires at least one view")
    if len(views) == 1:
        v = views[0]
        return {
            "fused_caption": v.caption,
            "primary_drone": v.drone_id,
            "weights": {v.drone_id: 1.0},
        }

    weights = normalized_weights(views)
    ranked = sorted(zip(views, weights), key=lambda vw: vw[1], reverse=True)
    primary_view, primary_weight = ranked[0]
    secondary = ranked[1:]

    fused_text = primary_view.caption.rstrip(". ")
    appended_terms: List[str] = []
    for view, _w in secondary:
        missing = _extract_missing_salient_terms(fused_text, view.caption)
        for term in missing:
            if term not in appended_terms:
                appended_terms.append(term)

    if appended_terms:
        fused_text += f"; additional drone views also report {', '.join(appended_terms)}"
    fused_text = fused_text.strip()
    if not fused_text.endswith("."):
        fused_text += "."

    logger.info(
        f"Fused {len(views)} views -> primary={primary_view.drone_id} "
        f"(w={primary_weight:.2f}); appended terms: {appended_terms}"
    )

    return {
        "fused_caption": fused_text,
        "primary_drone": primary_view.drone_id,
        "primary_weight": primary_weight,
        "weights": {v.drone_id: w for v, w in zip(views, weights)},
    }


if __name__ == "__main__":
    # Simple smoke test / usage example.
    example_views = [
        View(drone_id="/drone1", confidence=0.91,
             caption="A car accident scene on the highway; two vehicles collided head-on"),
        View(drone_id="/drone2", confidence=0.74,
             caption="Overturned vehicle with smoke visible near the guardrail"),
    ]
    result = fuse_captions(example_views)
    print(result["fused_caption"])
