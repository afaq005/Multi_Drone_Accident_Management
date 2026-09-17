import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from description_agent.fusion import View, fuse_captions, normalized_weights


def test_normalized_weights_sum_to_one():
    views = [
        View(drone_id="/drone1", confidence=0.8),
        View(drone_id="/drone2", confidence=0.4),
    ]
    weights = normalized_weights(views)
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights[0] > weights[1]  # higher confidence -> higher weight


def test_fusion_prefers_highest_confidence_as_primary():
    views = [
        View(drone_id="/drone1", confidence=0.6, caption="A car accident on the highway."),
        View(drone_id="/drone2", confidence=0.95, caption="Vehicle fire with heavy smoke visible."),
    ]
    result = fuse_captions(views)
    assert result["primary_drone"] == "/drone2"
    assert "fire" in result["fused_caption"].lower() or "smoke" in result["fused_caption"].lower()


def test_fusion_appends_salient_terms_from_secondary_view():
    views = [
        View(drone_id="/drone1", confidence=0.9, caption="A car accident scene on the highway."),
        View(drone_id="/drone2", confidence=0.5, caption="Overturned vehicle with fire visible."),
    ]
    result = fuse_captions(views)
    assert "fire" in result["fused_caption"].lower()
    assert "overturned" in result["fused_caption"].lower()


def test_single_view_passthrough():
    views = [View(drone_id="/drone1", confidence=0.7, caption="Accident detected.")]
    result = fuse_captions(views)
    assert result["fused_caption"] == "Accident detected."
    assert result["weights"] == {"/drone1": 1.0}
