import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dispatch_agent.rescue_center_router import (
    build_alert_payload,
    compute_severity,
    nearest_rescue_center,
    route_alert,
)
from model.utils import sigmoid


def test_severity_score_is_bounded_0_1():
    s = compute_severity(conf_yolo=0.9, log_p_blip2=-0.3)
    assert 0.0 <= s <= 1.0


def test_severity_increases_with_confidence():
    low = compute_severity(conf_yolo=0.2, log_p_blip2=-2.0)
    high = compute_severity(conf_yolo=0.95, log_p_blip2=-0.1)
    assert high > low


def test_nearest_rescue_center_picks_closest():
    centers = [
        ("A", 0.0, 0.0),
        ("B", 1.0, 1.0),
        ("C", 10.0, 10.0),
    ]
    name, dist = nearest_rescue_center(0.05, 0.05, centers)
    assert name == "A"
    assert dist >= 0


def test_route_alert_attaches_center_and_distance():
    payload = build_alert_payload(
        incident_type="fire", summary="Vehicle fire on shoulder.",
        lat=37.3355, lon=-122.0095, conf_yolo=0.9, log_p_blip2=-0.2,
    )
    routed = route_alert(payload)
    assert "rescue_center" in routed
    assert "distance_to_center_m" in routed
    assert routed["severity"] == payload["severity"]


def test_sigmoid_monotonic():
    assert sigmoid(-10) < sigmoid(0) < sigmoid(10)
