import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..")
)

from dispatch_agent.rescue_center_router import (
    build_alert_payload,
    compute_severity,
    nearest_rescue_center,
    route_alert,
)
from model.utils import sigmoid


def test_severity_score_is_bounded_0_1():
    s = compute_severity(
        conf_yolo=0.9,
        log_p_blip2=-0.3,
    )
    assert 0.0 <= s <= 1.0


def test_severity_increases_with_confidence():
    low = compute_severity(
        conf_yolo=0.2,
        log_p_blip2=-2.0,
    )
    high = compute_severity(
        conf_yolo=0.95,
        log_p_blip2=-0.1,
    )
    assert high > low


def test_nearest_rescue_center_gps_picks_closest():
    centers = [
        ("A", 37.3355, -122.0095),
        ("B", 37.3400, -122.0050),
        ("C", 37.3450, -122.0130),
    ]

    name, dist = nearest_rescue_center(
        37.3356,
        -122.0094,
        centers,
        coordinate_mode="gps",
    )

    assert name == "A"
    assert dist >= 0.0


def test_nearest_rescue_center_local_picks_closest():
    centers = [
        ("A", 0.0, 0.0),
        ("B", 10.0, 10.0),
        ("C", 100.0, 100.0),
    ]

    name, dist = nearest_rescue_center(
        1.0,
        1.0,
        centers,
        coordinate_mode="local",
    )

    assert name == "A"
    assert dist >= 0.0


def test_route_alert_gps_attaches_center_and_distance():
    payload = build_alert_payload(
        incident_type="fire",
        summary="Vehicle fire on shoulder.",
        coord_a=37.3355,
        coord_b=-122.0095,
        conf_yolo=0.9,
        log_p_blip2=-0.2,
        coordinate_mode="gps",
    )

    routed = route_alert(payload)

    assert "rescue_center" in routed
    assert "distance_to_center_m" in routed
    assert routed["coordinate_mode"] == "gps"
    assert routed["severity"] == payload["severity"]


def test_route_alert_local_attaches_center_and_distance():
    centers = [
        ("A", 0.0, 0.0),
        ("B", 20.0, 20.0),
    ]

    payload = build_alert_payload(
        incident_type="accident",
        summary="Multi-vehicle collision.",
        coord_a=2.0,
        coord_b=1.0,
        conf_yolo=0.85,
        log_p_blip2=-0.4,
        coordinate_mode="local",
    )

    routed = route_alert(
        payload,
        centers=centers,
    )

    assert routed["rescue_center"] == "A"
    assert "distance_to_center_local" in routed
    assert routed["coordinate_mode"] == "local"
    assert routed["severity"] == payload["severity"]


def test_sigmoid_monotonic():
    assert sigmoid(-10) < sigmoid(0) < sigmoid(10)
