# tests/test_pipeline_integration.py

import pytest

from planning_agent.llm_waypoint_node import (
    add_mission_metadata,
    normalize_planning_payload,
)
from model.utils import validate_planning_output
from coordination_agent.assignment_solver import solve_assignment
from coordination_agent.coordination_node import (
    _prepare_incidents_payload,
    _validate_local_pipeline,
)
from coordination_agent.mavros_waypoint_adapter import WaypointTracker
from description_agent.report_aggregator import IncidentReportAggregator
from dispatch_agent.rescue_center_router import (
    build_alert_payload,
    route_alert,
)
from dispatch_agent.ros_dispatch_node import fused_report_to_alert


def test_historical_waypoint_mapping_normalizes_and_adds_ids():
    payload = normalize_planning_payload(
        {
            "/drone1/waypoints": [[-165, -1.43, 10]],
            "/drone2/waypoints": [[102.2, 4.1, 10]],
        }
    )

    out = add_mission_metadata(payload)

    assert out["coordinate_mode"] == "local"
    assert len(out["incidents"]) == 2
    assert out["incidents"][0]["drone"] == "/drone1"

    assert out["incidents"][0][
        "incident_id"
    ].startswith(out["mission_id"])


def test_insufficient_waypoint_payload_is_valid_non_actionable_output():
    assert validate_planning_output(
        {
            "incidents": [],
            "intent": "insufficient_waypoint_information",
            "coordinate_mode": "local",
        }
    )


def test_k_greater_than_n_is_rejected():
    with pytest.raises(
        ValueError,
        match="infeasible",
    ):
        solve_assignment(
            {
                "/drone1": (
                    0.0,
                    0.0,
                    0.0,
                )
            },
            [
                {
                    "lat": 0.0,
                    "lon": 0.0,
                    "alt": 10.0,
                },
                {
                    "lat": 1.0,
                    "lon": 1.0,
                    "alt": 10.0,
                },
            ],
            coordinate_mode="local",
        )


def test_duplicate_operator_drone_assignment_is_rejected():
    with pytest.raises(
        ValueError,
        match="more than one incident",
    ):
        solve_assignment(
            {
                "/drone1": (
                    0.0,
                    0.0,
                    0.0,
                ),
                "/drone2": (
                    1.0,
                    1.0,
                    0.0,
                ),
            },
            [
                {
                    "lat": 0,
                    "lon": 0,
                    "alt": 10,
                    "drone": "/drone1",
                },
                {
                    "lat": 1,
                    "lon": 1,
                    "alt": 10,
                    "drone": "/drone1",
                },
            ],
            coordinate_mode="local",
        )


def test_ros_coordination_rejects_gps_payload():
    prepared = _prepare_incidents_payload(
        {
            "coordinate_mode": "gps",
            "incidents": [
                {
                    "lat": 37.0,
                    "lon": 127.0,
                    "alt": 10,
                }
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="local",
    ):
        _validate_local_pipeline(
            prepared
        )


def test_local_dispatch_requires_explicit_centers():
    payload = build_alert_payload(
        "accident",
        "Collision.",
        2.0,
        1.0,
        0.9,
        -0.3,
        coordinate_mode="local",
    )

    with pytest.raises(
        ValueError,
        match="explicit rescue-center",
    ):
        route_alert(payload)


def test_aggregator_and_dispatch_preserve_primary_severity_inputs():
    aggregator = IncidentReportAggregator(
        fusion_window_s=0
    )

    base = {
        "mission_id": "M1",
        "incident_id": "M1-I01",
        "incident_index": 0,
        "incident_type": "accident",
        "coordinate_mode": "local",
        "assigned_target": {
            "x": 1.0,
            "y": 2.0,
            "z": 10.0,
        },
    }

    aggregator.add_report(
        {
            **base,
            "event_id": "E1",
            "drone_id": "/drone1",
            "caption": "A car accident.",
            "confidence": 0.9,
            "log_likelihood": -0.4,
        }
    )

    aggregator.add_report(
        {
            **base,
            "event_id": "E2",
            "drone_id": "/drone2",
            "caption": (
                "Smoke near an overturned vehicle."
            ),
            "confidence": 0.7,
            "log_likelihood": -0.7,
        }
    )

    fused = aggregator.flush_ready(
        force=True
    )[0]

    assert fused[
        "primary_drone"
    ] == "/drone1"

    assert fused[
        "severity_confidence"
    ] == pytest.approx(0.9)

    assert fused[
        "severity_log_likelihood"
    ] == pytest.approx(-0.4)

    alert = fused_report_to_alert(
        fused,
        centers=[
            (
                "A",
                0.0,
                0.0,
            )
        ],
    )

    assert alert[
        "incident_id"
    ] == "M1-I01"

    assert alert[
        "rescue_center"
    ] == "A"


def test_waypoint_tracker_advances_and_completes():
    tracker = WaypointTracker(
        tolerance_m=0.25
    )

    tracker.set_waypoints(
        [
            (
                1,
                0,
                0,
            ),
            (
                2,
                0,
                0,
            ),
        ]
    )

    assert not tracker.update_position(
        (
            0,
            0,
            0,
        )
    )

    assert tracker.update_position(
        (
            1,
            0,
            0,
        )
    )

    assert tracker.current_waypoint() == (
        2.0,
        0.0,
        0.0,
    )

    assert tracker.update_position(
        (
            2,
            0,
            0,
        )
    )

    assert tracker.complete()
