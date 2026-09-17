import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coordination_agent.assignment_solver import (
    build_waypoint_payload,
    solve_assignment,
)


def test_optimal_assignment_matches_nearest_drone():
    """4 drones, 4 incidents arranged so the optimal (min total distance)
    solution is the identity matching — verifies Eq. 2's Hungarian solve."""
    drone_positions = {
        "/drone1": (0, 0, 0),
        "/drone2": (100, 0, 0),
        "/drone3": (0, 100, 0),
        "/drone4": (100, 100, 0),
    }
    incidents = [
        {"lat": 1, "lon": 1, "alt": 10},
        {"lat": 99, "lon": 1, "alt": 10},
        {"lat": 1, "lon": 99, "alt": 10},
        {"lat": 99, "lon": 99, "alt": 10},
    ]
    assignments = solve_assignment(drone_positions, incidents)
    assert assignments["/drone1"] == 0
    assert assignments["/drone2"] == 1
    assert assignments["/drone3"] == 2
    assert assignments["/drone4"] == 3


def test_explicit_operator_assignment_is_honored():
    """Section 6.2 hybrid mode: an explicit `drone` field bypasses Eq. 2
    optimization for that incident."""
    drone_positions = {"/drone1": (0, 0, 0), "/drone2": (1000, 1000, 0)}
    incidents = [
        {"lat": 1, "lon": 1, "alt": 10, "drone": "/drone2"},  # forced, even though far
        {"lat": 2, "lon": 2, "alt": 10},                       # resolved autonomously
    ]
    assignments = solve_assignment(drone_positions, incidents)
    assert assignments["/drone2"] == 0
    assert assignments["/drone1"] == 1


def test_waypoint_payload_shape():
    drone_positions = {"/drone1": (0, 0, 0)}
    incidents = [{"lat": 5, "lon": 5, "alt": 10}]
    assignments = solve_assignment(drone_positions, incidents)
    payload = build_waypoint_payload(assignments, incidents)
    assert payload == {"/drone1/waypoints": [[5, 5, 10]]}
