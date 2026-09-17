#!/usr/bin/env python3
"""
coordination_agent/assignment_solver.py
========================================
Implements A_coord from Section 3.2 / Eq. 2:

    min_X  sum_k sum_j c_kj * x_kj
    s.t.   sum_j x_kj = 1  (every incident gets exactly one drone)
           sum_k x_kj <= 1 (every drone is assigned to at most one incident)
           x_kj in {0, 1}

This is the classic linear-sum-assignment (min-cost bipartite matching)
problem, solved here with the Hungarian algorithm via
scipy.optimize.linear_sum_assignment for K == N (falls back to a padded
cost matrix for K != N, matching the paper's note that "the formulation
generalizes to K != N").

Also implements the hybrid dispatch mode of Section 6.2: if the planning
agent already supplied an explicit `drone` field for an incident, that
mapping is honored directly instead of being re-optimized.
"""
import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import DRONE_NAMESPACES  # noqa: E402
from model.utils import euclidean_distance, get_logger  # noqa: E402

logger = get_logger("coordination_agent")

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - graceful fallback
    _HAVE_SCIPY = False


def _build_cost_matrix(
    drone_positions: Dict[str, Tuple[float, float, float]],
    incidents: List[dict],
) -> Tuple["np.ndarray", List[str]]:
    """Cost c_kj = Euclidean distance from drone j's current position to
    incident k (paper's implementation choice, Section 3.2)."""
    drones = list(drone_positions.keys())
    K, N = len(incidents), len(drones)
    size = max(K, N)  # pad to square matrix so unmatched slots cost 0
    cost = np.zeros((size, size), dtype=float)
    for k, inc in enumerate(incidents):
        incident_pos = (float(inc["lat"]), float(inc["lon"]), float(inc.get("alt", 0.0)))
        for j, drone in enumerate(drones):
            cost[k, j] = euclidean_distance(drone_positions[drone], incident_pos)
    return cost, drones


def solve_assignment(
    drone_positions: Dict[str, Tuple[float, float, float]],
    incidents: List[dict],
) -> Dict[str, int]:
    """Resolves drone -> incident-index allocation.

    Honors any explicit per-drone mapping supplied by the planning agent
    (Section 6.2 hybrid mode); the remaining unassigned drones/incidents
    are resolved via the Hungarian algorithm (Eq. 2).

    Returns: {drone_namespace: incident_index}
    """
    assignments: Dict[str, int] = {}
    remaining_incidents: List[int] = []
    remaining_drones = set(drone_positions.keys())

    # Step 1: honor explicit operator-directed assignments.
    for idx, inc in enumerate(incidents):
        drone = inc.get("drone")
        if drone and drone in remaining_drones:
            assignments[drone] = idx
            remaining_drones.discard(drone)
        else:
            remaining_incidents.append(idx)

    if not remaining_incidents or not remaining_drones:
        return assignments

    # Step 2: solve Eq. 2 for whatever is left.
    sub_incidents = [incidents[i] for i in remaining_incidents]
    sub_drone_positions = {d: drone_positions[d] for d in remaining_drones}

    if not _HAVE_SCIPY:
        logger.warning(
            "scipy not installed; falling back to greedy nearest-neighbor "
            "assignment instead of the optimal Hungarian solution."
        )
        assignments.update(
            _greedy_assignment(sub_drone_positions, sub_incidents, remaining_incidents)
        )
        return assignments

    cost, drones = _build_cost_matrix(sub_drone_positions, sub_incidents)
    row_ind, col_ind = linear_sum_assignment(cost)

    K, N = len(sub_incidents), len(drones)
    for r, c in zip(row_ind, col_ind):
        if r < K and c < N:  # ignore padded (dummy) rows/cols
            assignments[drones[c]] = remaining_incidents[r]

    return assignments


def _greedy_assignment(
    drone_positions: Dict[str, Tuple[float, float, float]],
    incidents: List[dict],
    incident_indices: List[int],
) -> Dict[str, int]:
    """Fallback used only if scipy is unavailable; not the paper's optimal
    solver, but keeps the pipeline runnable in minimal environments."""
    assignments: Dict[str, int] = {}
    available_drones = dict(drone_positions)
    for local_idx, inc in enumerate(incidents):
        incident_pos = (float(inc["lat"]), float(inc["lon"]), float(inc.get("alt", 0.0)))
        if not available_drones:
            break
        best_drone = min(
            available_drones,
            key=lambda d: euclidean_distance(available_drones[d], incident_pos),
        )
        assignments[best_drone] = incident_indices[local_idx]
        del available_drones[best_drone]
    return assignments


def build_waypoint_payload(
    assignments: Dict[str, int], incidents: List[dict], default_alt: float = 10.0
) -> Dict[str, list]:
    """Formats the resolved assignment as the /droneN/waypoints payload
    shown in Section 6.2 of the paper."""
    payload: Dict[str, list] = {}
    for drone, incident_idx in assignments.items():
        inc = incidents[incident_idx]
        wp = [inc["lat"], inc["lon"], inc.get("alt", default_alt)]
        payload[f"{drone}/waypoints"] = [wp]
    return payload


def main():
    parser = argparse.ArgumentParser(description="Coordination Agent (Eq. 2 assignment solver)")
    parser.add_argument(
        "--incidents", type=str, required=True,
        help='JSON list of incidents, e.g. \'[{"lat":-165,"lon":-1.43,"alt":10}]\'',
    )
    parser.add_argument(
        "--drone-positions", type=str, default=None,
        help='JSON dict of current drone positions, e.g. \'{"/drone1":[0,0,0]}\'. '
             "Defaults to all drones at the origin.",
    )
    args = parser.parse_args()

    incidents = json.loads(args.incidents)
    if args.drone_positions:
        raw_positions = json.loads(args.drone_positions)
        drone_positions = {k: tuple(v) for k, v in raw_positions.items()}
    else:
        drone_positions = {ns: (0.0, 0.0, 0.0) for ns in DRONE_NAMESPACES}

    assignments = solve_assignment(drone_positions, incidents)
    payload = build_waypoint_payload(assignments, incidents)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
