#!/usr/bin/env python3
"""
coordination_agent/assignment_solver.py
========================================

Implements A_coord from Section 3.2 / Eq. 2:

    min_X  sum_k sum_j c_kj * x_kj

    s.t.   sum_j x_kj = 1
           every incident receives exactly one drone

           sum_k x_kj <= 1
           every drone is assigned to at most one incident

           x_kj in {0, 1}

The assignment is solved with scipy.optimize.linear_sum_assignment.

Under the one-drone-per-incident and at-most-one-incident-per-drone
constraints, the formulation supports K <= N, where K is the number
of incidents and N is the number of available drones.

K > N is infeasible under these constraints and is therefore rejected
explicitly rather than silently leaving incidents unassigned.

The hybrid dispatch mode of Section 6.2 is also supported: when the
planning agent supplies an explicit `drone` field for an incident, that
mapping is honored and the remaining incidents are optimized over the
remaining drones.

Distance modes
--------------

coordinate_mode="local":
    Euclidean distance in a shared Cartesian coordinate frame.

coordinate_mode="gps":
    Haversine horizontal distance between latitude/longitude coordinates,
    combined with altitude difference when altitude is available.

All drone and incident coordinates passed to the solver must use the same
coordinate mode.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import DRONE_NAMESPACES  # noqa: E402
from model.utils import (  # noqa: E402
    euclidean_distance,
    get_logger,
    haversine_distance_m,
)

logger = get_logger(
    "coordination_agent"
)

try:
    import numpy as np
    from scipy.optimize import (
        linear_sum_assignment,
    )

    _HAVE_SCIPY = True

except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


def _resolve_coordinate_mode(
    incidents: List[dict],
    coordinate_mode: Optional[str] = None,
) -> str:
    """
    Resolve the coordinate mode used by the assignment problem.

    Explicit function arguments take precedence. Otherwise the mode is
    inferred from incident metadata, defaulting to "local".
    """

    if coordinate_mode is not None:

        if coordinate_mode not in {
            "local",
            "gps",
        }:
            raise ValueError(
                "coordinate_mode must be "
                "'local' or 'gps'."
            )

        return coordinate_mode

    modes = {
        incident.get(
            "coordinate_mode",
            "local",
        )
        for incident in incidents
    }

    if len(modes) > 1:
        raise ValueError(
            "All incidents in one assignment "
            "problem must use the same "
            "coordinate_mode."
        )

    mode = (
        next(iter(modes))
        if modes
        else "local"
    )

    if mode not in {
        "local",
        "gps",
    }:
        raise ValueError(
            f"Unsupported coordinate_mode: "
            f"{mode}"
        )

    return mode


def _incident_position(
    incident: dict,
) -> Tuple[float, float, float]:
    """
    Return the incident coordinates.

    For backward compatibility with the planning-agent schema, the first
    two coordinates are stored under `lat` and `lon` even when they
    represent local Cartesian x/y coordinates.
    """

    return (
        float(
            incident["lat"]
        ),
        float(
            incident["lon"]
        ),
        float(
            incident.get(
                "alt",
                0.0,
            )
        ),
    )


def _distance(
    drone_position: Tuple[
        float,
        float,
        float,
    ],
    incident_position: Tuple[
        float,
        float,
        float,
    ],
    coordinate_mode: str,
) -> float:
    """
    Compute the assignment cost between one drone and one incident.

    local:
        3-D Euclidean distance.

    gps:
        Haversine horizontal distance combined with altitude difference.
    """

    if coordinate_mode == "local":

        return euclidean_distance(
            drone_position,
            incident_position,
        )

    if coordinate_mode == "gps":

        drone_lat = float(
            drone_position[0]
        )

        drone_lon = float(
            drone_position[1]
        )

        drone_alt = float(
            drone_position[2]
        )

        incident_lat = float(
            incident_position[0]
        )

        incident_lon = float(
            incident_position[1]
        )

        incident_alt = float(
            incident_position[2]
        )

        horizontal_distance = (
            haversine_distance_m(
                drone_lat,
                drone_lon,
                incident_lat,
                incident_lon,
            )
        )

        altitude_difference = (
            incident_alt
            - drone_alt
        )

        return math.hypot(
            horizontal_distance,
            altitude_difference,
        )

    raise ValueError(
        f"Unsupported coordinate_mode: "
        f"{coordinate_mode}"
    )


def _build_cost_matrix(
    drone_positions: Dict[
        str,
        Tuple[
            float,
            float,
            float,
        ],
    ],
    incidents: List[dict],
    coordinate_mode: str = "local",
):
    """
    Build the K x N assignment-cost matrix.

    K may be smaller than N. scipy.optimize.linear_sum_assignment handles
    rectangular matrices directly, so no dummy padding is required.
    """

    if not _HAVE_SCIPY:
        raise RuntimeError(
            "NumPy/SciPy are required to "
            "build the Hungarian cost matrix."
        )

    drones = list(
        drone_positions.keys()
    )

    K = len(
        incidents
    )

    N = len(
        drones
    )

    cost = np.zeros(
        (
            K,
            N,
        ),
        dtype=float,
    )

    for k, incident in enumerate(
        incidents
    ):

        incident_pos = (
            _incident_position(
                incident
            )
        )

        for j, drone in enumerate(
            drones
        ):

            cost[
                k,
                j,
            ] = _distance(
                drone_positions[
                    drone
                ],
                incident_pos,
                coordinate_mode,
            )

    return (
        cost,
        drones,
    )


def solve_assignment(
    drone_positions: Dict[
        str,
        Tuple[
            float,
            float,
            float,
        ],
    ],
    incidents: List[dict],
    coordinate_mode: Optional[str] = None,
) -> Dict[str, int]:
    """
    Resolve drone -> incident-index allocation.

    Explicit operator-directed drone assignments are honored first.
    Remaining incidents are allocated optimally using the Hungarian
    algorithm.

    Parameters
    ----------
    drone_positions:
        Mapping from drone namespace to its current coordinates.

    incidents:
        Incident dictionaries produced by the planning agent.

    coordinate_mode:
        "local" for Cartesian coordinates or "gps" for geographic
        coordinates. If omitted, the mode is inferred from incident
        metadata and defaults to "local".

    Returns
    -------
    dict
        {drone_namespace: incident_index}
    """

    if not incidents:
        return {}

    if not drone_positions:
        raise ValueError(
            "No drones are available for "
            "incident assignment."
        )

    mode = _resolve_coordinate_mode(
        incidents,
        coordinate_mode,
    )

    K = len(
        incidents
    )

    N = len(
        drone_positions
    )

    # Eq. 2 requires every incident to receive exactly one drone and
    # prevents a drone from serving more than one incident.
    if K > N:
        raise ValueError(
            f"Assignment infeasible: "
            f"{K} incidents were provided "
            f"but only {N} drones are "
            f"available. Eq. 2 requires "
            "one distinct drone per incident."
        )

    assignments: Dict[
        str,
        int,
    ] = {}

    remaining_incidents: List[
        int
    ] = []

    remaining_drones = set(
        drone_positions.keys()
    )

    # ------------------------------------------------------------------
    # Step 1: honor explicit operator-directed assignments.
    # ------------------------------------------------------------------

    for idx, incident in enumerate(
        incidents
    ):

        drone = incident.get(
            "drone"
        )

        if not drone:

            remaining_incidents.append(
                idx
            )

            continue

        if drone not in drone_positions:

            raise ValueError(
                f"Incident {idx} requests "
                f"unknown drone '{drone}'."
            )

        if drone not in remaining_drones:

            raise ValueError(
                f"Drone '{drone}' was assigned "
                "to more than one incident. "
                "Eq. 2 allows each drone to "
                "serve at most one incident."
            )

        assignments[
            drone
        ] = idx

        remaining_drones.remove(
            drone
        )

    if not remaining_incidents:
        return assignments

    if len(
        remaining_incidents
    ) > len(
        remaining_drones
    ):
        raise ValueError(
            "Assignment infeasible after "
            "operator-directed assignments: "
            "there are more remaining "
            "incidents than available drones."
        )

    # ------------------------------------------------------------------
    # Step 2: optimize the remaining assignments.
    # ------------------------------------------------------------------

    sub_incidents = [
        incidents[i]
        for i in remaining_incidents
    ]

    sub_drone_positions = {
        drone: drone_positions[
            drone
        ]
        for drone in remaining_drones
    }

    if not _HAVE_SCIPY:

        logger.warning(
            "scipy not installed; falling "
            "back to greedy nearest-neighbor "
            "assignment instead of the "
            "optimal Hungarian solution."
        )

        assignments.update(
            _greedy_assignment(
                sub_drone_positions,
                sub_incidents,
                remaining_incidents,
                coordinate_mode=mode,
            )
        )

        return assignments

    cost, drones = (
        _build_cost_matrix(
            sub_drone_positions,
            sub_incidents,
            coordinate_mode=mode,
        )
    )

    row_ind, col_ind = (
        linear_sum_assignment(
            cost
        )
    )

    for row, col in zip(
        row_ind,
        col_ind,
    ):

        assignments[
            drones[col]
        ] = remaining_incidents[
            row
        ]

    return assignments


def _greedy_assignment(
    drone_positions: Dict[
        str,
        Tuple[
            float,
            float,
            float,
        ],
    ],
    incidents: List[dict],
    incident_indices: List[int],
    coordinate_mode: str = "local",
) -> Dict[str, int]:
    """
    Greedy fallback used only when SciPy is unavailable.

    This keeps the software runnable in minimal environments but is not
    equivalent to the globally optimal Hungarian solution.
    """

    assignments: Dict[
        str,
        int,
    ] = {}

    available_drones = dict(
        drone_positions
    )

    for local_idx, incident in enumerate(
        incidents
    ):

        if not available_drones:
            raise ValueError(
                "No remaining drone available "
                "for incident assignment."
            )

        incident_pos = (
            _incident_position(
                incident
            )
        )

        best_drone = min(
            available_drones,
            key=lambda drone: _distance(
                available_drones[
                    drone
                ],
                incident_pos,
                coordinate_mode,
            ),
        )

        assignments[
            best_drone
        ] = incident_indices[
            local_idx
        ]

        del available_drones[
            best_drone
        ]

    return assignments


def build_waypoint_payload(
    assignments: Dict[
        str,
        int,
    ],
    incidents: List[dict],
    default_alt: float = 10.0,
) -> Dict[str, list]:
    """
    Format the resolved assignments as /droneN/waypoints payloads.

    The existing waypoint representation is retained for backward
    compatibility with the coordination ROS node.
    """

    payload: Dict[
        str,
        list,
    ] = {}

    for (
        drone,
        incident_idx,
    ) in assignments.items():

        incident = incidents[
            incident_idx
        ]

        waypoint = [
            float(
                incident["lat"]
            ),
            float(
                incident["lon"]
            ),
            float(
                incident.get(
                    "alt",
                    default_alt,
                )
            ),
        ]

        payload[
            f"{drone}/waypoints"
        ] = [
            waypoint
        ]

    return payload


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Coordination Agent "
            "(Eq. 2 assignment solver)"
        )
    )

    parser.add_argument(
        "--incidents",
        type=str,
        required=True,
        help=(
            "JSON list of incidents, e.g. "
            '\'[{"lat":-165,'
            '"lon":-1.43,'
            '"alt":10}]\''
        ),
    )

    parser.add_argument(
        "--drone-positions",
        type=str,
        default=None,
        help=(
            "JSON dict of drone positions, "
            "e.g. "
            "'{\"/drone1\":[0,0,0]}'. "
            "Defaults to all drones at "
            "the origin."
        ),
    )

    parser.add_argument(
        "--coordinate-mode",
        choices=[
            "local",
            "gps",
        ],
        default="local",
        help=(
            "Coordinate system used by both "
            "incident and drone positions."
        ),
    )

    args = parser.parse_args()

    incidents = json.loads(
        args.incidents
    )

    if args.drone_positions:

        raw_positions = json.loads(
            args.drone_positions
        )

        drone_positions = {
            key: tuple(
                value
            )
            for (
                key,
                value,
            ) in raw_positions.items()
        }

    else:

        drone_positions = {
            namespace: (
                0.0,
                0.0,
                0.0,
            )
            for namespace in (
                DRONE_NAMESPACES
            )
        }

    assignments = solve_assignment(
        drone_positions,
        incidents,
        coordinate_mode=(
            args.coordinate_mode
        ),
    )

    payload = build_waypoint_payload(
        assignments,
        incidents,
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
