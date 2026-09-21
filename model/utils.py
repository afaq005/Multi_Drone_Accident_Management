"""
model/utils.py
===============
Small, dependency-light helpers reused by multiple agents: geodesic
distance for rescue-center routing (Eq. 25), a sigmoid for the severity
score (Eq. 24), and JSON-schema validation for planning-agent output
(Section 6.2).
"""
import json
import logging
import math
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon) points.

    Used by the dispatch agent (Eq. 25) to pick the nearest rescue center,
    and can also be used as an alternative cost metric for the coordination
    agent's assignment problem (Eq. 2) when drone/incident positions are
    given in geographic coordinates rather than a local ENU frame.
    """
    R = 6371000.0  # Earth radius, meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Euclidean distance for local ENU/NED drone<->incident coordinates,
    the cost metric c_kj used in Eq. 2 of the paper."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid, used for the severity score in Eq. 24."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_log(p: float, eps: float = 1e-9) -> float:
    """log() guarded against p <= 0, used for log p_BLIP2 in Eq. 24."""
    return math.log(max(p, eps))


PLANNING_JSON_SCHEMA_KEYS = {"incidents"}
INCIDENT_KEYS_REQUIRED = {"lat", "lon", "alt"}

def validate_planning_output(payload: Dict[str, Any]) -> bool:
    """
    Validate the normalized planning-agent output.

    Valid actionable payloads contain one or more incidents with numeric
    lat/lon/alt fields. In local mode these legacy field names represent
    Cartesian x/y/z coordinates.

    An empty incident list is also valid when the planner explicitly
    reports that insufficient waypoint information was provided.
    """

    if not isinstance(payload, dict):
        return False

    if "incidents" not in payload:
        return False

    incidents = payload["incidents"]

    if not isinstance(incidents, list):
        return False

    coordinate_mode = payload.get(
        "coordinate_mode",
        "local",
    )

    if coordinate_mode not in {
        "local",
        "gps",
    }:
        return False

    # Graceful non-actionable planning response.
    if len(incidents) == 0:
        return payload.get("intent") == (
            "insufficient_waypoint_information"
        )

    for inc in incidents:

        if not isinstance(inc, dict):
            return False

        if not INCIDENT_KEYS_REQUIRED.issubset(
            inc.keys()
        ):
            return False

        try:
            float(inc["lat"])
            float(inc["lon"])
            float(inc["alt"])

        except (TypeError, ValueError):
            return False

        incident_mode = inc.get(
            "coordinate_mode",
            coordinate_mode,
        )

        if incident_mode not in {
            "local",
            "gps",
        }:
            return False

        drone = inc.get("drone")

        if drone is not None:
            if (
                not isinstance(drone, str)
                or not drone.startswith("/drone")
            ):
                return False

    return True
# def validate_planning_output(payload: Dict[str, Any]) -> bool:
#     """Validate the planning agent's JSON output against the minimal schema
#     described in Section 6.2: a list of incidents, each with lat/lon, and an
#     optional explicit `drone` field when the operator pre-assigns drones.

#     Returns True if valid, False otherwise (callers should exit gracefully
#     rather than crash, per the paper: "If the JSON schema is violated ...
#     the node exits gracefully.").
#     """
#     if not isinstance(payload, dict):
#         return False
#     if not PLANNING_JSON_SCHEMA_KEYS.issubset(payload.keys()):
#         return False
#     incidents = payload["incidents"]
#     if not isinstance(incidents, list) or len(incidents) == 0:
#         return False
#     for inc in incidents:
#         if not isinstance(inc, dict):
#             return False
#         if not INCIDENT_KEYS_REQUIRED.issubset(inc.keys()):
#             return False
#         try:
#             float(inc["lat"])
#             float(inc["lon"])
#         except (TypeError, ValueError):
#             return False
#     return True

def load_json_safely(text: str):
    """Remove common Markdown code fences before parsing LLM JSON output."""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        # Remove opening fence such as ``` or ```json.
        if lines:
            lines = lines[1:]

        # Remove closing fence.
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return json.loads(cleaned)
# def load_json_safely(text: str):
#     """Strip common LLM wrapping (markdown fences) before parsing JSON."""
#     cleaned = text.strip()
#     if cleaned.startswith("```"):
#         cleaned = cleaned.strip("`")
#         if cleaned.lower().startswith("json"):
#             cleaned = cleaned[4:]
#     return json.loads(cleaned.strip())
