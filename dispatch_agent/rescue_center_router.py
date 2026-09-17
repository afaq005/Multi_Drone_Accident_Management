#!/usr/bin/env python3
"""
dispatch_agent/rescue_center_router.py
=======================================
Implements A_disp's routing logic from Section 6.6:

  Eq. 24 — severity score:
      severity = sigmoid(alpha * conf_YOLO + beta * log p_BLIP2(y_hat | I))

  Eq. 25 — nearest-rescue-center selection:
      r* = argmin_r || g_k - l_r ||
      (distributes alerts across centers rather than concentrating on one)

Also builds the minimal JSON alert payload shown in Section 6.6.
"""
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import RESCUE_CENTERS, SEVERITY_ALPHA, SEVERITY_BETA  # noqa: E402
from model.utils import get_logger, haversine_distance_m, sigmoid  # noqa: E402

logger = get_logger("dispatch_agent.router")


def compute_severity(conf_yolo: float, log_p_blip2: float,
                      alpha: float = SEVERITY_ALPHA, beta: float = SEVERITY_BETA) -> float:
    """Eq. 24."""
    return sigmoid(alpha * conf_yolo + beta * log_p_blip2)


def nearest_rescue_center(
    incident_lat: float, incident_lon: float,
    centers: List[Tuple[str, float, float]] = RESCUE_CENTERS,
) -> Tuple[str, float]:
    """Eq. 25: r* = argmin_r || g_k - l_r ||. Returns (center_name, distance_m)."""
    best_name, best_dist = None, float("inf")
    for name, lat, lon in centers:
        dist = haversine_distance_m(incident_lat, incident_lon, lat, lon)
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name, best_dist


def build_alert_payload(
    incident_type: str,
    summary: str,
    lat: float,
    lon: float,
    conf_yolo: float,
    log_p_blip2: float,
    image_paths: Optional[List[str]] = None,
    event_id: Optional[str] = None,
) -> dict:
    """Builds the minimal JSON payload shown in Section 6.6."""
    severity = compute_severity(conf_yolo, log_p_blip2)
    return {
        "event_id": event_id or uuid.uuid4().hex[:8].upper(),
        "type": incident_type,
        "summary": summary,
        "lat": lat,
        "lon": lon,
        "severity": round(severity, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "images": image_paths or [],
    }


def route_alert(payload: dict) -> dict:
    """Attaches the chosen rescue center (Eq. 25) to an alert payload."""
    center_name, distance_m = nearest_rescue_center(payload["lat"], payload["lon"])
    payload = dict(payload)
    payload["rescue_center"] = center_name
    payload["distance_to_center_m"] = round(distance_m, 1)
    logger.info(
        f"Routed alert {payload['event_id']} (severity={payload['severity']}) "
        f"to {center_name} ({distance_m:.0f} m away)"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Dispatch Agent routing (Eq. 24-25)")
    parser.add_argument("--type", type=str, default="accident")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--conf", type=float, default=0.85, help="YOLO detection confidence")
    parser.add_argument("--log-p-blip2", type=float, default=-0.5,
                         help="BLIP-2 caption log-likelihood")
    args = parser.parse_args()

    payload = build_alert_payload(
        args.type, args.summary, args.lat, args.lon, args.conf, args.log_p_blip2
    )
    routed = route_alert(payload)
    print(json.dumps(routed, indent=2))


if __name__ == "__main__":
    main()
