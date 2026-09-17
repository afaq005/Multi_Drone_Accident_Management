"""
model/config.py
================
Central configuration shared by every agent (planning, coordination,
perception, description, dispatch, simulation). Keeping this in one place
means each agent script stays a thin wrapper around the paper's equations
instead of re-declaring constants.

All values can be overridden with environment variables so the same code
works locally, in Docker, and inside the Gazebo/ROS container without
editing source.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Fleet / mission configuration
# ---------------------------------------------------------------------------
NUM_DRONES = _env_int("MDAM_NUM_DRONES", 4)
DRONE_NAMESPACES = [f"/drone{i+1}" for i in range(NUM_DRONES)]

# Cruise / mission altitude (meters AGL), matches Section 6.3 (10 m AGL)
CRUISE_ALTITUDE_M = _env_float("MDAM_CRUISE_ALT", 10.0)

# 360-degree scan: yaw update period in seconds (Section 6.3)
YAW_SWEEP_PERIOD_S = _env_float("MDAM_YAW_PERIOD", 5.0)

# Waypoint publish rate (Hz) and repeat count to overcome packet loss (Section 6.2)
WAYPOINT_PUBLISH_HZ = _env_float("MDAM_WP_HZ", 2.0)
WAYPOINT_REPEAT_COUNT = _env_int("MDAM_WP_REPEATS", 4)

# ---------------------------------------------------------------------------
# Perception agent (YOLOv11n) — Section 3.2 / Eq. 21
# ---------------------------------------------------------------------------
YOLO_CLASSES = ["accident", "fire"]
YOLO_CONF_THRESHOLD = _env_float("MDAM_YOLO_CONF", 0.5)
YOLO_WEIGHTS_PATH = os.environ.get(
    "MDAM_YOLO_WEIGHTS", "perception_agent/weights/yolov11n_accident_fire.pt"
)
YOLO_IMG_SIZE = _env_int("MDAM_YOLO_IMGSZ", 640)

# Minimum drone displacement (meters) before a new detection event is
# persisted again at (roughly) the same location — Eq. 22.
DETECTION_DEDUP_DISTANCE_M = _env_float("MDAM_DEDUP_DIST", 10.0)
DETECTION_EVENT_WINDOW = _env_int("MDAM_EVENT_WINDOW", 3)  # rolling frame window

# ---------------------------------------------------------------------------
# Description agent (fine-tuned BLIP-2) — Section 6.5 / Eq. 23
# ---------------------------------------------------------------------------
BLIP2_BASE_MODEL = os.environ.get("MDAM_BLIP2_BASE", "Salesforce/blip2-opt-2.7b")
BLIP2_WEIGHTS_PATH = os.environ.get(
    "MDAM_BLIP2_WEIGHTS", "description_agent/weights/blip2_finetuned"
)
BLIP2_FINETUNE_EPOCHS = _env_int("MDAM_BLIP2_EPOCHS", 5)
BLIP2_MAX_NEW_TOKENS = _env_int("MDAM_BLIP2_MAXTOK", 64)

# ---------------------------------------------------------------------------
# Dispatch agent (Piper TTS + rescue-center routing) — Section 6.6 / Eq. 24-25
# ---------------------------------------------------------------------------
PIPER_VOICE_MODEL = os.environ.get(
    "MDAM_PIPER_VOICE", "en_US-libritts-high.onnx"
)
SEVERITY_ALPHA = _env_float("MDAM_SEVERITY_ALPHA", 1.0)  # alpha in Eq. 24
SEVERITY_BETA = _env_float("MDAM_SEVERITY_BETA", 1.0)    # beta in Eq. 24

MQTT_BROKER_HOST = os.environ.get("MDAM_MQTT_HOST", "localhost")
MQTT_BROKER_PORT = _env_int("MDAM_MQTT_PORT", 1883)
MQTT_ALERT_TOPIC = os.environ.get("MDAM_MQTT_TOPIC", "emergency/alerts")

# Rescue centers: (name, lat, lon). Replace with real coordinates for
# deployment; these four defaults roughly bracket the four Gazebo
# accident sites used in Fig. 2 / Section 6.1.
RESCUE_CENTERS: List[Tuple[str, float, float]] = [
    ("Rescue Center A", 37.3355, -122.0095),
    ("Rescue Center B", 37.3400, -122.0050),
    ("Rescue Center C", 37.3310, -122.0150),
    ("Rescue Center D", 37.3450, -122.0130),
]

# ---------------------------------------------------------------------------
# Planning agent (GPT-4o mini) — Section 6.2
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.environ.get("MDAM_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = _env_float("MDAM_OPENAI_TEMP", 0.0)
OPENAI_MAX_TOKENS = _env_int("MDAM_OPENAI_MAXTOK", 512)

# ROS topic names shared across nodes
TOPIC_TRANSCRIPT = "/llm_waypoint_request"
TOPIC_WAYPOINTS_FMT = "{ns}/waypoints"           # geometry_msgs/PoseArray
TOPIC_DETECTION_FMT = "{ns}/yolo_detection/detected"  # std_msgs/Bool
TOPIC_CAMERA_FMT = "{ns}_camera/image_raw"


@dataclass
class MissionState:
    """Lightweight, serializable mission state shared between agents.

    This mirrors the "shared mission state" block in Fig. 1 of the paper.
    It is intentionally plain-data (dict/list/float) so it can be published
    as JSON over ROS std_msgs/String or MQTT without a custom message type.
    """
    incidents: List[dict] = field(default_factory=list)   # [{"lat":.., "lon":.., "drone": Optional[str]}]
    assignments: dict = field(default_factory=dict)        # {drone_ns: incident_index}
    detections: dict = field(default_factory=dict)         # {drone_ns: [event dicts]}
    reports: List[dict] = field(default_factory=list)      # fused incident reports (Eq. 23/24)
