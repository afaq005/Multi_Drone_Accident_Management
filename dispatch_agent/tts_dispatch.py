#!/usr/bin/env python3
"""
dispatch_agent/tts_dispatch.py
===============================

Implements A_disp audio/MQTT delivery.

Builds and routes an alert, synthesizes its summary with Piper TTS,
and publishes the resulting JSON payload to the configured MQTT broker.

GPS mode may use the configured default geographic rescue centers.

Local Cartesian mode requires explicit rescue-center coordinates defined
in the same coordinate frame as the incident.

GPS example:

    python dispatch_agent/tts_dispatch.py \
        --summary "Head-on collision, no visible fire." \
        --coordinate-mode gps \
        --lat 37.33445 \
        --lon -122.00898 \
        --conf 0.87 \
        --log-p-blip2 -0.4

Local example:

    python dispatch_agent/tts_dispatch.py \
        --summary "Head-on collision." \
        --coordinate-mode local \
        --x -165 \
        --y -1.43 \
        --centers-json rescue_centers.json \
        --conf 0.87 \
        --log-p-blip2 -0.4
"""

import argparse
import json
import os
import sys
import wave
from typing import List, Optional, Tuple

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    MQTT_ALERT_TOPIC,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    PIPER_VOICE_MODEL,
)
from model.utils import get_logger  # noqa: E402
from dispatch_agent.rescue_center_router import (  # noqa: E402
    build_alert_payload,
    route_alert,
)

logger = get_logger(
    "dispatch_agent.tts"
)


def synthesize_audio(
    text: str,
    output_wav_path: str,
    voice_model: str = PIPER_VOICE_MODEL,
) -> Optional[str]:
    """
    Convert text to WAV using Piper.

    Returns None if Piper or its voice model is unavailable.
    """

    try:
        from piper import PiperVoice
    except ImportError:
        logger.warning(
            "piper-tts not installed. "
            "Skipping audio synthesis."
        )
        return None

    if not os.path.exists(
        voice_model
    ):
        logger.warning(
            f"Piper voice model not found at {voice_model}. "
            "Set MDAM_PIPER_VOICE to a valid model."
        )
        return None

    voice = PiperVoice.load(
        voice_model
    )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                output_wav_path
            )
        ),
        exist_ok=True,
    )

    with wave.open(
        output_wav_path,
        "wb",
    ) as wav_file:
        voice.synthesize(
            text,
            wav_file,
        )

    logger.info(
        f"Synthesized dispatch audio -> {output_wav_path}"
    )

    return output_wav_path


def publish_mqtt(
    payload: dict,
    host: str = MQTT_BROKER_HOST,
    port: int = MQTT_BROKER_PORT,
    topic: str = MQTT_ALERT_TOPIC,
) -> bool:
    """Publish alert JSON to the configured MQTT broker."""

    try:
        import paho.mqtt.publish as mqtt_publish
    except ImportError:
        logger.warning(
            "paho-mqtt not installed. "
            "Printing payload instead."
        )

        print(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return False

    try:
        mqtt_publish.single(
            topic,
            payload=json.dumps(
                payload
            ),
            hostname=host,
            port=port,
        )

        logger.info(
            f"Published alert "
            f"{payload.get('event_id')} "
            f"to mqtt://{host}:{port}/{topic}"
        )

        return True

    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"MQTT publish failed: {exc}"
        )
        return False


def load_centers_json(
    path: str,
) -> List[
    Tuple[
        str,
        float,
        float,
    ]
]:
    """Load rescue-center coordinates from JSON."""

    if not os.path.isfile(
        path
    ):
        raise FileNotFoundError(
            f"Rescue-center file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(
            f
        )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "Rescue-center JSON must contain a list."
        )

    centers = []

    for item in raw:
        if (
            not isinstance(
                item,
                (list, tuple),
            )
            or len(item) != 3
        ):
            raise ValueError(
                "Each rescue center must have the form "
                "[name, coordinate_a, coordinate_b]."
            )

        name, a, b = item

        centers.append(
            (
                str(name),
                float(a),
                float(b),
            )
        )

    if not centers:
        raise ValueError(
            "At least one rescue center is required."
        )

    return centers


def dispatch(
    incident_type: str,
    summary: str,
    coord_a: float,
    coord_b: float,
    conf_yolo: float,
    log_p_blip2: float,
    coordinate_mode: str = "gps",
    centers: Optional[
        List[
            Tuple[
                str,
                float,
                float,
            ]
        ]
    ] = None,
    audio_dir: str = "dispatch_agent/audio_out",
) -> dict:
    """
    Build payload -> route -> synthesize audio -> MQTT.

    `route_alert` itself supplies configured default centers only for
    GPS mode. Local mode therefore requires explicit centers.
    """

    payload = build_alert_payload(
        incident_type=incident_type,
        summary=summary,
        coord_a=coord_a,
        coord_b=coord_b,
        conf_yolo=conf_yolo,
        log_p_blip2=log_p_blip2,
        coordinate_mode=coordinate_mode,
    )

    # Do NOT substitute RESCUE_CENTERS here.
    # route_alert safely defaults them only for GPS mode.
    routed = route_alert(
        payload,
        centers=centers,
    )

    os.makedirs(
        audio_dir,
        exist_ok=True,
    )

    audio_path = os.path.join(
        audio_dir,
        f"{routed['event_id']}.wav",
    )

    routed["audio_file"] = (
        synthesize_audio(
            summary,
            audio_path,
        )
    )

    publish_mqtt(
        routed
    )

    return routed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch Agent "
            "(Piper TTS + MQTT)"
        )
    )

    parser.add_argument(
        "--type",
        type=str,
        default="accident",
    )

    parser.add_argument(
        "--summary",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--coordinate-mode",
        choices=[
            "gps",
            "local",
        ],
        default="gps",
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--x",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--y",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--centers-json",
        type=str,
        default=None,
        help=(
            "Optional in GPS mode; "
            "required in local mode."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--log-p-blip2",
        type=float,
        default=-0.5,
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default="dispatch_agent/audio_out",
    )

    args = parser.parse_args()

    if args.coordinate_mode == "gps":
        if (
            args.lat is None
            or args.lon is None
        ):
            parser.error(
                "--lat and --lon are required for GPS mode."
            )

        coord_a = args.lat
        coord_b = args.lon

        centers = (
            load_centers_json(
                args.centers_json
            )
            if args.centers_json
            else None
        )

    else:
        if (
            args.x is None
            or args.y is None
        ):
            parser.error(
                "--x and --y are required for local mode."
            )

        if not args.centers_json:
            parser.error(
                "--centers-json is required for local mode."
            )

        coord_a = args.x
        coord_b = args.y

        centers = load_centers_json(
            args.centers_json
        )

    result = dispatch(
        incident_type=args.type,
        summary=args.summary,
        coord_a=coord_a,
        coord_b=coord_b,
        conf_yolo=args.conf,
        log_p_blip2=args.log_p_blip2,
        coordinate_mode=args.coordinate_mode,
        centers=centers,
        audio_dir=args.audio_dir,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
