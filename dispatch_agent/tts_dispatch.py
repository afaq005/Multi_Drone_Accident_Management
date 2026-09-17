#!/usr/bin/env python3
"""
dispatch_agent/tts_dispatch.py
===============================
Implements A_disp's audio/MQTT delivery from Section 3.2 / 6.6: converts a
routed alert's text summary into audio using Piper TTS (a local, lightweight
VITS-style neural TTS engine [19]) and publishes the JSON payload + audio
reference to the emergency-services MQTT backbone.

Requires the `piper-tts` package and a downloaded voice model
(MDAM_PIPER_VOICE in model/config.py, default en_US-libritts-high.onnx).
See https://github.com/OHF-Voice/piper1-gpl for voice downloads.

Usage:
    python tts_dispatch.py --summary "Head-on collision, no visible fire." \
        --lat 37.33445 --lon -122.00898 --conf 0.87 --log-p-blip2 -0.4
"""
import argparse
import json
import os
import sys
import wave
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

logger = get_logger("dispatch_agent.tts")


def synthesize_audio(text: str, output_wav_path: str, voice_model: str = PIPER_VOICE_MODEL) -> Optional[str]:
    """Converts `text` to a WAV file via Piper TTS. Returns the output path,
    or None if Piper is unavailable (the alert is still published — audio
    generation degrades gracefully, matching the paper's emphasis on the
    JSON payload as the primary artifact)."""
    try:
        from piper import PiperVoice
    except ImportError:
        logger.warning(
            "piper-tts not installed (`pip install piper-tts`). Skipping audio "
            "synthesis; the JSON alert will still be dispatched."
        )
        return None

    if not os.path.exists(voice_model):
        logger.warning(
            f"Piper voice model not found at {voice_model}. Download one from "
            "https://github.com/OHF-Voice/piper1-gpl and set MDAM_PIPER_VOICE."
        )
        return None

    voice = PiperVoice.load(voice_model)
    with wave.open(output_wav_path, "wb") as wav_file:
        voice.synthesize(text, wav_file)
    logger.info(f"Synthesized dispatch audio -> {output_wav_path}")
    return output_wav_path


def publish_mqtt(payload: dict, host: str = MQTT_BROKER_HOST, port: int = MQTT_BROKER_PORT,
                  topic: str = MQTT_ALERT_TOPIC) -> bool:
    """Publishes the alert JSON to the emergency-services MQTT backbone
    (Section 6.6). Returns True on success."""
    try:
        import paho.mqtt.publish as mqtt_publish
    except ImportError:
        logger.warning(
            "paho-mqtt not installed (`pip install paho-mqtt`). Skipping MQTT "
            "publish; alert payload printed to stdout instead."
        )
        print(json.dumps(payload, indent=2))
        return False

    try:
        mqtt_publish.single(
            topic, payload=json.dumps(payload), hostname=host, port=port
        )
        logger.info(f"Published alert {payload.get('event_id')} to mqtt://{host}:{port}/{topic}")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"MQTT publish failed: {exc}")
        return False


def dispatch(incident_type: str, summary: str, lat: float, lon: float,
             conf_yolo: float, log_p_blip2: float, audio_dir: str = "dispatch_agent/audio_out") -> dict:
    """End-to-end dispatch: build payload -> route to nearest center (Eq. 25)
    -> synthesize audio -> publish over MQTT."""
    payload = build_alert_payload(incident_type, summary, lat, lon, conf_yolo, log_p_blip2)
    routed = route_alert(payload)

    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{routed['event_id']}.wav")
    synthesized = synthesize_audio(summary, audio_path)
    routed["audio_file"] = synthesized

    publish_mqtt(routed)
    return routed


def main():
    parser = argparse.ArgumentParser(description="Dispatch Agent (Piper TTS + MQTT)")
    parser.add_argument("--type", type=str, default="accident")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--log-p-blip2", type=float, default=-0.5)
    args = parser.parse_args()

    result = dispatch(args.type, args.summary, args.lat, args.lon, args.conf, args.log_p_blip2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
