#!/usr/bin/env python3
"""
planning_agent/llm_waypoint_node.py
====================================

Implements A_plan from Section 3.2 / 6.2.

The planning agent subscribes to a natural-language mission request on

    /llm_waypoint_request

and calls GPT-4o mini with the system prompt reproduced in
prompt_templates/system_prompt.txt.

For reproducibility, this implementation supports both:

1. The waypoint-map JSON format used in the original experiments:

       {
           "/drone1/waypoints": [[-165, -1.43, 10]],
           "/drone2/waypoints": [[102.2, 4.1, 10]]
       }

2. The normalized incident format used by the current modular pipeline:

       {
           "incidents": [
               {
                   "lat": -165,
                   "lon": -1.43,
                   "alt": 10,
                   "drone": "/drone1"
               }
           ],
           "intent": "..."
       }

Historical waypoint-map responses are converted internally into the
normalized incident representation before publication.

The resulting planning payload is augmented with:

    mission_id
    coordinate_mode
    incident_id

and published on:

    /planning_agent/incidents

for the coordination agent.

The full ROS pipeline evaluated in simulation uses local Cartesian
coordinates, so the default coordinate_mode is "local".

Standalone usage:

    python planning_agent/llm_waypoint_node.py \
        --transcript \
        "Fly drone1 to (-165, -1.43, 10), drone2 to (102.2, 4.1, 10)"

Requires:
    OPENAI_API_KEY
"""

import argparse
import json
import os
import re
import sys
import uuid

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from model.config import (  # noqa: E402
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    TOPIC_TRANSCRIPT,
)
from model.utils import (  # noqa: E402
    get_logger,
    load_json_safely,
    validate_planning_output,
)

logger = get_logger(
    "planning_agent"
)


PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompt_templates",
    "system_prompt.txt",
)


# Matches:
#   /drone1/waypoints
#   /drone2/waypoints
#   ...
WAYPOINT_KEY_PATTERN = re.compile(
    r"^/drone[^/]+/waypoints$"
)


def load_system_prompt() -> str:
    """Load the planning-agent system prompt from disk."""

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        prompt = f.read().strip()

    if not prompt:
        raise ValueError(
            f"Planning prompt is empty: "
            f"{PROMPT_PATH}"
        )

    return prompt


def _waypoint_mapping_to_incidents(
    payload: dict,
) -> dict:
    """
    Convert the waypoint-map format used in the original experiments
    into the normalized incident representation used by the current
    coordination agent.

    Example input:

        {
            "/drone1/waypoints": [[-165, -1.43, 10]],
            "/drone2/waypoints": [[102.2, 4.1, 10]]
        }

    Example output:

        {
            "incidents": [
                {
                    "lat": -165,
                    "lon": -1.43,
                    "alt": 10,
                    "drone": "/drone1"
                },
                ...
            ],
            "intent": "operator_directed_waypoints"
        }
    """

    incidents = []

    waypoint_keys = [
        key
        for key in payload
        if WAYPOINT_KEY_PATTERN.match(
            str(key)
        )
    ]

    if not waypoint_keys:
        raise ValueError(
            "Payload is neither an incident "
            "object nor a waypoint mapping."
        )

    for key in waypoint_keys:

        waypoints = payload[key]

        if not isinstance(
            waypoints,
            list,
        ):
            raise ValueError(
                f"{key} must contain a list "
                "of waypoints."
            )

        drone_namespace = key[
            :-len("/waypoints")
        ]

        for waypoint in waypoints:

            if (
                not isinstance(
                    waypoint,
                    (list, tuple),
                )
                or len(waypoint) != 3
            ):
                raise ValueError(
                    f"Invalid waypoint for "
                    f"{key}: {waypoint!r}. "
                    "Expected [x, y, z]."
                )

            x, y, z = waypoint

            try:
                x = float(x)
                y = float(y)
                z = float(z)

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    f"Waypoint for {key} "
                    "contains a non-numeric "
                    f"value: {waypoint!r}"
                ) from exc

            incidents.append(
                {
                    # These historical field names are retained because
                    # the current assignment solver uses lat/lon as its
                    # first two generic coordinate fields. In local mode
                    # they represent x/y.
                    "lat": x,
                    "lon": y,
                    "alt": z,
                    "drone": (
                        drone_namespace
                    ),
                    "coordinate_mode": (
                        "local"
                    ),
                }
            )

    return {
        "incidents": incidents,
        "intent": (
            "operator_directed_waypoints"
        ),
        "coordinate_mode": "local",
    }


def normalize_planning_payload(
    payload: dict,
) -> dict:
    """
    Normalize GPT output into the current incident schema.

    Supports:
        - current {"incidents": [...]} representation
        - historical /droneN/waypoints mapping
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Planning output must be "
            "a JSON object."
        )

    if "incidents" in payload:

        normalized = dict(
            payload
        )

    else:

        normalized = (
            _waypoint_mapping_to_incidents(
                payload
            )
        )

    if "intent" not in normalized:
        normalized[
            "intent"
        ] = "mission_request"

    if "coordinate_mode" not in normalized:
        normalized[
            "coordinate_mode"
        ] = "local"

    return normalized


def add_mission_metadata(
    payload: dict,
) -> dict:
    """
    Add stable mission and incident identifiers after GPT parsing.

    These identifiers are generated by software rather than the LLM so
    downstream agents do not depend on the model inventing consistent IDs.
    """

    enriched = dict(
        payload
    )

    mission_id = enriched.get(
        "mission_id"
    )

    if not mission_id:

        mission_id = (
            uuid.uuid4()
            .hex[:8]
            .upper()
        )

    enriched[
        "mission_id"
    ] = mission_id

    coordinate_mode = (
        enriched.get(
            "coordinate_mode",
            "local",
        )
    )

    incidents = []

    for index, incident in enumerate(
        enriched.get(
            "incidents",
            []
        )
    ):

        incident = dict(
            incident
        )

        if not incident.get(
            "incident_id"
        ):

            incident[
                "incident_id"
            ] = (
                f"{mission_id}-"
                f"I{index + 1:02d}"
            )

        incident[
            "coordinate_mode"
        ] = incident.get(
            "coordinate_mode",
            coordinate_mode,
        )

        incidents.append(
            incident
        )

    enriched[
        "incidents"
    ] = incidents

    return enriched


def _error_payload(
    intent: str,
) -> dict:
    """Create a consistent graceful-error planning payload."""

    return {
        "incidents": [],
        "intent": intent,
        "coordinate_mode": "local",
    }


def call_gpt4o_mini(
    transcript: str,
) -> dict:
    """
    Call GPT-4o mini using the planning-agent system prompt.

    The raw model response is parsed and normalized into the current
    incident representation.

    On API, JSON, or schema failure, an empty incident list is returned
    so that downstream nodes fail gracefully rather than receiving
    malformed commands.
    """

    try:
        from openai import OpenAI

    except ImportError:

        logger.error(
            "openai package not installed. "
            "Install it with "
            "`pip install openai`."
        )

        return _error_payload(
            "error_no_openai_sdk"
        )

    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:

        logger.error(
            "OPENAI_API_KEY not set; "
            "planning agent exiting "
            "gracefully."
        )

        return _error_payload(
            "error_no_api_key"
        )

    client = OpenAI(
        api_key=api_key
    )

    try:

        system_prompt = (
            load_system_prompt()
        )

    except Exception as exc:

        logger.error(
            f"Failed to load planning "
            f"prompt: {exc}"
        )

        return _error_payload(
            "error_prompt_unavailable"
        )

    try:

        response = (
            client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=(
                    OPENAI_TEMPERATURE
                ),
                max_tokens=(
                    OPENAI_MAX_TOKENS
                ),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            transcript
                        ),
                    },
                ],
            )
        )

        raw_text = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as exc:  # noqa: BLE001

        logger.error(
            f"OpenAI API call failed: "
            f"{exc}"
        )

        return _error_payload(
            "error_api_call_failed"
        )

    if not raw_text:

        logger.error(
            "Planning agent received "
            "an empty model response."
        )

        return _error_payload(
            "error_empty_response"
        )

    logger.debug(
        f"Raw GPT response: "
        f"{raw_text!r}"
    )

    try:

        raw_payload = (
            load_json_safely(
                raw_text
            )
        )

    except (
        json.JSONDecodeError,
        ValueError,
    ):

        logger.error(
            "Planning agent received "
            f"non-JSON output: "
            f"{raw_text!r}"
        )

        return _error_payload(
            "error_invalid_json"
        )

    try:

        payload = (
            normalize_planning_payload(
                raw_payload
            )
        )

    except ValueError as exc:

        logger.error(
            "Could not normalize "
            "planning output: "
            f"{exc}"
        )

        return _error_payload(
            "error_schema_violation"
        )

    # Validate the normalized representation rather than requiring the
    # historical prompt itself to emit the newer incident schema.
    if not validate_planning_output(
        payload
    ):

        logger.error(
            "Planning-agent output "
            "failed schema validation "
            f"after normalization: "
            f"{payload}"
        )

        return _error_payload(
            "error_schema_violation"
        )

    return payload


def process_transcript(
    transcript: str,
) -> dict:
    """
    Process one mission transcript and attach deterministic software-side
    mission metadata.
    """

    transcript = (
        transcript.strip()
    )

    if not transcript:

        logger.warning(
            "Received empty mission "
            "transcript."
        )

        return _error_payload(
            "error_empty_transcript"
        )

    logger.info(
        f"Processing transcript: "
        f"{transcript!r}"
    )

    payload = call_gpt4o_mini(
        transcript
    )

    # Do not create a mission ID for failed/empty planning output.
    if payload.get(
        "incidents"
    ):

        payload = add_mission_metadata(
            payload
        )

    logger.info(
        "Planning agent output: %s",
        json.dumps(
            payload
        ),
    )

    return payload


# --------------------------------------------------------------------------
# ROS integration
# --------------------------------------------------------------------------

def run_ros_node():

    import rospy
    from std_msgs.msg import String

    rospy.init_node(
        "planning_agent_node",
        anonymous=False,
    )

    pub = rospy.Publisher(
        "/planning_agent/incidents",
        String,
        queue_size=10,
    )

    def _callback(
        msg: "String",
    ):

        payload = process_transcript(
            msg.data
        )

        pub.publish(
            String(
                data=json.dumps(
                    payload
                )
            )
        )

    rospy.Subscriber(
        TOPIC_TRANSCRIPT,
        String,
        _callback,
        queue_size=10,
    )

    logger.info(
        f"Planning agent listening "
        f"on {TOPIC_TRANSCRIPT}"
    )

    logger.info(
        "Publishing normalized "
        "mission payloads on "
        "/planning_agent/incidents"
    )

    rospy.spin()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Planning Agent "
            "(GPT-4o mini)"
        )
    )

    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help=(
            "Run once with a mission "
            "transcript instead of "
            "starting a ROS node."
        ),
    )

    args = parser.parse_args()

    if args.transcript:

        payload = process_transcript(
            args.transcript
        )

        print(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return

    try:

        run_ros_node()

    except ImportError:

        logger.warning(
            "rospy not available."
        )

        logger.warning(
            "Use --transcript to run "
            "standalone, e.g.:"
        )

        logger.warning(
            "  python "
            "planning_agent/"
            "llm_waypoint_node.py "
            "--transcript "
            '"Fly drone1 to '
            '(-165, -1.43, 10), '
            'drone2 to '
            '(102.2, 4.1, 10)"'
        )


if __name__ == "__main__":
    main()
