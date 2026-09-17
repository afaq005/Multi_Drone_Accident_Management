#!/usr/bin/env python3
"""
planning_agent/llm_waypoint_node.py
====================================
Implements A_plan from Section 3.2 / 6.2 of the paper.

Subscribes to the free-form emergency-call transcript on
`/llm_waypoint_request` (std_msgs/String), calls the GPT-4o mini API with a
strongly constrained system prompt at temperature=0, and republishes the
per-incident coordinates as JSON on `/planning_agent/incidents`
(std_msgs/String) for the coordination agent to consume.

Runs standalone (`python llm_waypoint_node.py --transcript "..."`) for
testing without ROS, or as a ROS node when rospy is available.

Requires: OPENAI_API_KEY environment variable.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.config import (  # noqa: E402
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    TOPIC_TRANSCRIPT,
)
from model.utils import get_logger, load_json_safely, validate_planning_output  # noqa: E402

logger = get_logger("planning_agent")

PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "prompt_templates", "system_prompt.txt"
)


def load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def call_gpt4o_mini(transcript: str) -> dict:
    """Calls the OpenAI GPT-4o mini API with the constrained system prompt
    and returns the parsed JSON payload, or an empty-incidents payload on
    any parsing/schema failure (graceful exit per Section 6.2)."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. `pip install openai`.")
        return {"incidents": [], "intent": "error_no_openai_sdk"}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; planning agent exiting gracefully.")
        return {"incidents": [], "intent": "error_no_api_key"}

    client = OpenAI(api_key=api_key)
    system_prompt = load_system_prompt()

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
        raw_text = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        logger.error(f"OpenAI API call failed: {exc}")
        return {"incidents": [], "intent": "error_api_call_failed"}

    try:
        payload = load_json_safely(raw_text)
    except (json.JSONDecodeError, ValueError):
        logger.error(f"Planning agent received non-JSON output: {raw_text!r}")
        return {"incidents": [], "intent": "error_invalid_json"}

    if not validate_planning_output(payload):
        logger.error(f"Planning agent output failed schema validation: {payload}")
        return {"incidents": [], "intent": "error_schema_violation"}

    return payload


def process_transcript(transcript: str) -> dict:
    logger.info(f"Processing transcript: {transcript!r}")
    payload = call_gpt4o_mini(transcript)
    logger.info(f"Planning agent output: {json.dumps(payload)}")
    return payload


# --------------------------------------------------------------------------
# ROS integration (optional — falls back to CLI mode if rospy unavailable)
# --------------------------------------------------------------------------
def run_ros_node():
    import rospy
    from std_msgs.msg import String

    rospy.init_node("planning_agent_node", anonymous=False)
    pub = rospy.Publisher("/planning_agent/incidents", String, queue_size=10)

    def _callback(msg: "String"):
        payload = process_transcript(msg.data)
        pub.publish(String(data=json.dumps(payload)))

    rospy.Subscriber(TOPIC_TRANSCRIPT, String, _callback)
    logger.info(f"Planning agent listening on {TOPIC_TRANSCRIPT} ...")
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="Planning Agent (GPT-4o mini)")
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Run once with a transcript string instead of starting a ROS node.",
    )
    args = parser.parse_args()

    if args.transcript:
        payload = process_transcript(args.transcript)
        print(json.dumps(payload, indent=2))
        return

    try:
        run_ros_node()
    except ImportError:
        logger.warning("rospy not available. Falling back to interactive CLI mode.")
        logger.warning("Use --transcript \"...\" to run a single transcript, e.g.:")
        logger.warning(
            '  python llm_waypoint_node.py --transcript '
            '"Fly drone 1 to (-165, -1.43, 10), drone 2 to (102.2, 4.1, 10)"'
        )


if __name__ == "__main__":
    main()
