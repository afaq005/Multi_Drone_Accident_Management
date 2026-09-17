# Contributing

Thanks for your interest in improving the Multi-Drone Accident Management
system. This repo implements the agentic architecture from *"An Agentic
Multi-Drone Framework for Autonomous Multi-Accident Detection, Reasoning,
and Response."* Contributions that keep the code aligned with the paper's
five-agent decomposition (planning / coordination / perception /
description / dispatch) are especially welcome.

## Getting started

```bash
git clone https://github.com/afaq005/Multi_Drone_Accident_Management.git
cd Multi_Drone_Accident_Management
pip install -r requirements.txt   # or: conda env create -f environment.yml
pytest tests/
```

ROS/Gazebo/MAVROS components (`simulation/`) require a working ROS 1
(Noetic) + Gazebo 11 + ArduPilot SITL + MAVROS installation; see
`docs/SIMULATION_SETUP.md` if present, or the ArduPilot/MAVROS official
docs.

## Where things live

| Agent | Paper section | Code |
|---|---|---|
| Planning | §6.2 | `planning_agent/` |
| Coordination | §3.2, Eq. 2 | `coordination_agent/` |
| Perception | §6.4, Eq. 21-22 | `perception_agent/` |
| Description | §6.5, Eq. 18-20, 23 | `description_agent/` |
| Dispatch | §6.6, Eq. 24-25 | `dispatch_agent/` |
| Flight stack | §6.1, §6.3 | `simulation/` |

## Reporting issues

Please include: which agent/module, whether you're running the full
Gazebo pipeline or a standalone script, your Python/ROS versions, and
(if applicable) the exact command and traceback.

## Pull requests

1. Fork and branch from `main`.
2. Keep changes scoped to one agent/module where possible.
3. Add/update tests under `tests/` for any logic change (assignment
   solver, fusion, severity scoring, dataset augmentation, etc.).
4. Reference the relevant paper section/equation in your PR description
   if you're changing core algorithm behavior.

## Code style

Plain PEP 8, type hints where practical, no hard framework lock-in beyond
what's declared in `requirements.txt` — standalone/CLI mode should keep
working even without ROS installed, since several agents (assignment
solver, fusion, dispatch routing) are pure-Python and unit-testable on
their own.
