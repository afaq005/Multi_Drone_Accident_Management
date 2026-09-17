#!/usr/bin/env bash
#
# simulation/launch/start_sitl_fleet.sh
# ======================================
# Spins up 4 ArduCopter SITL instances against the Gazebo world
# (four_accident_sites.world) and bridges each with MAVProxy, matching the
# excerpt in Section 6.1:
#
#   sim_vehicle.py --no-mavproxy -v ArduCopter -f gazebo-drone1 -I0 \
#       --out=udp:127.0.0.1:14551 &
#   mavproxy.py --master=tcp:127.0.0.1:5760 --out=udp:127.0.0.1:14551
#
# Requires: ArduPilot SITL (sim_vehicle.py), MAVProxy, and a
# gazebo-drone{1..4} frame definition (ArduPilot's Tools/autotest/pysim
# Gazebo plugin, or the ardupilot_gazebo plugin).
#
# Usage: ./start_sitl_fleet.sh
set -euo pipefail

NUM_DRONES=${MDAM_NUM_DRONES:-4}
BASE_MAVLINK_UDP_PORT=14550
BASE_MAVPROXY_TCP_PORT=5760

echo "[start_sitl_fleet] Launching $NUM_DRONES ArduCopter SITL instances..."

for i in $(seq 0 $((NUM_DRONES - 1))); do
  instance=$i
  udp_out_port=$((BASE_MAVLINK_UDP_PORT + instance))
  echo "  -> Drone $((instance + 1)): instance -I${instance}, out=udp:127.0.0.1:${udp_out_port}"

  sim_vehicle.py \
    --no-mavproxy \
    -v ArduCopter \
    -f "gazebo-drone$((instance + 1))" \
    -I"${instance}" \
    --out="udp:127.0.0.1:${udp_out_port}" \
    > "/tmp/sitl_drone$((instance + 1)).log" 2>&1 &

  sleep 2
done

echo "[start_sitl_fleet] All SITL instances launched. Logs in /tmp/sitl_droneN.log"
echo "[start_sitl_fleet] Bring up MAVROS bridges next: roslaunch simulation/launch/mavros_bridge.launch"
wait
