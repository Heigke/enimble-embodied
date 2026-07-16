#!/usr/bin/env python3
"""
ENIMBLE - Unitree robot body adapter  (TEMPLATE)
================================================

Reads the robot's REAL body from the Unitree SDK `LowState` and returns the same
4-vector the embodied demo expects: [motor_load, motor_temp, battery_strain, motion].

The important part: this works **identically in the unitree_mujoco SIMULATOR and on
the real robot**, because both speak the same `LowState` DDS messages. So you can
validate the whole embodied loop on a normal machine (sim) before touching hardware.

  sim:   run unitree_mujoco (simulate_python)  ->  python robot_demo.py
  robot: point it at the robot's network interface  ->  python robot_demo.py

--------------------------------------------------------------------------------
TEMPLATE NOTE: field names follow the documented `unitree_sdk2_python` LowState.
Please verify them against your SDK version - anything missing degrades to 0.0
rather than crashing. Normalisation constants below are rough; tune to taste.
--------------------------------------------------------------------------------

Deps (only for the robot/sim path):  pip install -r requirements-robot.txt
Docs: https://github.com/unitreerobotics/unitree_sdk2_python
      https://github.com/unitreerobotics/unitree_mujoco
"""
import time
import numpy as np

# Go2 / B2 / H1 / Go2w  -> "go2" (unitree_go IDL)
# G1 / H1-2             -> "g1"  (unitree_hg IDL)
ROBOT = "go2"

# --- rough normalisation ranges (tune per platform) ---
TAU_MAX  = 30.0     # |estimated joint torque| that counts as "fully loaded" [N·m]
TEMP_MIN = 25.0
TEMP_MAX = 90.0     # motor temperature °C -> 0..1
GYRO_MAX = 4.0      # |angular velocity| that counts as "fully moving" [rad/s]


def _import_lowstate(robot):
    if robot in ("g1", "h1_2", "hg"):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_   # G1 / H1-2
    else:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_   # Go2 / B2 / H1
    return LowState_


class RobotBody:
    """Live 4-vector of the robot's physical state, each component in 0..1."""

    def __init__(self, robot=ROBOT, net_interface="", domain_id=0):
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        # Real robot: ChannelFactoryInitialize(0, "eth0")  (its network interface)
        # unitree_mujoco sim: use the domain id / interface from the sim's config.yaml
        if net_interface:
            ChannelFactoryInitialize(domain_id, net_interface)
        else:
            ChannelFactoryInitialize(domain_id)
        LowState_ = _import_lowstate(robot)
        self._latest = None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_msg, 10)

    def _on_msg(self, msg):
        self._latest = msg

    def read_body(self, timeout=2.0):
        t0 = time.time()
        while self._latest is None and (time.time() - t0) < timeout:
            time.sleep(0.02)
        s = self._latest
        if s is None:
            return np.zeros(4)                      # no telemetry yet

        motors = list(getattr(s, "motor_state", []) or [])
        taus  = [abs(float(getattr(m, "tau_est", 0.0)))     for m in motors]
        temps = [float(getattr(m, "temperature", TEMP_MIN)) for m in motors]

        motor_load = np.clip(np.mean(taus) / TAU_MAX, 0, 1) if taus else 0.0
        motor_temp = np.clip((np.mean(temps) - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1) if temps else 0.0

        bms = getattr(s, "bms_state", None)
        soc = float(getattr(bms, "soc", 100.0)) if bms is not None else 100.0
        battery_strain = np.clip(1.0 - soc / 100.0, 0, 1)   # low charge -> high strain

        imu  = getattr(s, "imu_state", None)
        gyro = list(getattr(imu, "gyroscope", [0.0, 0.0, 0.0])) if imu is not None else [0.0, 0.0, 0.0]
        motion = np.clip(np.linalg.norm(gyro) / GYRO_MAX, 0, 1)

        return np.array([motor_load, motor_temp, battery_strain, motion], dtype=float)


if __name__ == "__main__":
    # Quick check: print the robot's live body (run the sim or connect the robot first).
    names = ["motor_load", "motor_temp", "battery   ", "motion    "]
    body = RobotBody()
    print(f"Streaming {ROBOT} body (Ctrl-C to stop)...\n")
    try:
        while True:
            b = body.read_body()
            row = "  ".join(f"{n} {v:4.2f}" for n, v in zip(names, b))
            print(row)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
