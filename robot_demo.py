#!/usr/bin/env python3
"""
ENIMBLE - embodied loop on a Unitree robot (or the unitree_mujoco simulator)
============================================================================

Runs the SAME mechanism as `embodied_demo.py`, but grounds the model in the
ROBOT's real body (motor load/temperature, battery, motion) instead of a laptop's
CPU. Because unitree_mujoco speaks the same LowState messages as the hardware,
this script is identical for sim and for the real robot.

  1) start the simulator (or connect the robot):
        # sim:  in unitree_mujoco/simulate_python  ->  python unitree_mujoco.py
  2) run:
        python robot_demo.py

TEMPLATE: see robot_adapter.py for the field-name / normalisation notes.
"""
import time
import numpy as np

import embodied_demo as ed          # reuse the exact same tiny model
from robot_adapter import RobotBody


def main():
    rb = RobotBody()
    print("=" * 66)
    print(" ENIMBLE - embodied loop on the robot's real body")
    print("=" * 66)
    print("Reading the robot's body (move it / vary load to see the loop respond)\n")

    strains, wired, cut = [], [], []
    for i in range(40):
        body = rb.read_body()
        s  = ed.strain_of(body)
        bw = ed.think_budget(body, wire=True)      # grounded in the body
        bc = ed.think_budget(body, wire=False)     # wire cut (control)
        strains.append(s); wired.append(bw); cut.append(bc)
        mood = "strained -> think less / conserve" if s > 0.5 else "fresh -> think deeper"
        print(f"  [{i:02d}] strain={s:4.2f}  think_budget={bw:2d}/{ed.MAX_STEPS}  ({mood})")
        time.sleep(0.4)

    corr = lambda y: float(np.corrcoef(strains, y)[0, 1]) if np.std(y) > 1e-9 else 0.0
    print("\n" + "-" * 66)
    print(f"  over the run:  wired corr(strain, think_budget) = {corr(wired):+.2f}")
    print(f"                 cut-the-wire                      = {corr(cut):+.2f}")
    print("-" * 66)
    print("  The robot's real body drives how much the model thinks; cut the wire")
    print("  and it goes flat. Same mechanism as the laptop demo, now on the robot.")
    print("  (Vary the robot's state during the run so the body actually changes.)\n")


if __name__ == "__main__":
    main()
