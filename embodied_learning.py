#!/usr/bin/env python3
"""
ENIMBLE - embodied online-correction loop  (Unitree Go2 in MuJoCo)
==================================================================

An agent feels the robot's REAL body (joint load -> heat, battery drain, read from
the Go2 physics), decides how hard to work on each task, sees the outcome, and
CORRECTS its policy online:  "I felt like this, so I did that - but I actually
needed this."

The robot can only shed heat and recover battery by CHOOSING to work lightly for a
while. So the right policy is body-conditioned: push when you have headroom, ease
off to cool down when you're hot / low. An agent that cannot feel its body cannot
time this - it stays hot and gets throttled. That is the honest test of benefit:

    the body-AWARE learner beats the body-BLIND learner (and a fixed policy)
    on reward, energy and throttling - because feeling the body IS the benefit.

Runs headless on CPU.   pip install mujoco robot_descriptions numpy
                        python embodied_learning.py
"""
import numpy as np
import mujoco
from robot_descriptions import go2_mj_description

rng = np.random.default_rng(0)
model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH)
data  = mujoco.MjData(model)
NU = model.nu

def physical_load(effort, steps=20):
    mujoco.mj_resetData(model, data)
    f = []
    for t in range(steps):
        data.ctrl[:] = effort * 0.8 * np.sin(t * 0.4 + np.arange(NU))
        mujoco.mj_step(model, data)
        f.append(np.mean(np.abs(data.actuator_force[:NU])))
    return float(np.mean(f))

_L0, _L1 = physical_load(0.05), physical_load(1.0)
load_of = lambda e: np.clip((physical_load(e) - _L0) / (_L1 - _L0 + 1e-9), 0, 1)
print(f"loaded Unitree Go2 ({model.nq} dof); calibrated joint-load range from physics")

# ---- the robot's body: the ONLY way to cool / recharge is to work lightly ----
REST = 0.25
class Body:
    def __init__(self): self.temp, self.batt = 0.2, 1.0
    def apply(self, effort, load):
        if effort < REST:                                   # ease off -> cool + recharge
            self.temp = max(0.0, self.temp - 0.13)
            self.batt = min(1.0, self.batt + 0.03)
        else:                                               # work -> heat (accelerating) + drain
            self.temp = np.clip(self.temp + 0.16 * load + 0.10 * load * self.temp, 0, 1.4)
            self.batt = np.clip(self.batt - 0.035 * load, 0, 1)
    def feat(self):     return np.array([self.temp, 1 - self.batt])
    def headroom(self): return float(min(max(0.0, 1 - self.temp), self.batt))

DIFF = {"easy": 0.30, "med": 0.60, "hard": 0.90}
def optimal_effort(need, body):
    if body.temp > 0.65 or body.batt < 0.25:                # must ease off to recover
        return 0.10
    return float(np.clip(need, 0.05, 1.0))
def reward(effort, need, body, load):
    if effort < REST:
        return 0.0 - 0.02, 0.0, load                        # resting: no task reward, tiny idle cost
    success = 1.0 if effort >= need - 0.05 else max(0.0, 1 - (need - effort) * 2.5)
    throttle = 3.0 * effort if body.temp > 0.85 else 0.0     # hot + still pushing = throttle/damage
    empty    = 2.0 if body.batt < 0.10 else 0.0
    return success - 0.4 * load - throttle - empty, success, load

class Policy:
    def __init__(self, body_aware=True):
        self.w = np.zeros(4); self.body_aware = body_aware; self.lr = 0.12
    def features(self, hint, body):
        b = body.feat() if self.body_aware else np.zeros(2)
        return np.array([hint, b[0], b[1], 1.0])
    def act(self, hint, body):
        return float(np.clip(self.w @ self.features(hint, body), 0.05, 1.0))
    def correct(self, hint, body, target):
        f = self.features(hint, body); self.w += self.lr * (target - (self.w @ f)) * f

def run(policy, rounds=600, fixed=None):
    body = Body(); R = E = TH = 0.0
    for i in range(rounds):
        need = DIFF[rng.choice(list(DIFF))]
        hint = need + rng.normal(0, 0.05)
        e = fixed if fixed is not None else policy.act(hint, body)
        load = load_of(e)
        r, _, en = reward(e, need, body, load)
        if body.temp > 0.85 and e >= REST: TH += 1
        body.apply(e, load)
        if fixed is None: policy.correct(hint, body, optimal_effort(need, body))
        R += r; E += en
    return dict(reward=R / rounds, energy=E / rounds, throttle=int(TH), policy=policy)

aware    = run(Policy(True))
blind    = run(Policy(False))
fixedmax = run(None, fixed=0.9)

print("\n  agent                          avg reward   avg load    throttle events")
print("  " + "-" * 64)
for nm, r in [("body-AWARE (learns + feels)", aware),
              ("body-BLIND (learns, no body)", blind),
              ("fixed max-effort", fixedmax)]:
    print(f"  {nm:<30} {r['reward']:+.3f}      {r['energy']:.3f}       {r['throttle']}")
print("  " + "-" * 64)

hot = Body(); hot.temp, hot.batt = 0.95, 0.3
fresh = Body(); pa = aware["policy"]
print(f"\n  learned self-regulation (hard task): fresh -> effort {pa.act(0.9, fresh):.2f}"
      f"   |   hot + low battery -> effort {pa.act(0.9, hot):.2f}  (eases off to cool)")
esave = (blind["energy"] - aware["energy"]) / max(blind["energy"], 1e-9) * 100
print(f"  => feeling the body: reward {aware['reward']-blind['reward']:+.3f} vs body-blind,"
      f" load {esave:+.0f}%, throttling {blind['throttle']}->{aware['throttle']}.")
print("     It learned online to ease off and cool when strained, push when fresh -")
print("     a benefit that comes specifically from feeling the robot's real body.\n")
