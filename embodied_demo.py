#!/usr/bin/env python3
"""
ENIMBLE - Embodied AI : minimal runnable demo
=============================================

Grounds a small model's *compute-allocation* decision in the machine's OWN real
hardware state, and verifies the coupling by intervention ("cut the wire").

  the real body (CPU load / temp / memory / jitter)
        -> wired into the model's internal SELF-STATE
              -> which sets how long the model "thinks"
                    -> verified: cut the wire and the decision goes flat.

This is a MINIMAL, self-contained illustration of the *mechanism*. It runs on any
normal computer (CPU, numpy - no GPU, no model download). It is deliberately NOT
the full research system: the verified results use large (7B-32B) LLMs and
mechanistic-interpretability tooling on GPUs. See README.md -> "Hardware requirements".

Run:  python embodied_demo.py
"""

import numpy as np

try:
    import psutil
except ImportError:
    raise SystemExit("Missing dependency. Run:  pip install -r requirements.txt")

rng = np.random.default_rng(0)

BODY_DIM  = 4     # [cpu_load, cpu_temp, memory, load_jitter]
H         = 16    # dimension of the model's internal self-state
MAX_STEPS = 12    # maximum "thinking" budget the model can choose
STRAIN_W  = np.array([0.50, 0.30, 0.15, 0.05])   # how much each signal counts toward "strain"

# --------------------------------------------------------------------------
#  1. read the machine's REAL body
# --------------------------------------------------------------------------
def _cpu_temp_norm():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "acpitz", "cpu_thermal", "zenpower"):
            if temps.get(key):
                cur = np.mean([s.current for s in temps[key] if s.current])
                return float(np.clip((cur - 30.0) / 60.0, 0, 1))   # ~30-90C -> 0..1
    except Exception:
        pass
    return float("nan")   # not exposed on this machine

def read_body():
    """A live 4-vector of this computer's physical state, each component in 0..1."""
    load = psutil.cpu_percent(interval=0.2) / 100.0
    mem  = psutil.virtual_memory().percent / 100.0
    temp = _cpu_temp_norm()
    if np.isnan(temp):
        temp = load          # fall back to a load proxy if no thermal sensor
    percore = psutil.cpu_percent(percpu=True)
    jitter  = (np.std(percore) / 100.0) if percore else 0.0
    return np.array([load, temp, mem, jitter], dtype=float)

def strain_of(body):
    return float(np.clip(STRAIN_W @ body, 0, 1))

# --------------------------------------------------------------------------
#  2. the (tiny) model: an internal self-state + a compute-allocation readout
#     NOTE: fixed weights, designed to be legible. In the real system this
#     mapping is *learned in the weights* and verified by ablation + controls.
# --------------------------------------------------------------------------
W_aff  = rng.standard_normal((H, BODY_DIM)) * 0.6      # afferent: body -> self-state ("the nerve")
w_task = rng.standard_normal(H) * 0.5                  # a task-difficulty input into the self-state
W_rand = rng.standard_normal((H, BODY_DIM)) * 0.6      # a random body->self map (specificity control)

# the readout reads the direction that STRAIN takes inside the self-state, and
# turns it DOWN: more strain -> think less (save power). fresh -> think longer.
_strain_dir = W_aff @ STRAIN_W
w_read = -_strain_dir / (np.linalg.norm(_strain_dir) + 1e-9)

def self_state(body, task=0.0, wire=True, W=None):
    afferent = (W if W is not None else W_aff) @ body if wire else 0.0
    return np.tanh(w_task * task + afferent)

def think_budget(body, task=0.0, wire=True, W=None):
    """How many reasoning steps the model chooses to spend, 1..MAX_STEPS."""
    s = self_state(body, task, wire, W)
    p = 1.0 / (1.0 + np.exp(-(w_read @ s)))     # 0..1 propensity to keep thinking
    return 1 + int(round(p * (MAX_STEPS - 1)))

# --------------------------------------------------------------------------
#  3. demonstration
# --------------------------------------------------------------------------
def line(c="-"): print(c * 66)

def live():
    body = read_body()
    names = ["cpu_load", "cpu_temp", "memory  ", "jitter  "]
    print("\nThis machine's body right now:")
    for n, v in zip(names, body):
        bar = "#" * int(v * 24)
        print(f"  {n}  {v:4.2f}  |{bar:<24}|")
    s = strain_of(body)
    b = think_budget(body)
    mood = "strained -> thinking less, saving power" if s > 0.5 else "fresh -> free to think deeper"
    print(f"\n  overall strain = {s:4.2f}")
    print(f"  model grounds this in its self-state -> chooses {b}/{MAX_STEPS} reasoning steps  ({mood})")

def sweep(n=500):
    """Sample many DIFFERENT body states; correlate real strain with the model's decision.

    We sample bodies at random (not a single monotone ramp) on purpose: along a
    monotone ramp *any* linear map trivially correlates with strain. Random states
    are the honest test - only a map that genuinely reads the strain direction survives.
    """
    strains, wired, cut, randmap = [], [], [], []
    for _ in range(n):
        body = rng.random(BODY_DIM)                      # random physical state, each signal 0..1
        strains.append(strain_of(body))
        wired.append(  think_budget(body, wire=True))
        cut.append(    think_budget(body, wire=False))
        randmap.append(think_budget(body, wire=True, W=W_rand))
    corr = lambda y: float(np.corrcoef(strains, y)[0, 1]) if np.std(y) > 1e-9 else 0.0
    return corr(wired), corr(cut), corr(randmap)

def report_bar(label, r):
    mag = abs(r)
    bar = "#" * int(mag * 30)
    print(f"  {label:<34} corr = {r:+.2f}  |{bar:<30}|")

def main():
    line("=")
    print(" ENIMBLE - Embodied AI : minimal runnable demo")
    line("=")
    live()

    print("\nSweeping body states (idle -> heavy load) and measuring whether the")
    print("model's THINK-BUDGET decision actually tracks the body:\n")
    r_wired, r_cut, r_rand = sweep()
    report_bar("real body, wired -> self-state", r_wired)
    report_bar("cut the wire (ablate afferent)", r_cut)
    report_bar("random body->self map (control)", r_rand)

    line()
    verdict = "PASS" if (abs(r_wired) > 0.8 and abs(r_cut) < 0.2 and abs(r_rand) < 0.3) else "see values"
    print(f"  [{verdict}]  Only the real body, wired through the model's own self-state,")
    print( "         drives the decision. Cut the wire -> it goes flat.")
    print( "         That is the mechanism, verified by intervention.")
    line()
    print("\nThis is the minimal mechanism only. The full, verified results (large")
    print("LLMs, learned-in-weight coupling, shuffle/specificity controls) require a")
    print("GPU - see README.md > 'Hardware requirements'.  ->  https://www.enimble.se/embodied\n")

if __name__ == "__main__":
    main()
