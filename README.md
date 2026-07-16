# ENIMBLE — Embodied AI (minimal demo)

**Grounding a model's own computation in its real physical body — and proving it by intervention.**

Most "AI + robot" systems bolt a model on top and feed it sensor readings as external
inputs. We do the opposite: we wire a machine's **real physical state** (energy,
temperature, load, motor current, IMU …) directly into the model's **own internal
self-state**, so it self-regulates its compute and energy *from the inside*. And we
verify the coupling the only honest way — by intervening on the model's internals and
watching what breaks.

> One-page overview: **https://www.enimble.se/embodied**

---

## What this repo is (and isn't)

This repo is a **minimal, runnable demonstration of the mechanism** — small enough to run
on any laptop in seconds, so you can see the idea working end-to-end:

```
real body (this computer's CPU load / temp / memory / jitter)
   → wired into the model's internal SELF-STATE
        → which sets how long the model "thinks" (its compute budget)
             → verified: cut the wire and the decision goes flat.
```

It is **not** the full research system. The published, verified results use large
(7B–32B parameter) language models, a learned-in-the-weights coupling, and
mechanistic-interpretability tooling (sparse-autoencoder features, activation
interventions, shuffle/specificity controls) on GPUs. This demo illustrates the *shape*
of that mechanism with a tiny, legible model — the method itself is intentionally not
included here.

---

## Where the research is (high level)

The full system has been validated on large models — high level, and verified by
intervention rather than anecdote:

- Demonstrated on a **27B-parameter** LLM, with a cross-check on a second GPU vendor.
- The body→self coupling lives **in the weights** (not a bolt-on): remove the trained
  coupling and the effect **collapses**.
- It is **specific to the real body**: a random or time-scrambled signal produces
  essentially no effect.
- It **generalizes** to physical states the coupling never saw during training, and the
  base model's behavior is **preserved** (a small, contained edit).
- The model carries its real measured energy in its **own internal workspace**, and can
  **express its state grounded in that real energy** — verified specific to the real body.
- We also report the honest **nulls** — where a variant doesn't yet work — because the
  standard here is intervention, not a good-looking anecdote.

Everything is a **functional** result. No claim is made that anything is *felt*.

---

## Quickstart

```bash
pip install -r requirements.txt      # numpy + psutil only
python embodied_demo.py
```

No GPU, no model download, no internet. Runs in a few seconds.

### What you'll see

```
This machine's body right now:
  cpu_load  0.20  |####                    |
  cpu_temp  0.35  |########                |
  memory    0.24  |#####                   |
  jitter    0.03  |                        |

  overall strain = 0.24
  model grounds this in its self-state -> chooses 4/12 reasoning steps  (fresh -> free to think deeper)

Sweeping body states (idle -> heavy load):
  real body, wired -> self-state     corr = -0.84   (strained -> thinks less, saves power)
  cut the wire (ablate afferent)     corr = +0.00   (decision stops tracking the body)
  random body->self map (control)    corr = -0.08   (only the real, properly-wired body drives it)

  [PASS] Only the real body, wired through the model's own self-state, drives the decision.
         Cut the wire -> it goes flat. That is the mechanism, verified by intervention.
```

The three numbers are the whole point:

| Test | What it rules out | Expected |
|---|---|---|
| **real body, wired** | — (the effect itself) | strong correlation |
| **cut the wire** | a bolt-on effect living outside the model | ~0 (collapses) |
| **random body→self map** | any signal would do (not specific) | ~0 |

Only when the effect is **strong when wired, dead when cut, and dead for a random map** is
the coupling real, load-bearing, and specific to the body — not a script.

---

## How it maps onto a robot (e.g. Unitree Go2 / G1)

The demo reads a laptop's body. On a robot the body is richer and already exposed:

| Demo (laptop) | Robot (Go2 / G1) |
|---|---|
| CPU load | overall compute / actuation load |
| CPU temperature | battery temp, motor/driver temps |
| memory / jitter | battery state-of-charge, joint currents, IMU |

The runtime loop is lightweight and runs **on-board** (the model reads the robot's own
telemetry into its self-state and decides how much to think / how aggressively to act /
when to conserve power). The heavy training and verification happen **off-board** on a
GPU workstation, once — see below.

---

## Try it on a robot — in simulation first (no hardware needed)

You don't need a physical robot to test the full loop. Unitree's **`unitree_mujoco`**
simulator speaks the *same* `LowState` messages as the real robot, so the adapter and
demo below run **identically in sim and on hardware** — validate everything on a normal
machine first.

Files:
- **`robot_adapter.py`** — reads the robot's real body (joint torque/temperature,
  battery, IMU) from the Unitree SDK `LowState` into the same 4-vector the model uses.
- **`robot_demo.py`** — runs the embodied loop + "cut the wire" control on the robot's
  body instead of a laptop's CPU.
- **`embodied_learning.py`** — an agent in a real Unitree **Go2 MuJoCo** sim that feels
  the robot's body (joint load → heat, battery) and **learns online to self-regulate**:
  push when fresh, ease off to cool when hot / low. It corrects its policy each round
  ("I did that — I actually needed this"). The benefit is honest and specific to feeling
  the body — in our run the body-**aware** learner cut throttling events **10×** (41→4)
  and beat the body-**blind** learner on reward, while a fixed max-effort policy throttled
  ~600×. Runs headless on CPU: `pip install mujoco robot_descriptions numpy && python embodied_learning.py`
- **`embodied_bilinear.py`** — why the body should **gate the thought (bilinear), not be added**.
  The right effort depends on body *and* task together (rest cheap tasks when strained, but
  push the valuable ones), so the body's effect must depend on the task. A best-possible
  additive model can't represent that (5× higher fit error; it gives a strained robot 0.72
  effort on a hard task it should do at 0.90, and can't tell easy from hard when tired); the
  gated model gets it right. This is the deeper coupling — and the reason the research weaves
  the body in bilinearly. Pure CPU: `python embodied_bilinear.py`
- **`embodied_deep.py`** — the two pieces together: an agent that **learns online** in the Go2
  physics loop **and** couples the body **bilinearly**. Honest ordering, exactly as theory
  predicts — blind < additive < bilinear: the bilinear agent earns **~14% more value** on the
  same body budget, because it learned to rest the cheap tasks (effort 0.05) but keep working
  the valuable ones (0.84) when strained — an interaction the additive agent can't represent.
  `python embodied_deep.py`
- **`experiments.py`** — comprehensive study on the **Unitree G1** humanoid (29 actuators) with a richer body (thermal **accumulator** + load **derivative**) over multiple seeds. Result: the bilinear coupling beats additive in **6/6 seeds** (+169% mean value) and is the only agent with net-positive value — the deeper coupling is a robust, honest advantage, not a one-off. `python experiments.py`
- **`experiments2.py`** — richness sweep on the **G1**: does a richer body help? Honest finding — the gating advantage is **robust at every richness level** (bilinear positive, additive negative), and the additive agent gains **nothing** from extra channels (it can't route them into the decision). What matters is the body **gating** the computation, not the channel count. `python experiments2.py`

Setup (sim):
```bash
pip install -r requirements-robot.txt          # unitree-sdk2py + mujoco
# 1) run the simulator:  https://github.com/unitreerobotics/unitree_mujoco
#    (simulate_python; pick the Go2 or G1 scene in its config)
# 2) then, in this repo:
python robot_adapter.py     # prints the robot's live body vector
python robot_demo.py        # runs the embodied loop on the robot's body
```

Platform note: **Go2 / B2 / H1** use the `unitree_go` IDL; **G1 / H1-2** use `unitree_hg`.
Set `ROBOT = "go2"` or `"g1"` at the top of `robot_adapter.py`. The adapter is a **template**
— field names follow the documented SDK; verify them against your SDK version (anything
missing degrades to `0.0` rather than crashing).

Repos: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) ·
[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

---

## Hardware requirements

Be clear on the two very different regimes:

### 1. This minimal demo — any normal computer
- **CPU only.** No GPU. Python 3.9+, `numpy`, `psutil`.
- ~50 MB, runs in seconds. Works on a laptop, a NUC, or a robot's on-board computer.

### 2. The full, verified research — a GPU workstation
The published results (large LLMs + learned-in-weight coupling + interpretability
verification) need real GPU memory:

| Model size | GPU memory (bf16 inference) | Notes |
|---|---|---|
| ~7B | ~16 GB | e.g. a single 24 GB card |
| ~14B | ~28–32 GB | |
| ~27–32B | ~48–64 GB | the self-state-feature work runs here |

- **Reference platform:** an NVIDIA **GB10** (Grace-Blackwell, 119 GB unified memory);
  the cross-hardware result was also validated on an **AMD** GPU.
- **Software:** PyTorch, `transformers`, sparse-autoencoder + activation-intervention tooling.
- **Time:** minutes to a few hours per experiment (training the small in-weight coupling +
  running the intervention controls).

### 3. On-robot deployment — modest
For the *deployed* loop, the model can be small and/or quantized and run on the robot's
on-board compute (a Jetson Orin-class module, as on Unitree EDU units). Inference is
cheap; the loop is a lightweight read → self-state → decision cycle. The GPU workstation
is only needed to *train and verify* the coupling once, not to run it on the robot.

---

## Honest ceiling

We claim a **functional** result: the model holds a verified, load-bearing internal model
of its own physical state that measurably drives its behavior. We make **no claim** that
the system *feels* anything in any conscious sense — that is a separate, unproven question,
and we will not overclaim it.

---

## Contact

**Eric Bergvall** — ENIMBLE
· web: https://www.enimble.se · one-pager: https://www.enimble.se/embodied
· email: bergvall.eric@gmail.com

Prepared for evaluation and collaboration with Unitree Robotics.
