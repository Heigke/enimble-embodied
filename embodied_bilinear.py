#!/usr/bin/env python3
"""
ENIMBLE - deeper coupling: why the body must GATE the thought, not just be added
================================================================================

Clean, honest demonstration of a representational fact (no RL, no clipping tricks):

The right amount of effort depends on the body AND the task *together*. When the
robot is strained it should REST through cheap tasks but still push the valuable
ones. So the body's effect on the decision must DEPEND on the task - a body x task
interaction. An additive model (body as one more input feature) can only apply a
fixed offset, so it cannot represent this. A bilinear/gated model can.

We fit both models to the true (body, task) -> effort map and measure the error.

  ADDITIVE   e = w . [need, headroom, 1]                 (best-possible least squares)
  BILINEAR   e = v . ( tanh(Wt.[need,1]) * sigmoid(Wb.[headroom,1]) )   (gated)

Runs on CPU:  python embodied_bilinear.py
"""
import numpy as np
rng = np.random.default_rng(0)

# --- the true body-conditioned optimum (has a genuine body x task interaction) ---
def optimal(need, headroom):
    strained = headroom < 0.45
    valuable = need >= 0.75                 # hard tasks are worth the strain
    if strained and not valuable:
        return 0.10                         # rest through cheap tasks when strained
    return float(need)                      # otherwise do the task

NEEDS = [0.30, 0.60, 0.90]
X = np.array([[rng.choice(NEEDS), rng.random()] for _ in range(6000)])   # [need, headroom]
Y = np.array([optimal(n, hr) for n, hr in X])

# --- ADDITIVE: best possible linear fit on [need, headroom, 1] ---
A = np.c_[X, np.ones(len(X))]
w, *_ = np.linalg.lstsq(A, Y, rcond=None)
add_pred = lambda n, hr: float(np.array([n, hr, 1]) @ w)
add_mse  = float(np.mean((A @ w - Y) ** 2))

# --- BILINEAR: gated model, trained by gradient descent ---
H = 12
Wt = rng.standard_normal((H, 2)) * 0.4
Wb = rng.standard_normal((H, 2)) * 0.4
v  = rng.standard_normal(H) * 0.4
def bfwd(n, hr):
    t = np.array([n, 1.0]); b = np.array([hr, 1.0])
    h = np.tanh(Wt @ t); g = 1 / (1 + np.exp(-(Wb @ b))); u = h * g
    return float(v @ u), (t, b, h, g, u)
lr = 0.02
for epoch in range(400):
    idx = rng.permutation(len(X))
    for i in idx[:1500]:
        n, hr = X[i]; tgt = Y[i]
        e, (t, b, h, g, u) = bfwd(n, hr); d = 2 * (e - tgt)
        v  -= lr * d * u
        Wt -= lr * np.outer(d * v * g * (1 - h * h), t)
        Wb -= lr * np.outer(d * v * h * g * (1 - g), b)
bil_pred = lambda n, hr: bfwd(n, hr)[0]
bil_mse  = float(np.mean([(bfwd(n, hr)[0] - optimal(n, hr)) ** 2 for n, hr in X]))

# --- report ---
print(f"  fit error (MSE) over the whole (body, task) map:")
print(f"     ADDITIVE (best linear) : {add_mse:.4f}")
print(f"     BILINEAR (gated)       : {bil_mse:.4f}   ({add_mse/max(bil_mse,1e-9):.0f}x lower error)\n")

print("  what each model does with a STRAINED body (headroom 0.15):")
print("     task     true    additive   bilinear")
for nm, n in [("easy", 0.30), ("hard", 0.90)]:
    print(f"     {nm:<6}  {optimal(n,0.15):.2f}    {add_pred(n,0.15):6.2f}     {bil_pred(n,0.15):6.2f}")
print("\n  => strained, the right move is REST easy (0.10) but WORK hard (0.90).")
print("     The best additive model is forced to a compromise for both (it can't tell")
print("     easy from hard when tired). The bilinear model gets both right - the body")
print("     genuinely reshapes the thought. That is the deeper body<->thought<->action coupling,")
print("     and the same reason the research wants the body woven in bilinearly, not added.")
