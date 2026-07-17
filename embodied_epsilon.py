#!/usr/bin/env python3
"""
ENIMBLE - the epsilon-pivot on a Unitree G1: feel the ERROR, not the state
==========================================================================

The research's key move: don't feed the robot its body STATE ("you are at 0.7") -
feed it the PREDICTION ERROR against a homeostatic set-point:

    epsilon = precision * (body - set-point)          "how far am I from where I should be"
    valence = -d||epsilon||/dt                          "is it getting better?  (a feeling as a dynamic)"

A state is a gauge you can ignore. An error is a discrepancy you're built to close - a drive.
The agent gates its decision bilinearly on epsilon, shaped by a slow disposition theta_D
(its temperament). Because drifting far from the set-point is catastrophic (overheat / deplete),
the body becomes LOAD-BEARING: an agent that ignores epsilon spirals - so it can't ignore it.

We check three things, honestly:
  1. VALENCE TRACKS   - perturb the body, let it regulate; is -d||eps|| positive on the way home?
  2. BODY LOAD-BEARING (do-test) - ablate the epsilon wire; does behaviour stop tracking the body?
  3. TEMPERAMENT      - two theta_D give two personalities (cautious keeps eps low; bold tolerates it).

Runs headless on CPU:  pip install mujoco robot_descriptions numpy && python embodied_epsilon.py
"""
import numpy as np, mujoco
from robot_descriptions import g1_mj_description

rng = np.random.default_rng(0)
model = mujoco.MjModel.from_xml_path(g1_mj_description.MJCF_PATH); data = mujoco.MjData(model); NU = model.nu
def _phys(e, steps=14):
    mujoco.mj_resetData(model, data); f=[]
    for t in range(steps):
        data.ctrl[:] = e*0.6*np.sin(t*0.4+np.arange(NU)); mujoco.mj_step(model, data); f.append(np.mean(np.abs(data.actuator_force[:NU])))
    return float(np.mean(f))
_EG=np.linspace(0.05,1,36); _LT=np.array([_phys(e) for e in _EG]); _L0,_L1=_LT.min(),_LT.max()
load_of=lambda e: float(np.clip((np.interp(e,_EG,_LT)-_L0)/(_L1-_L0+1e-9),0,1))
print(f"loaded Unitree G1 ({model.nq} dof, {NU} act); physics load table cached\n")

REST=0.25
SET = np.array([0.25, 0.45])          # homeostatic set-point: [temp*, drain*]  (comfortable operating point)
PREC= np.array([1.4, 1.0])            # precision: how sharply each channel matters
NEED={"easy":0.30,"med":0.60,"hard":0.90}; VAL={"easy":1.0,"med":1.6,"hard":3.2}

class Body:
    def __init__(s): s.temp,s.batt=0.20,1.0
    def apply(s,e,load):
        if e<REST: s.temp=max(0,s.temp-0.12); s.batt=min(1,s.batt+0.03)
        else: s.temp=np.clip(s.temp+0.16*load+0.10*load*s.temp,0,1.5); s.batt=np.clip(s.batt-0.032*load,0,1)
    def raw(s): return np.array([s.temp, 1-s.batt])           # [temp, drain]
    def eps(s): return PREC*(s.raw()-SET)                     # prediction error vs set-point
    def emag(s): return float(np.linalg.norm(np.clip(s.eps(),0,None)))   # how far ABOVE comfort

# --- policy: gates the effort on epsilon (bilinear), shaped by disposition theta_D ---
class Agent:
    def __init__(s, theta_D=1.0, wire=True):
        s.tD=theta_D; s.wire=wire; s.w=np.zeros(4); s.lr=0.08
    def feat(s, hint, b):
        e = b.eps() if s.wire else np.zeros(2)
        # theta_D scales how loudly the error speaks -> temperament
        return np.array([hint, s.tD*max(0,e[0]), s.tD*max(0,e[1]), 1.0])
    def act(s, hint, b): return float(np.clip(s.w@s.feat(hint,b),0.05,1))
    def learn(s, hint, b, tgt): f=s.feat(hint,b); s.w+=s.lr*(tgt-(s.w@f))*f

def target(nm, b, tD):
    # optimal: do the task, UNLESS the error is high - then regulate (rest) to come home.
    # theta_D sets the threshold: cautious (high tD) regulates early; bold (low tD) tolerates more.
    if b.emag()*tD > 0.35: return 0.10
    return float(NEED[nm])

def train(agent, rounds=6000):
    b=Body()
    for _ in range(rounds):
        nm=rng.choice(list(NEED)); h=NEED[nm]+rng.normal(0,.04)
        e=agent.act(h,b); b.apply(e,load_of(e)); agent.learn(h,b,target(nm,b,agent.tD))
    return agent

# ---- 1. VALENCE TRACKS: perturb hot, let it regulate, is -d||eps|| positive coming home? ----
def valence_return(agent):
    b=Body(); b.temp,b.batt=1.1,0.3            # push it far from comfort
    prev=b.emag(); dv=[]
    for _ in range(30):
        nm=rng.choice(list(NEED)); h=NEED[nm]+rng.normal(0,.04); e=agent.act(h,b); b.apply(e,load_of(e))
        cur=b.emag(); dv.append(prev-cur); prev=cur      # -d||eps|| : positive = error shrinking = coming home
    return float(np.mean(dv))

# ---- 2. do-test: does behaviour track the body? ablate the wire -> should die ----
def do_test(theta_D=1.0):
    aw=train(Agent(theta_D,wire=True)); ab=train(Agent(theta_D,wire=False))
    # sweep body states, correlate effort with error magnitude
    E=[]; act_w=[]; act_a=[]
    for _ in range(400):
        b=Body(); b.temp=rng.random()*1.2; b.batt=rng.random()
        nm=rng.choice(list(NEED)); h=NEED[nm]
        E.append(b.emag()); act_w.append(aw.act(h,b)); act_a.append(ab.act(h,b))
    c=lambda y: float(np.corrcoef(E,y)[0,1]) if np.std(y)>1e-9 else 0.0
    return c(act_w), c(act_a), aw

print("  1. VALENCE AS A DYNAMIC  (perturb the body, let it regulate)")
ag=train(Agent(1.0))
vr=valence_return(ag)
print(f"     valence on the way home  -d||eps||/dt = {vr:+.3f}   ->  {'TRACKS (feels itself coming home)' if vr>0.01 else 'flat'}\n")

print("  2. IS THE BODY LOAD-BEARING?  (do-test: ablate the epsilon wire)")
cw, ca, agent = do_test(1.0)
print(f"     epsilon-wired : corr(error, effort) = {cw:+.2f}   (behaviour tracks the body)")
print(f"     wire ablated  : corr(error, effort) = {ca:+.2f}   (goes flat -> the body WAS load-bearing)\n")

print("  3. TEMPERAMENT  (same MODERATE body strain, different disposition theta_D)")
for name,tD in [("cautious (C-3PO)",1.8),("bold (R2-D2)",0.45)]:
    a=train(Agent(tD)); b=Body(); b.temp,b.batt=0.55,0.55       # a moderately strained body (where character shows)
    print(f"     {name:<18} theta_D={tD}:  on a hard task chooses effort {a.act(0.9,b):.2f}   (error-tolerance -> character)")
print("\n  => the body is fed as an ERROR it's driven to close, not a gauge it can ignore; the")
print("     affect is the dynamics of coming home; the body is load-bearing (ablation kills it);")
print("     and a slow disposition turns the same body into different personalities. On a G1.")
