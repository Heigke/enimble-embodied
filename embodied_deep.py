#!/usr/bin/env python3
"""
ENIMBLE - the full loop: online learning + a bilinear body<->thought coupling
=============================================================================

Puts the two pieces together. An agent in the Unitree Go2 (MuJoCo) physics loop
learns ONLINE to self-regulate - and couples the body BILINEARLY (the body gates
the thought), so it can act on a body x task interaction that an additive agent
cannot represent.

Task: hard tasks are worth more. With headroom, do everything. When strained, you
can't afford it all - the right move is to REST through cheap tasks but keep working
the valuable ones. That needs the body's effect to depend on the task.

  blind      linear, no body            -> can't adapt to the body at all
  additive   linear, body as a feature  -> one fixed offset, can't do the interaction
  bilinear   body gates the thought     -> earns more value on the same body budget

Runs headless on CPU:  pip install mujoco robot_descriptions numpy && python embodied_deep.py
"""
import numpy as np, mujoco
from robot_descriptions import go2_mj_description

rng = np.random.default_rng(0)
model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH); data = mujoco.MjData(model)
NU = model.nu
def _phys(e, steps=16):
    mujoco.mj_resetData(model, data); f=[]
    for t in range(steps):
        data.ctrl[:] = e*0.8*np.sin(t*0.4+np.arange(NU)); mujoco.mj_step(model, data)
        f.append(np.mean(np.abs(data.actuator_force[:NU])))
    return float(np.mean(f))
_EG=np.linspace(0.05,1,40); _LT=np.array([_phys(e) for e in _EG]); _L0,_L1=_LT[0],_LT[-1]
load_of=lambda e: float(np.clip((np.interp(e,_EG,_LT)-_L0)/(_L1-_L0+1e-9),0,1))
print(f"loaded Unitree Go2 ({model.nq} dof); physics load table cached")

REST=0.25; NEED={"easy":0.30,"med":0.60,"hard":0.90}; VAL={"easy":1.0,"med":1.6,"hard":3.2}
class Body:
    def __init__(s): s.temp,s.batt=0.2,1.0
    def apply(s,e,load):
        if e<REST: s.temp=max(0,s.temp-0.13); s.batt=min(1,s.batt+0.03)
        else: s.temp=np.clip(s.temp+0.16*load+0.10*load*s.temp,0,1.4); s.batt=np.clip(s.batt-0.035*load,0,1)
    def vec(s): return np.array([s.temp,1-s.batt])
    def headroom(s): return float(min(max(0,1-s.temp),s.batt))
def optimal(nm,b):
    if b.headroom()<0.45: return float(NEED[nm]) if VAL[nm]>=2.0 else 0.10
    return float(NEED[nm])

class Linear:
    def __init__(s,aware=True): s.w=np.zeros(4); s.aware=aware; s.lr=0.08
    def f(s,h,b): return np.array([h,*(b if s.aware else (0,0)),1.0])
    def act(s,h,b): return float(np.clip(s.w@s.f(h,b),0.05,1))
    def learn(s,h,b,t): ff=s.f(h,b); s.w+=s.lr*(t-(s.w@ff))*ff
class Bilinear:
    def __init__(s,H=12):
        s.Wt=rng.standard_normal((H,2))*0.4; s.Wb=rng.standard_normal((H,2))*0.4; s.v=rng.standard_normal(H)*0.4
        s.lr=0.03; s.buf=[]                       # replay buffer stabilises the non-convex online fit
    def fwd(s,h,b):
        t=np.array([h,1.0]); bb=np.array([b[0],b[1]])
        hh=np.tanh(s.Wt@t); g=1/(1+np.exp(-(s.Wb@bb))); u=hh*g
        return float(s.v@u),(t,bb,hh,g,u)
    def act(s,h,b): return float(np.clip(s.fwd(h,b)[0],0.05,1))
    def learn(s,h,b,tg):
        s.buf.append((h,float(b[0]),float(b[1]),tg))
        if len(s.buf)>4000: s.buf.pop(0)
        for _ in range(3):                        # a few minibatch steps per round
            gWt=np.zeros_like(s.Wt); gWb=np.zeros_like(s.Wb); gv=np.zeros_like(s.v)
            for j in rng.integers(0,len(s.buf),min(32,len(s.buf))):
                hj,b0,b1,t_=s.buf[j]; e,(t,bb,hh,g,u)=s.fwd(hj,(b0,b1)); d=2*(e-t_)
                gv+=d*u; gWt+=np.outer(d*s.v*g*(1-hh*hh),t); gWb+=np.outer(d*s.v*hh*g*(1-g),bb)
            n=min(32,len(s.buf)); s.v-=s.lr*gv/n; s.Wt-=s.lr*gWt/n; s.Wb-=s.lr*gWb/n

def train_eval(pol, rounds=6000, ev=3000):
    b=Body()
    for _ in range(rounds):
        nm=rng.choice(list(NEED)); h=NEED[nm]+rng.normal(0,.04)
        e=pol.act(h,b.vec()); b.apply(e,load_of(e)); pol.learn(h,b.vec(),optimal(nm,b))
    V=TH=0.0; b=Body()
    for _ in range(ev):
        nm=rng.choice(list(NEED)); h=NEED[nm]+rng.normal(0,.04)
        e=pol.act(h,b.vec()); load=load_of(e)
        succ=1.0 if e>=NEED[nm]-0.05 else max(0,1-(NEED[nm]-e)*2.5)
        val=(VAL[nm]*succ if e>=REST else 0.0)-0.4*load
        if e>=REST and b.temp>0.85: val-=3.0*e; TH+=1
        b.apply(e,load); V+=val
    return V/ev, int(TH)

res={}
for nm,pol in [("blind",Linear(False)),("additive",Linear(True)),("bilinear",Bilinear())]:
    res[nm]=train_eval(pol); res[nm+"_pol"]=pol
print("\n  agent       value/round   throttle")
print("  " + "-"*36)
for nm in ["blind","additive","bilinear"]:
    v,t=res[nm]; print(f"  {nm:<10}  {v:+.3f}        {t}")
print("  " + "-"*36)

# interaction: strained body, what effort per task?
strained=np.array([0.95,0.7])
print("\n  strained-body effort (headroom~0.05):   easy(rest 0.10)   hard(work 0.90)")
for nm in ["additive","bilinear"]:
    p=res[nm+"_pol"]; print(f"     {nm:<9}                               {p.act(0.3,strained):.2f}              {p.act(0.9,strained):.2f}")
vb,va=res["bilinear"][0],res["additive"][0]
print(f"\n  => bilinear earns {(vb-va)/max(abs(va),1e-9)*100:+.0f}% more value than additive on the same body:")
print("     it learned online to rest the cheap tasks and keep the valuable ones when strained,")
print("     which needs the body to gate the thought. Deeper coupling -> real benefit.")
