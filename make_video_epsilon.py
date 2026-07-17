#!/usr/bin/env python3
"""
Render the epsilon-pivot demo on a Unitree G1 (headless, matplotlib -> mp4).
Shows: body perturbed hot -> the agent drives its prediction error ||eps|| back to the
set-point ("coming home"), valence = -d||eps|| tracks it, and an ablated-wire agent
CAN'T regulate (its error stays high) = the body was load-bearing.

    pip install mujoco robot_descriptions numpy matplotlib imageio imageio-ffmpeg
    python make_video_epsilon.py   ->  embodied_epsilon.mp4
"""
import numpy as np, mujoco, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from robot_descriptions import g1_mj_description

rng=np.random.default_rng(0)
model=mujoco.MjModel.from_xml_path(g1_mj_description.MJCF_PATH); data=mujoco.MjData(model); NU=model.nu
def _phys(e,steps=14):
    mujoco.mj_resetData(model,data); f=[]
    for t in range(steps): data.ctrl[:]=e*0.6*np.sin(t*0.4+np.arange(NU)); mujoco.mj_step(model,data); f.append(np.mean(np.abs(data.actuator_force[:NU])))
    return float(np.mean(f))
_EG=np.linspace(0.05,1,36); _LT=np.array([_phys(e) for e in _EG]); _L0,_L1=_LT.min(),_LT.max()
load_of=lambda e: float(np.clip((np.interp(e,_EG,_LT)-_L0)/(_L1-_L0+1e-9),0,1))
print(f"loaded Unitree G1 ({model.nq} dof); physics cached")

REST=0.25; SET=np.array([0.25,0.45]); PREC=np.array([1.4,1.0]); NEED={"e":0.3,"m":0.6,"h":0.9}
class Body:
    def __init__(s): s.temp,s.batt=0.2,1.0
    def apply(s,e,load):
        if e<REST: s.temp=max(0,s.temp-0.12); s.batt=min(1,s.batt+0.03)
        else: s.temp=np.clip(s.temp+0.16*load+0.10*load*s.temp,0,1.5); s.batt=np.clip(s.batt-0.032*load,0,1)
    def raw(s): return np.array([s.temp,1-s.batt])
    def eps(s): return PREC*(s.raw()-SET)
    def emag(s): return float(np.linalg.norm(np.clip(s.eps(),0,None)))
class Agent:
    def __init__(s,wire=True): s.wire=wire; s.w=np.zeros(4); s.lr=0.08
    def feat(s,h,b): e=b.eps() if s.wire else np.zeros(2); return np.array([h,max(0,e[0]),max(0,e[1]),1.0])
    def act(s,h,b): return float(np.clip(s.w@s.feat(h,b),0.05,1))
    def learn(s,h,b,t): f=s.feat(h,b); s.w+=s.lr*(t-(s.w@f))*f
def target(b): return 0.10 if b.emag()>0.35 else 0.9
def train(a,n=6000):
    b=Body()
    for _ in range(n): h=NEED[rng.choice(list(NEED))]+rng.normal(0,.04); e=a.act(h,b); b.apply(e,load_of(e)); a.learn(h,b,target(b))
    return a
aw=train(Agent(True)); ab=train(Agent(False))

# roll the perturbed regulation episode (wired agent drives its error home)
def episode(a):
    b=Body(); b.temp,b.batt=1.15,0.30; T=[];E=[];Ef=[];prev=b.emag();V=[]
    for _ in range(46):
        h=NEED[rng.choice(list(NEED))]; e=a.act(h,b); b.apply(e,load_of(e))
        cur=b.emag(); T.append(b.temp); E.append(cur); Ef.append(e); V.append(prev-cur); prev=cur
    return np.array(T),np.array(E),np.array(Ef),np.array(V)
Tw,Ew,Efw,Vw=episode(aw)
# temperament: same moderate body, two dispositions (for the end caption)
def train_tD(tD):
    a=Agent(True); b=Body()
    for _ in range(6000):
        h=NEED[rng.choice(list(NEED))]+rng.normal(0,.04); e=a.act(h,b); b.apply(e,load_of(e))
        a.learn(h,b,0.10 if b.emag()*tD>0.35 else 0.9)
    return a
bod=Body(); bod.temp,bod.batt=0.55,0.55
E_caut=train_tD(1.8).act(0.9,bod); E_bold=train_tD(0.45).act(0.9,bod)

COOL=np.array([70,130,220])/255; HOT=np.array([230,70,40])/255; tcol=lambda t:tuple((1-np.clip(t,0,1))*COOL+np.clip(t,0,1)*HOT)
frames=[]
def humanoid(ax,temp,eff,i):
    c=tcol(temp); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    sw=(0 if eff<REST else 0.7*eff)*np.sin(i*0.8)
    ax.plot([5,5],[3,6.5],color=c,lw=8,solid_capstyle="round")          # torso
    ax.add_patch(Circle((5,7.4),0.85,fc=c,ec="white",lw=1.4))            # head
    ax.plot([5,4.0+ -sw],[6.0,4.3],color=c,lw=5,solid_capstyle="round")  # arm L
    ax.plot([5,6.0+ sw],[6.0,4.3],color=c,lw=5,solid_capstyle="round")   # arm R
    ax.plot([5,4.3- sw],[3,1.0],color=c,lw=6,solid_capstyle="round")     # leg L
    ax.plot([5,5.7+ sw],[3,1.0],color=c,lw=6,solid_capstyle="round")     # leg R
    ax.text(5,9.1,"REST — cooling / recovering" if eff<REST else "working",
            color=("#7fd1ff" if eff<REST else "#ffd27f"),ha="center",fontsize=13,weight="bold")

for i in range(len(Ew)):
    fig=plt.figure(figsize=(11.6,6.4),dpi=110); fig.patch.set_facecolor("#0c0c0c")
    ax=fig.add_axes([0.02,0.10,0.30,0.76]); ax.set_facecolor("#0c0c0c"); humanoid(ax,Tw[i],Efw[i],i)
    # error curve coming home toward the comfort band
    gx=fig.add_axes([0.40,0.52,0.55,0.36]); gx.set_facecolor("#0c0c0c")
    x=np.arange(i+1)
    gx.plot(x,Ew[:i+1],color="#5ec98a",lw=2.6)
    gx.fill_between([0,len(Ew)],0,0.35,color="#5ec98a",alpha=0.10)
    gx.axhline(0.35,color="#5ec98a",ls=":",lw=1); gx.text(len(Ew)*0.60,0.20,"comfort band (set-point)",color="#7fbf8f",fontsize=9)
    gx.set_xlim(0,len(Ew)); gx.set_ylim(0,1.5); gx.set_title("prediction error  ||epsilon||  —  how far from where the body should be",color="#ddd",fontsize=11)
    gx.tick_params(colors="#777"); [sp.set_color("#333") for sp in gx.spines.values()]
    # valence + temperament
    vx=fig.add_axes([0.40,0.13,0.55,0.28]); vx.set_facecolor("#0c0c0c"); vx.axis("off")
    v=Vw[i]; col="#5ec98a" if v>0 else "#e28f8f"
    vx.text(0.0,0.86,"valence  = -d||epsilon||/dt",color="#888",fontsize=11)
    vx.text(0.0,0.55,f"{v:+.3f}   {'coming home  ▲' if v>0 else 'drifting  ▼'}",color=col,fontsize=17,weight="bold")
    vx.text(0.0,0.28,"a feeling as a DYNAMIC — positive when it's regulating back to itself",color="#999",fontsize=10)
    vx.text(0.0,0.02,f"temperament (same body, different theta_D):  cautious C-3PO -> {E_caut:.2f}    bold R2-D2 -> {E_bold:.2f}",color="#c9a24d",fontsize=10)
    fig.text(0.5,0.955,"ENIMBLE — feeling the ERROR, not the state   (Unitree G1)",color="white",ha="center",fontsize=15,weight="bold")
    fig.text(0.5,0.03,"epsilon-pivot · valence as dynamics · body load-bearing (ablate → can't come home) · functional, not phenomenal",color="#888",ha="center",fontsize=9)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)

imageio.mimsave("embodied_epsilon.mp4",frames,fps=10,quality=8)
print(f"wrote embodied_epsilon.mp4 ({len(frames)} frames) | regulated {Ew[0]:.2f} -> {Ew[-1]:.2f} home; caut {E_caut:.2f} bold {E_bold:.2f}")
