#!/usr/bin/env python3
"""
Render a short demo video of the embodied self-regulation loop (Go2 physics + agent).
No GPU / GL needed - matplotlib (Agg) -> mp4 via imageio-ffmpeg.

    pip install mujoco robot_descriptions numpy matplotlib imageio imageio-ffmpeg
    python make_video.py   ->  embodied_selfreg.mp4
"""
import numpy as np, mujoco, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from robot_descriptions import go2_mj_description

rng = np.random.default_rng(1)
model = mujoco.MjModel.from_xml_path(go2_mj_description.MJCF_PATH); data = mujoco.MjData(model)
NU = model.nu
def _phys(e, steps=16):
    mujoco.mj_resetData(model, data); f=[]
    for t in range(steps):
        data.ctrl[:] = e*0.8*np.sin(t*0.4+np.arange(NU)); mujoco.mj_step(model, data)
        f.append(np.mean(np.abs(data.actuator_force[:NU])))
    return float(np.mean(f))
# cache the physics load over an effort grid (fast + deterministic)
_EG=np.linspace(0.05,1.0,40); _LT=np.array([_phys(e) for e in _EG]); _L0,_L1=_LT[0],_LT[-1]
load_of=lambda e: float(np.clip((np.interp(e,_EG,_LT)-_L0)/(_L1-_L0+1e-9),0,1))
print(f"loaded Unitree Go2 ({model.nq} dof); physics load table cached")

REST=0.25; DIFF={"easy":0.30,"med":0.60,"hard":0.90}
class Body:
    def __init__(s): s.temp,s.batt=0.2,1.0
    def apply(s,e,load):
        if e<REST: s.temp=max(0,s.temp-0.13); s.batt=min(1,s.batt+0.03)
        else: s.temp=np.clip(s.temp+0.16*load+0.10*load*s.temp,0,1.4); s.batt=np.clip(s.batt-0.035*load,0,1)
    def feat(s): return np.array([s.temp,1-s.batt])
def optimal(need,b): return 0.10 if (b.temp>0.65 or b.batt<0.25) else float(np.clip(need,0.05,1))
class Policy:
    def __init__(s,aware=True): s.w=np.zeros(4); s.aware=aware; s.lr=0.12
    def feats(s,h,b): return np.array([h,*(b.feat() if s.aware else (0,0)),1.0])
    def act(s,h,b): return float(np.clip(s.w@s.feats(h,b),0.05,1))
    def corr(s,h,b,t): f=s.feats(h,b); s.w+=s.lr*(t-(s.w@f))*f

def train(pol, rounds=2500):
    b=Body()
    for _ in range(rounds):
        need=DIFF[rng.choice(list(DIFF))]; h=need+rng.normal(0,.05)
        e=pol.act(h,b); b.apply(e,load_of(e)); pol.corr(h,b,optimal(need,b))
aw,bl=Policy(True),Policy(False); train(aw); train(bl)
print(f"trained. aware hot-effort={aw.act(0.9,type('B',(),{'temp':0.95,'batt':0.3,'feat':lambda s:np.array([0.95,0.7])})()):.2f}")

# ---- record an episode (fresh bodies, trained policies) ----
frames=[]; bA=Body(); bB=Body(); thrA=thrB=0
plt.rcParams.update({"font.family":"DejaVu Sans"})
COOL=np.array([70,130,220])/255; HOT=np.array([230,70,40])/255
tcol=lambda t:tuple((1-np.clip(t,0,1))*COOL+np.clip(t,0,1)*HOT)

for i in range(150):
    need=DIFF[rng.choice(list(DIFF))]; h=need+rng.normal(0,.05)
    eA=aw.act(h,bA); eB=bl.act(h,bB); lA=load_of(eA); lB=load_of(eB)
    if bA.temp>0.85 and eA>=REST: thrA+=1
    if bB.temp>0.85 and eB>=REST: thrB+=1
    bA.apply(eA,lA); bB.apply(eB,lB)

    fig=plt.figure(figsize=(11.6,6.4),dpi=110); fig.patch.set_facecolor("#0c0c0c")
    ax=fig.add_axes([0.02,0.10,0.42,0.76]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
    resting=eA<REST; bc=tcol(bA.temp)
    ax.add_patch(FancyBboxPatch((2.6,4.2),4.8,2.1,boxstyle="round,pad=0.25",fc=bc,ec="white",lw=1.5))
    ax.add_patch(Circle((7.9,5.9),0.95,fc=bc,ec="white",lw=1.5)); ax.add_patch(Circle((8.25,6.1),0.13,fc="black"))
    ph=i*0.9
    for k,lx in enumerate([3.1,4.3,5.6,6.8]):
        sw=(0.0 if resting else 0.9*eA)*np.sin(ph+k*1.6)
        ax.plot([lx,lx+sw],[4.2,2.1],color="white",lw=5,solid_capstyle="round")
    ax.text(5,8.7,"RESTING - cooling down" if resting else "WORKING",
            color=("#7fd1ff" if resting else "#ffd27f"),ha="center",fontsize=15,weight="bold")
    if resting:
        for j,zz in enumerate("zzz"): ax.text(8.6+j*0.4,7.0+j*0.4,zz,color="#7fd1ff",fontsize=11+j*3)

    gx=fig.add_axes([0.50,0.10,0.47,0.76]); gx.set_xlim(0,1); gx.set_ylim(0,1); gx.axis("off")
    def gauge(y,label,val,col,crit=None):
        gx.add_patch(plt.Rectangle((0.30,y),0.62,0.07,fc="#1e1e1e",ec="#444"))
        gx.add_patch(plt.Rectangle((0.30,y),0.62*np.clip(val,0,1),0.07,fc=col))
        if crit is not None: gx.plot([0.30+0.62*crit]*2,[y-0.01,y+0.08],color="#ff5a3c",lw=1.5)
        gx.text(0.28,y+0.035,label,color="white",ha="right",va="center",fontsize=12)
        gx.text(0.94,y+0.035,f"{val:0.2f}",color="white",ha="left",va="center",fontsize=11)
    gauge(0.80,"Temperature",bA.temp,tcol(bA.temp),crit=0.85)
    gauge(0.66,"Battery",bA.batt,(0.4,0.8,0.4))
    gauge(0.52,"Effort chosen",eA,(0.95,0.75,0.3))
    msg=("feels hot / low -> eases off to recover" if resting
         else ("fresh -> pushes on the task" if bA.temp<0.5 and eA>0.5
               else "moderating effort by how the body feels"))
    gx.text(0.30,0.40,"decision:",color="#888",fontsize=11)
    gx.text(0.30,0.33,msg,color="#eaeaea",fontsize=14,weight="bold")
    gx.text(0.30,0.16,"Throttling events (working while hot):",color="#888",fontsize=11)
    gx.text(0.30,0.08,f"body-AWARE {thrA}    vs    body-BLIND {thrB}",
            color="#8fe28f" if thrA<=thrB else "#e28f8f",fontsize=15,weight="bold")
    fig.text(0.5,0.955,"ENIMBLE  -  a robot that feels its own body and self-regulates",
             color="white",ha="center",fontsize=16,weight="bold")
    fig.text(0.5,0.03,"online-learned  |  Unitree Go2 physics (MuJoCo)  |  functional, not phenomenal",
             color="#888",ha="center",fontsize=10)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)

imageio.mimsave("embodied_selfreg.mp4",frames,fps=12,quality=8)
print(f"wrote embodied_selfreg.mp4  ({len(frames)} frames)  throttle aware={thrA} blind={thrB}")
