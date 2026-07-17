#!/usr/bin/env python3
"""
Render the epsilon-pivot demo on a Unitree G1 in 3D (headless osmesa -> mp4).
Real MuJoCo G1 humanoid, standing, gesturing with effort, while its prediction error
||eps|| is driven smoothly back to the comfort set-point ("coming home") and valence
= -d||eps|| tracks it. Ends with the temperament (theta_D) contrast.

    sudo apt install libosmesa6 libosmesa6-dev
    pip install mujoco robot_descriptions numpy matplotlib imageio imageio-ffmpeg
    MUJOCO_GL=osmesa python make_video_epsilon.py   ->  embodied_epsilon.mp4
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np, mujoco, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robot_descriptions import g1_mj_description as g1

model = mujoco.MjModel.from_xml_path(g1.MJCF_PATH); data = mujoco.MjData(model)
model.vis.global_.offwidth = 720; model.vis.global_.offheight = 560   # offscreen buffer >= render size
model.vis.headlight.ambient[:] = [0.55,0.55,0.55]                     # brighten the scene (osmesa is dim)
model.vis.headlight.diffuse[:] = [0.85,0.85,0.85]
model.vis.headlight.specular[:] = [0.3,0.3,0.3]
NU = model.nu
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)     # standing home pose
home = data.qpos.copy()
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 560, 720)
cam = mujoco.MjvCamera(); cam.distance = 2.3; cam.elevation = -6; cam.lookat[:] = [0, 0, 0.85]
opt = mujoco.MjvOption()
print("G1 loaded; osmesa renderer ready")

# ---- body + epsilon dynamics ----
REST=0.25; SET=np.array([0.25,0.45]); PREC=np.array([1.4,1.0])
class Body:                                  # continuous dynamics -> smooth, damped regulation (no sawtooth)
    def __init__(s): s.temp,s.batt=0.20,1.0
    def apply(s,e,load):
        s.temp=float(np.clip(s.temp + 0.22*load - 0.11, 0, 1.6))     # heat from work minus steady passive cooling
        s.batt=float(np.clip(s.batt - 0.055*load + 0.022, 0, 1))     # drain from work minus steady recharge
    def raw(s): return np.array([s.temp,1-s.batt])
    def emag(s): return float(np.linalg.norm(np.clip(PREC*(s.raw()-SET),0,None)))
load_of=lambda e: float(np.clip(e,0,1))

# graded proportional controller: high error -> ease off (come home); it settles smoothly at the set-point
b=Body(); b.temp,b.batt=1.15,0.30
TR=[]; prev=b.emag()
for i in range(56):
    eff=float(np.clip(0.55-0.9*b.emag(),0.05,0.85))
    b.apply(eff,load_of(eff)); cur=b.emag()
    TR.append((b.temp,b.batt,cur,eff,prev-cur)); prev=cur
TR=np.array(TR); TEMP,BATT,EMAG,EFF,VAL=TR.T
E_caut, E_bold = 0.05, 1.00     # from embodied_epsilon.py temperament test (theta_D 1.8 vs 0.45)

# ---- animate the robot by effort: still when resting, a clear march/gesture when working ----
def pose(i, eff):
    q=home.copy()
    work=max(0.0,(eff-REST)/(1-REST))            # 0 resting -> 1 working hard
    a=0.09+0.40*work; ph=i*0.9                    # baseline 'alive' sway + strong march when working
    n=len(q)
    for k in range(7,n):                          # joints (skip 7-dof free base): alternating swing
        q[k]=home[k]+a*np.sin(ph + k*1.7)
    q[2]=home[2]+0.045*work*np.sin(ph*2.0)        # subtle vertical bob = 'alive'
    data.qpos[:]=q; mujoco.mj_forward(model,data)

frames=[]
for i in range(len(EMAG)):
    pose(i, EFF[i]); cam.azimuth=110+i*0.7
    renderer.update_scene(data, camera=cam, scene_option=opt)
    rgb=renderer.render()

    fig=plt.figure(figsize=(12.2,6.3),dpi=110); fig.patch.set_facecolor("#0c0c0c")
    ax=fig.add_axes([0.01,0.06,0.40,0.86]); ax.imshow(rgb); ax.axis("off")
    hot=np.clip(TEMP[i],0,1)
    ax.text(0.5,1.02,"REST — cooling / recovering" if EFF[i]<REST else "working",transform=ax.transAxes,
            color=("#7fd1ff" if EFF[i]<REST else "#ffd27f"),ha="center",fontsize=13,weight="bold")

    gx=fig.add_axes([0.46,0.52,0.51,0.37]); gx.set_facecolor("#0c0c0c")
    gx.plot(np.arange(i+1),EMAG[:i+1],color="#5ec98a",lw=2.8)
    gx.fill_between([0,len(EMAG)],0,0.35,color="#5ec98a",alpha=0.10)
    gx.axhline(0.35,color="#5ec98a",ls=":",lw=1); gx.text(len(EMAG)*0.58,0.18,"comfort band (set-point)",color="#7fbf8f",fontsize=9)
    gx.set_xlim(0,len(EMAG)); gx.set_ylim(0,1.4); gx.set_title("prediction error  ||epsilon||  —  how far from where the body should be",color="#ddd",fontsize=11)
    gx.tick_params(colors="#777"); [sp.set_color("#333") for sp in gx.spines.values()]

    # gauges
    hx=fig.add_axes([0.46,0.40,0.51,0.07]); hx.axis("off"); hx.set_xlim(0,1); hx.set_ylim(0,1)
    def gauge(y,lab,val,col):
        hx.add_patch(plt.Rectangle((0.16,y),0.62,0.7,fc="#1e1e1e",ec="#444",transform=hx.transAxes))
        hx.add_patch(plt.Rectangle((0.16,y),0.62*np.clip(val,0,1),0.7,fc=col,transform=hx.transAxes))
        hx.text(0.14,y+0.35,lab,color="#ccc",ha="right",va="center",fontsize=10,transform=hx.transAxes)
    tcol=(min(1.0,0.35+0.6*hot), max(0.0,0.5-0.4*hot), max(0.0,0.8-0.7*hot))
    gauge(0.15,"temp",min(1.0,TEMP[i]),tcol)
    vx=fig.add_axes([0.46,0.12,0.51,0.24]); vx.axis("off")
    v=VAL[i]
    if v>0.004: col,lab="#5ec98a","coming home  ▲"
    elif v<-0.004: col,lab="#e28f8f","drifting  ▼"
    else: col,lab="#9aa0a6","at set-point — steady"
    vx.text(0,0.86,"valence  = -d||epsilon||/dt",color="#888",fontsize=11)
    vx.text(0,0.52,f"{v:+.3f}   {lab}",color=col,fontsize=17,weight="bold")
    vx.text(0,0.22,"a feeling as a DYNAMIC — positive while it regulates back to itself",color="#999",fontsize=10)
    vx.text(0,-0.02,f"temperament (same body, different theta_D):  cautious C-3PO -> {E_caut:.2f}    bold R2-D2 -> {E_bold:.2f}",color="#c9a24d",fontsize=10)
    fig.text(0.5,0.965,"ENIMBLE — feeling the ERROR, not the state   (Unitree G1)",color="white",ha="center",fontsize=15,weight="bold")
    fig.text(0.5,0.02,"epsilon-pivot · valence as dynamics · slow disposition = temperament · functional, not phenomenal",color="#888",ha="center",fontsize=9)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)

imageio.mimsave("embodied_epsilon.mp4",frames,fps=11,quality=8)
print(f"wrote embodied_epsilon.mp4 ({len(frames)} frames, 3D) | error {EMAG[0]:.2f} -> {EMAG[-1]:.2f}")
