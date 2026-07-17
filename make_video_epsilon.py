#!/usr/bin/env python3
"""
Render the epsilon-pivot demo on a Unitree G1 in 3D (headless osmesa -> mp4)
with the REAL research loop in the loop:

    the ε-pivot agent picks an effort  ->  that effort DRIVES the real G1 actuators
    ->  MuJoCo physics moves the limbs and reports real actuator forces
    ->  that real mechanical work heats/drains the body
    ->  the body's prediction error ε = precision*(body - set-point) grows
    ->  the agent reads ε (bilinear gate, shaped by disposition θ_D) and eases off
    ->  it regulates back toward the set-point; valence = -d||ε||/dt tracks the return.

So the robot moves because it is working, and works less when it needs to come home.
Nothing here is a scripted sine on the skeleton — the motion is physics driven by the policy,
and the body signal is measured from the mechanics, matching the main research.

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

rng = np.random.default_rng(0)
model = mujoco.MjModel.from_xml_path(g1.MJCF_PATH); data = mujoco.MjData(model)
model.vis.global_.offwidth = 720; model.vis.global_.offheight = 560
model.vis.headlight.ambient[:]  = [0.55,0.55,0.55]      # brighten the scene (osmesa is dim)
model.vis.headlight.diffuse[:]  = [0.85,0.85,0.85]
model.vis.headlight.specular[:] = [0.30,0.30,0.30]
NU = model.nu
if model.nkey > 0: mujoco.mj_resetDataKeyframe(model, data, 0)     # standing home pose
home = data.qpos.copy()
mujoco.mj_forward(model, data)
# position-servo target at the standing pose (so oscillations ride ON the stance, not collapse it)
home_ctrl = np.zeros(NU)
for a in range(NU):
    j = model.actuator_trnid[a, 0]
    if j >= 0: home_ctrl[a] = home[model.jnt_qposadr[j]]
renderer = mujoco.Renderer(model, 560, 720)
cam = mujoco.MjvCamera(); cam.distance = 2.05; cam.elevation = -8; cam.lookat[:] = [0, 0, 0.85]
cam.azimuth = 135     # front 3/4 view (shows the chest), gently drifting in the loop
opt = mujoco.MjvOption()
print(f"G1 loaded ({model.nq} dof, {NU} act); osmesa renderer ready")

# ---- REAL physics: drive the actuators at 'effort', keep the pelvis upright, measure the mechanical load ----
def drive(eff, phase, substeps=6):
    """Apply an effort-scaled motion command, step real physics, return the measured mean actuator force."""
    amp = 0.05 + 0.45*eff                                    # harder effort -> larger limb excursions
    f = 0.0
    for t in range(substeps):
        osc = amp*np.sin(phase + np.arange(NU)*0.6 + t*0.5)
        data.ctrl[:] = home_ctrl + osc
        mujoco.mj_step(model, data)
        data.qpos[0:7] = home[0:7]; data.qvel[0:6] = 0.0     # pin the pelvis (no toppling); limbs move freely
        f += float(np.mean(np.abs(data.actuator_force[:NU])))
    mujoco.mj_forward(model, data)
    return f/substeps

# calibrate the real load range once, so we can normalise measured force to [0,1]
_cal = np.array([drive(e, 0.0) for e in np.linspace(0.05, 1.0, 12)])
_L0, _L1 = float(_cal.min()), float(_cal.max())
norm_load = lambda f: float(np.clip((f-_L0)/(_L1-_L0+1e-9), 0, 1))

# ---- body + epsilon dynamics (continuous -> smooth, damped regulation) ----
REST=0.25; SET=np.array([0.25,0.45]); PREC=np.array([1.4,1.0])
class Body:
    def __init__(s): s.temp,s.batt=0.20,1.0
    def apply(s,load):
        s.temp=float(np.clip(s.temp + 0.22*load - 0.11, 0, 1.6))     # heat from REAL work minus passive cooling
        s.batt=float(np.clip(s.batt - 0.055*load + 0.022, 0, 1))     # drain from REAL work minus steady recharge
    def raw(s): return np.array([s.temp,1-s.batt])
    def emag(s): return float(np.linalg.norm(np.clip(PREC*(s.raw()-SET),0,None)))
def controller(b, tD=1.0):
    # ε-pivot policy: high prediction error -> ease off and come home; else do the work. θ_D = disposition.
    return float(np.clip(0.55 - 0.9*tD*b.emag(), 0.05, 0.85))

# ---- pass 1: run the closed loop to get the full trajectory (needed so the graph can grow with it) ----
mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
b=Body(); b.temp,b.batt=1.05,0.35
PERTURB={14:0.55, 30:0.62, 44:0.55}         # the world knocks it off comfort -> it must regulate back, repeatedly
N=58; TR=[]; prev=b.emag()
for i in range(N):
    if i in PERTURB: b.temp=float(min(1.6, b.temp+PERTURB[i]))
    eff=controller(b)                         # policy reads the CURRENT error, picks effort
    load=norm_load(drive(eff, i*0.9))         # effort drives REAL actuators; load MEASURED from mechanics
    b.apply(load); cur=b.emag()
    TR.append((b.temp,b.batt,cur,eff,prev-cur)); prev=cur
TR=np.array(TR); TEMP,BATT,EMAG,EFF,VAL=TR.T
E_caut, E_bold = 0.05, 1.00     # from embodied_epsilon.py temperament test (θ_D 1.8 vs 0.45)

# ---- pass 2: re-run the identical physics deterministically, render each state + draw the dashboard ----
mujoco.mj_resetDataKeyframe(model, data, 0); mujoco.mj_forward(model, data)
frames=[]
for i in range(N):
    eff=EFF[i]
    _=drive(eff, i*0.9)
    cam.azimuth=135+11*np.sin(i*0.13)     # gentle look-around (not a spin), keeps it clearly 3D
    renderer.update_scene(data, camera=cam, scene_option=opt); rgb=renderer.render()

    fig=plt.figure(figsize=(12.2,6.3),dpi=110); fig.patch.set_facecolor("#0c0c0c")
    ax=fig.add_axes([0.01,0.06,0.40,0.86]); ax.imshow(rgb); ax.axis("off")
    ax.text(0.5,1.02,"REST — easing off, coming home" if eff<REST else "working (real torque)",
            transform=ax.transAxes,color=("#7fd1ff" if eff<REST else "#ffd27f"),ha="center",fontsize=13,weight="bold")

    gx=fig.add_axes([0.46,0.52,0.51,0.37]); gx.set_facecolor("#0c0c0c")
    gx.plot(np.arange(i+1),EMAG[:i+1],color="#5ec98a",lw=2.8)
    gx.fill_between([0,N],0,0.35,color="#5ec98a",alpha=0.10)
    gx.axhline(0.35,color="#5ec98a",ls=":",lw=1); gx.text(N*0.55,0.18,"comfort band (set-point)",color="#7fbf8f",fontsize=9)
    gx.set_xlim(0,N); gx.set_ylim(0,1.6); gx.set_title("prediction error  ||ε||  —  driven by REAL measured work, regulated back each time",color="#ddd",fontsize=10.5)
    gx.tick_params(colors="#777"); [sp.set_color("#333") for sp in gx.spines.values()]

    hx=fig.add_axes([0.46,0.40,0.51,0.07]); hx.axis("off"); hx.set_xlim(0,1); hx.set_ylim(0,1)
    hot=np.clip(TEMP[i],0,1)
    hx.add_patch(plt.Rectangle((0.16,0.15),0.62,0.7,fc="#1e1e1e",ec="#444",transform=hx.transAxes))
    hx.add_patch(plt.Rectangle((0.16,0.15),0.62*min(1.0,TEMP[i]),0.7,
                 fc=(min(1.0,0.35+0.6*hot),max(0.0,0.5-0.4*hot),max(0.0,0.8-0.7*hot)),transform=hx.transAxes))
    hx.text(0.14,0.5,"temp",color="#ccc",ha="right",va="center",fontsize=10,transform=hx.transAxes)

    vx=fig.add_axes([0.46,0.12,0.51,0.24]); vx.axis("off")
    v=VAL[i]
    if v>0.004: col,lab="#5ec98a","coming home  ▲"
    elif v<-0.004: col,lab="#e28f8f","knocked off comfort  ▼"
    else: col,lab="#9aa0a6","at set-point — steady"
    vx.text(0,0.86,"valence  = -d||ε||/dt",color="#888",fontsize=11)
    vx.text(0,0.52,f"{v:+.3f}   {lab}",color=col,fontsize=17,weight="bold")
    vx.text(0,0.22,"a feeling as a DYNAMIC — positive while it regulates back to itself",color="#999",fontsize=10)
    vx.text(0,-0.02,f"temperament (same body, different θ_D):  cautious C-3PO → {E_caut:.2f}    bold R2-D2 → {E_bold:.2f}",color="#c9a24d",fontsize=10)
    fig.text(0.5,0.965,"ENIMBLE — the body in the loop: effort → real work → error → come home   (Unitree G1)",color="white",ha="center",fontsize=14.5,weight="bold")
    fig.text(0.5,0.02,"ε-pivot · physics-driven motion · valence as dynamics · slow disposition = temperament · functional, not phenomenal",color="#888",ha="center",fontsize=9)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)

imageio.mimsave("embodied_epsilon.mp4",frames,fps=11,quality=8)
print(f"wrote embodied_epsilon.mp4 ({len(frames)} frames, real physics) | error {EMAG[0]:.2f} -> {EMAG[-1]:.2f} | load range [{_L0:.3f},{_L1:.3f}]")
