"""CPU MuJoCo model of Jumper for open-loop jump experiments.

Reads ONLY data from a Jumper checkout (assets/jumper/jumper.xml, tasks/jumper/jump/ref/high_jump_flat.npz) and
never imports or runs its code. The MJCF has no actuators, so the servos are re-implemented here from the numbers in
Jumper's own files: PD with kp, kd and a torque cap with a speed envelope.

CAVEAT: this is my approximation of Jumper's servo model (torque-speed curve simplified to a straight line from the
corner speed to the cap speed, no thermal derating, no latency). It is NOT mjlab's simulator. Treat results as
feasibility hints, to be confirmed in Jumper's own simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


class JumperPhys:
    def __init__(self, jumper_root, kp=20.0, kd=0.5, tau_plateau=1.7464, corner_rpm=293.5, cap_rpm=611.0, sub=1):
        root = Path(jumper_root)
        spec = mujoco.MjSpec.from_file(str(root / "assets/jumper/jumper.xml"))
        # the MJCF has no ground (mjlab adds terrain at run time), so add a plain plane
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], friction=[1.0, 0.005, 0.0001])
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        ref = np.load(root / "tasks/jumper/jump/ref/high_jump_flat.npz", allow_pickle=False)
        self.ref = {k: ref[k] for k in ref.files if k != "meta"}
        self.meta = json.loads(str(ref["meta"]))
        self.joints = self.meta["joint_order"]
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in self.joints])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in self.joints])
        self.lo = np.array([self.m.joint(j).range[0] for j in self.joints])
        self.hi = np.array([self.m.joint(j).range[1] for j in self.joints])
        self.kp, self.kd, self.P = kp, kd, tau_plateau
        self.w_corner, self.w_cap = corner_rpm * 2 * np.pi / 60, cap_rpm * 2 * np.pi / 60
        self.dt = self.m.opt.timestep
        # floor geoms: world-body geoms
        self.floor = {i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == 0}
        self.base_body = self.m.body("base_link").id if self._has_body("base_link") else 1
        self.rec_hz = self.meta["rec_hz"]
        self.cmd_hz = self.meta["control_hz"]   # q_cmd is sampled at the controller rate (200 Hz), not the 250 Hz recording rate

    def _has_body(self, n):
        try:
            self.m.body(n); return True
        except KeyError:
            return False

    def reset(self):
        d, ref = self.d, self.ref
        mujoco.mj_resetData(self.m, d)
        d.qpos[:3] = ref["base_pos"][0]
        d.qpos[3:7] = ref["base_quat"][0]
        d.qpos[self.qadr] = ref["q"][0]
        mujoco.mj_forward(self.m, d)

    def torque_limit(self, w):
        aw = np.abs(w)
        env = np.clip(1.0 - (aw - self.w_corner) / (self.w_cap - self.w_corner), 0.0, 1.0)
        return self.P * np.where(aw <= self.w_corner, 1.0, env)

    def airborne(self):
        d = self.d
        n = d.ncon
        if n == 0:
            return True
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        return not np.any((g1 == 0) | (g2 == 0))     # geom 0 is the added floor

    def rollout(self, qcmd, hold_s=0.6, record=False):
        """qcmd: (N,22) targets at rec_hz. After the end, the last target is held for hold_s. Returns metrics (+traj)."""
        self.reset()
        d = self.d
        N = len(qcmd)
        T = N / self.cmd_hz + hold_s
        steps = int(T / self.dt)
        times = np.arange(N) / self.cmd_hz
        base0 = d.qpos[:3].copy()
        # heading of the base at start (yaw) to express distance along the initial forward axis
        w, x, y, z = d.qpos[3:7]
        yaw0 = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        fwd, left = np.array([np.cos(yaw0), np.sin(yaw0)]), np.array([-np.sin(yaw0), np.cos(yaw0)])
        traj, flight, was_air, t_to, t_td, x_to, x_td, peak = [], 0.0, False, None, None, None, None, d.qpos[2]
        min_z, max_tilt = 9, 0.0
        tt = np.arange(steps) * self.dt
        TG = np.stack([np.interp(tt, times, qcmd[:, j]) for j in range(22)], 1)
        for i in range(steps):
            t = tt[i]
            tgt = TG[i]
            q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
            tau = self.kp * (tgt - q) - self.kd * qd
            lim = self.torque_limit(qd)
            d.qfrc_applied[self.dadr] = np.clip(tau, -lim, lim)
            mujoco.mj_step(self.m, d)
            air = self.airborne()
            if air and not was_air and t > 0.05:
                t_to, x_to = t, d.qpos[:2].copy()
            if (not air) and was_air:
                t_td, x_td = t, d.qpos[:2].copy()
            if air:
                flight += self.dt
            was_air = air
            peak = max(peak, d.qpos[2])
            min_z = min(min_z, d.qpos[2])
            qw, qx, qy, qz = d.qpos[3:7]
            tilt = np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))
            max_tilt = max(max_tilt, tilt)
            if record and i % 5 == 0:
                traj.append(np.r_[t, d.qpos[:7], air])
            if not np.isfinite(d.qpos).all():
                break
        disp = d.qpos[:2] - base0[:2]
        out = dict(rise=peak - base0[2], peak_z=peak, flight_s=flight, end_fwd=float(disp @ fwd), end_side=float(disp @ left),
                   end_tilt=float(tilt), max_tilt=float(max_tilt), end_z=float(d.qpos[2]), min_z=float(min_z),
                   td_fwd=None if x_td is None else float((x_td - base0[:2]) @ fwd),
                   to_fwd=None if x_to is None else float((x_to - base0[:2]) @ fwd))
        return (out, np.array(traj)) if record else out
