"""Run Jumper's own `play --app` with the spider brain holding the gamepad.   (stock mode ran on a live Jumper sim; --jump-style directional is NEW, untested there)

    python scripts/play_with_brain.py --jumper /content/jumper --seed 0 --prey-entity ball -- \\
        --app out/bundle_example/jumper.app --scene football --headless --steps 1500 --video clip.mp4

Everything after `--` goes to Jumper's scripts/play.py unchanged. We patch mjrl.app_play.AppPlayer so each control tick:
  1. the prey (the football scene's ball, puppeted along a slow wandering path) is placed,
  2. robot pose + prey position -> bearing/distance -> a synthetic camera frame of a blob,
  3. frame -> eyes -> frozen random spider brain -> linear readout -> pad state,
  4. Jumper's controller reads that pad. Jumper's own policies do all joint control.
Third-party code (Jumper, Apache-2.0) is imported, not vendored. Read it before running it.
"""
import argparse, math, os, runpy, sys
import numpy as np
from pathlib import Path


def _bootstrap():
    """Install this repo's few dependencies into whatever Python is running us, and say plainly if that fails."""
    import importlib, subprocess
    need = {"yaml": "pyyaml", "pyarrow": "pyarrow", "scipy": "scipy", "pandas": "pandas", "numpy": "numpy"}
    missing = []
    for mod, pipname in need.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pipname)
    if not missing:
        return
    print(f"[spiderbrain] python={sys.executable}; installing missing: {missing}", flush=True)
    import shutil
    uv = shutil.which("uv")
    if uv:        # a uv-made venv has no pip
        r = subprocess.run([uv, "pip", "install", "--python", sys.executable, *missing], capture_output=True, text=True)
    else:
        r = subprocess.run([sys.executable, "-m", "pip", "install", *missing], capture_output=True, text=True)
    if r.returncode and not uv:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", *missing], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-1500:], flush=True)
    if r.returncode:
        raise SystemExit(f"[spiderbrain] could not install {missing} into {sys.executable}; see pip output above")
    importlib.invalidate_caches()


def yaw_from_quat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def prey_path(s, start):
    """Slow wander around `start` (metres, seconds). Lateral eyes need motion, and a moving ball films better."""
    return (start[0] + 0.25 * math.sin(0.5 * s), start[1] + 0.15 * math.sin(0.9 * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True, type=Path, help="path to a KingKongRobotics/jumper checkout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--prey-entity", default=None, help="scene entity to use as prey (football scene: 'ball')")
    ap.add_argument("--prey-start", type=float, nargs=2, default=(1.2, 0.5), help="prey start, metres from the robot's start")
    ap.add_argument("--no-puppet", action="store_true", help="do not move the prey entity; just read its position")
    ap.add_argument("--fidget-deg", type=float, default=3.0, help="apparent prey jitter (lateral eyes are motion detectors)")
    ap.add_argument("--jump-style", choices=["stock", "directional"], default="stock",
                    help="stock: press A, Jumper's own vertical jump. directional: our forward-biased jump drives the joints")
    ap.add_argument("--jump-params", type=Path, default=None, help="JSON bias (e.g. configs/directional_jump_v3.json); default for --jump-style directional = configs/directional_jump_v4c.json")
    ap.add_argument("--jump-scale", type=float, default=1.0, help="0 = the stock jump's commands replayed open-loop, 1 = full forward bias")
    ap.add_argument("--instinct", choices=["hunt", "threat", "tun", "court", "ice"], default=None,
                    help="run the survival-crab agent in this scripted scenario (hunt/threat/tun/court need the football scene's ball)")
    ap.add_argument("--instinct-log", default="instinct_log.csv", help="per-tick CSV of the agent's state and drives (for the HUD overlay)")
    ap.add_argument("--log", default="brain_log.csv", help="per-tick CSV of what the brain saw and commanded")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.jump_params is None and a.jump_style == "directional":
        a.jump_params = Path(__file__).resolve().parent.parent / "configs" / "directional_jump_v4c.json"   # best verified in Jumper's sim
    if a.jump_params:
        a.jump_params = a.jump_params.resolve()      # the script chdirs into the Jumper checkout later
    _bootstrap()
    rest = [r for r in a.rest if r != "--"]

    sys.path[:0] = [str(a.jumper), str(a.jumper / "scripts"), str(a.jumper / "rl"), str(Path(__file__).resolve().parents[1] / "src")]
    from spiderbrain.config import load_config, with_profile
    from spiderbrain.generate import generate
    from spiderbrain.robot.loop import BrainController, render_prey_frame
    from spiderbrain.sim.readout import fit_ridge_readout

    cfg = with_profile(load_config(), a.profile)
    conn = generate(cfg, a.seed)
    readout, r2 = fit_ridge_readout(conn, cfg, episodes=60)
    print(f"[spiderbrain] seed {a.seed}: readout held-out R^2 turn/forward = {r2[0]:.2f}/{r2[1]:.2f}", flush=True)

    from mjrl import app_play
    Orig = app_play.AppPlayer

    class BrainAppPlayer(Orig):
        def __init__(self, app, env, *, pad=None):
            self._bc = BrainController(conn, cfg, control_hz=app.control_hz, brain_hz=50.0, readout=readout, gaze_follow=False, drive_pad=not a.instinct)
            super().__init__(app, env, pad=self._bc.pad)       # BrainPad has state() and connected
            self._env_ref, self._t, self._hz, self.last_cmd = env, 0, app.control_hz, {}
            self._pounces = 0
            self._bc.pad.jump_style = a.jump_style
            self._dj, self._dj_start = None, None
            if a.jump_style == "directional":
                import json as _json
                from spiderbrain.robot.directional_jump import DirectionalJump
                man = _json.loads((app.root / "bundle.json").read_text("utf-8"))
                traj = _json.loads((app.root / man["modes"]["jump"]["trajectory"]).read_text("utf-8"))
                self._dj = DirectionalJump(np.asarray(traj["q_cmd"]), traj["joint_order"], params=(_json.loads(a.jump_params.read_text()) if a.jump_params else None), scale=a.jump_scale)
                print(f"[spiderbrain] directional jump armed: {len(traj['q_cmd'])} commands, scale {a.jump_scale}", flush=True)
            print(f"[spiderbrain] Jumper control rate {app.control_hz:g} Hz; brain ticks every {self._bc.every} control steps", flush=True)
            self.rows, self.irows = [], []
            if a.instinct:
                from spiderbrain.instinct.agent import FrictionEstimator
                from spiderbrain.instinct.scenarios import World, make_agent
                self._agent, self._world, self._fric = make_agent(a.instinct, a.seed), World(a.instinct, a.seed), FrictionEstimator()
                self._intent, self._caught, self._njumps, self._padcmd = None, False, 0, {"turn_ccw": 0.0, "forward": 0.0, "pounce": False}
                print(f"[spiderbrain] INSTINCT scenario '{a.instinct}', personality {self._agent.p}", flush=True)
            import atexit
            atexit.register(self._flush)

        def _flush(self):
            import csv
            if self.irows:
                with open(a.instinct_log, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t_s", "state", "why", "energy", "fear", "mate", "viability", "shadow", "metabolism", "stress", "grip", "role", "ball_x", "ball_y",
                                "robot_x", "robot_y", "yaw_deg", "pad_phase", "loom", "crouched", "prey_dist", "threat_dist", "mate_dist"])
                    w.writerows(self.irows)
                print(f"[spiderbrain] wrote {len(self.irows)} instinct ticks to {a.instinct_log}; events: {self._agent.events}", flush=True)
            if not self.rows:
                return
            with open(a.log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "dist_m", "bearing_deg", "turn_cmd", "fwd_cmd", "pounce_cmd", "gate_open", "robot_x", "robot_y", "prey_x", "prey_y", "yaw_deg", "search", "pad_ly", "pad_rx", "pad_A"])
                w.writerows(self.rows)
            print(f"[spiderbrain] wrote {len(self.rows)} ticks to {a.log}; pounce commands with gate open: {self._pounces}", flush=True)

        def __call__(self, obs):
            out = super().__call__(obs)
            if self._dj is not None:
                self._drive_directional_jump()
            return out

        def _drive_directional_jump(self):
            """After Jumper's controller has written its targets, overwrite them while our jump plays."""
            import torch
            now_s = (self._tick - 1) * self._step_us / 1e6
            data = self._robot.data
            if self._bc.pad.jump_request and not self._dj.active:
                self._bc.pad.jump_request = False
                wire = [list(self._robot.joint_names)[i] for i in self._wire_from_sim]
                row = data.joint_pos[0]
                cur = np.array([float(row[self._wire_from_sim[wire.index(nm)]]) for nm in self._dj.names])
                self._dj.start(now_s, cur)
                self._dj_start = [float(v) for v in data.root_link_pos_w[0][:3]] + [yaw_from_quat_wxyz(data.root_link_quat_w[0])]
                print(f"\n[spiderbrain] t={now_s:.2f}s DIRECTIONAL JUMP start at ({self._dj_start[0]:.2f},{self._dj_start[1]:.2f})", flush=True)
            if self._dj_start is not None:
                self._dj_zmax = max(getattr(self, "_dj_zmax", 0.0) if self._dj.active else 0.0, float(data.root_link_pos_w[0][2]))
            tgt = self._dj.target(now_s)
            if tgt is None:
                if self._dj_start is not None:
                    p = [float(v) for v in data.root_link_pos_w[0][:3]]
                    ddx, ddy = p[0] - self._dj_start[0], p[1] - self._dj_start[1]
                    fwd = ddx * math.cos(self._dj_start[3]) + ddy * math.sin(self._dj_start[3])
                    side = -ddx * math.sin(self._dj_start[3]) + ddy * math.cos(self._dj_start[3])
                    print(f"\n[spiderbrain] directional jump done: forward {fwd:+.3f} m, sideways {side:+.3f} m, z {p[2]:.3f}, peak z {getattr(self, '_dj_zmax', 0.0):.3f} (start z {self._dj_start[2]:.3f})", flush=True)
                    self.jump_results = getattr(self, "jump_results", []) + [(round(now_s, 2), round(fwd, 3), round(side, 3))]
                    self._dj_start = None
                return
            from spiderbrain.robot.directional_jump import KP, KD
            names = list(self._robot.joint_names)
            wire = [names[i] for i in self._wire_from_sim]
            vals = [float(tgt[self._dj.names.index(nm)]) for nm in wire]
            n = self._env.unwrapped.num_envs
            dev = self._env.unwrapped.device
            self._targets[:, self._sim_ids] = torch.tensor(vals, device=dev).repeat(n, 1)
            for act, idx in self._gains:
                act.set_gains(slice(None), kp=torch.full((n, len(idx)), KP, device=dev), kd=torch.full((n, len(idx)), KD, device=dev))

        def _origin(self, u):
            try:
                return [float(v) for v in u.scene.env_origins[0][:2]]
            except Exception:
                return [0.0, 0.0]

        def _place_prey(self, u, x, y):
            """x, y are WORLD coordinates."""
            import torch
            ball = u.scene[a.prey_entity]
            dev = ball.data.root_link_pos_w.device
            z = float(ball.data.root_link_pos_w[0][2]) if self._t else 0.055
            st = torch.tensor([[x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=dev)
            ball.write_root_state_to_sim(st)

        def _person_instinct(self, now_us):
            """Survival-crab mode: scenario world -> percepts -> SurvivalAgent -> pad. The spider brain still runs and supplies the hunt."""
            from spiderbrain.instinct.agent import Percept
            from spiderbrain.instinct.scenarios import angular_size, bearing_to
            u = self._env_ref.unwrapped
            s = self._t / self._hz
            dt = 1.0 / self._hz
            robot = u.scene["robot"]
            rx_, ry_ = [float(v) for v in robot.data.root_link_pos_w[0][:2]]
            yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
            if self._t == 0:
                self._x0, self._y0, self._yaw0 = rx_, ry_, yaw
                print(f"[spiderbrain] robot starts at ({rx_:.2f},{ry_:.2f}) facing {math.degrees(yaw):+.0f} deg", flush=True)
            c0, s0 = math.cos(self._yaw0), math.sin(self._yaw0)
            ix, iy = (rx_ - self._x0) * c0 + (ry_ - self._y0) * s0, -(rx_ - self._x0) * s0 + (ry_ - self._y0) * c0
            iyaw = (yaw - self._yaw0 + math.pi) % (2 * math.pi) - math.pi
            pose = (ix, iy, iyaw)
            jr = getattr(self, "jump_results", [])
            landed = len(jr) > self._njumps
            self._njumps = len(jr)
            w = self._world.step(s, dt, pose, caught=self._caught)
            self._caught = False
            if a.prey_entity and not a.no_puppet:
                self._place_prey(u, self._x0 + w.ball[0] * c0 - w.ball[1] * s0, self._y0 + w.ball[0] * s0 + w.ball[1] * c0)
            dist, bearing = bearing_to(pose, w.ball)
            visible = w.role != "none" and abs(bearing) < math.radians(65) and dist < 3.0
            pad = self._bc.pad
            if self._t % self._bc.every == 0:
                walking_ok = pad.phase == "stalk" and pad._h_move is None
                grip = self._fric.update(s, rx_, ry_, float(self._padcmd.get("forward", 0.0)), walking_ok)
                pc = Percept(t=s, stress=w.stress, friction=grip, jump_ready=(pad.phase == "stalk"), mate_receptive=w.mate_receptive)
                if visible and w.role == "prey":
                    pc.prey_dist, pc.prey_bearing = dist, bearing
                    pc.prey_caught = landed and dist < 0.30
                elif visible and w.role == "threat":
                    pc.threat_dist, pc.threat_bearing, pc.threat_theta = dist, bearing, angular_size(dist)
                elif visible and w.role == "mate":
                    pc.mate_dist, pc.mate_bearing = dist, bearing
                if pc.prey_caught:
                    self._caught = True
                # the spider brain only ever sees prey; threats and mates are the agent's business
                frame = render_prey_frame(bearing + math.radians(a.fidget_deg) * math.sin(self._t / 5.0), dist, size_m=0.11) if (visible and w.role == "prey") else np.zeros((120, 160, 3), np.float32)
                _, self.last_cmd = self._bc.tick(frame, depth_m=dist, fovea_contrast=1.0 if abs(bearing) < 0.4 else 0.0)
                i = self._agent.tick(pc, self._bc.every / self._hz, brain_cmd=self.last_cmd, pad_phase=pad.phase)
                self._intent = i
                self._padcmd = {"turn_ccw": i.turn_ccw, "forward": i.forward * i.speed_scale, "strafe_left": i.strafe_left, "pounce": i.pounce,
                                "escape": i.state == "flee_jump", "crouch": i.crouch, "gesture": i.gesture}
                self._gate = (dist if (visible and w.role == "prey") else None, 1.0 if abs(bearing) < 0.4 else 0.0)
                d = self._agent.drives()
                self.irows.append((round(s, 3), i.state, i.why, d["energy"], d["fear"], d["mate"], d["viability"], d["shadow"], d["metabolism"],
                                   round(w.stress, 3), round(grip, 3), w.role, round(w.ball[0], 3), round(w.ball[1], 3), round(ix, 3), round(iy, 3),
                                   round(math.degrees(iyaw), 1), pad.phase, round(self._agent.last_loom, 3), int(pad.crouched),
                                   round(pc.prey_dist, 3) if pc.prey_dist is not None else "", round(pc.threat_dist, 3) if pc.threat_dist is not None else "",
                                   round(pc.mate_dist, 3) if pc.mate_dist is not None else ""))
                if self._t % int(self._hz) == 0:
                    print(f"[spiderbrain] t={s:5.1f}s {i.state:10s} {i.why[:44]:44s} E={d['energy']:.2f} F={d['fear']:.2f} V={d['viability']:.2f} grip={grip:.2f} phase={pad.phase}", flush=True)
            dist_g, fov_g = getattr(self, "_gate", (None, None))
            pad.update(self._padcmd, dt, distance_m=dist_g, fovea_contrast=fov_g)
            self._t += 1
            return super()._person(now_us)

        def _person(self, now_us):
            if a.instinct:
                return self._person_instinct(now_us)
            u = self._env_ref.unwrapped
            s = self._t / self._hz
            robot = u.scene["robot"]
            rx_, ry_ = [float(v) for v in robot.data.root_link_pos_w[0][:2]]
            yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
            if self._t == 0:                         # the prey lives in the robot's INITIAL frame (+x ahead, +y left)
                self._x0, self._y0, self._yaw0 = rx_, ry_, yaw
                print(f"[spiderbrain] robot starts at ({rx_:.2f},{ry_:.2f}) facing {math.degrees(yaw):+.0f} deg", flush=True)
            fx, fy = prey_path(s, a.prey_start)
            wx = self._x0 + fx * math.cos(self._yaw0) - fy * math.sin(self._yaw0)
            wy = self._y0 + fx * math.sin(self._yaw0) + fy * math.cos(self._yaw0)
            if a.prey_entity and not a.no_puppet:
                self._place_prey(u, wx, wy)
            if a.prey_entity:
                px, py = [float(v) for v in u.scene[a.prey_entity].data.root_link_pos_w[0][:2]]
            else:
                px, py = wx, wy
            dx, dy = px - rx_, py - ry_
            dist = math.hypot(dx, dy)
            bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi      # + = left (CCW)
            seen = bearing + math.radians(a.fidget_deg) * math.sin(self._t / 5.0)
            _, self.last_cmd = self._bc.tick(render_prey_frame(seen, dist, size_m=0.11), depth_m=dist,
                                             fovea_contrast=1.0 if abs(bearing) < 0.4 else 0.0)
            if self.last_cmd.get("pounce") and self._bc.pad.gate_ok:
                self._pounces += 1
            if self._t % int(self._hz) == 0:
                print(f"[spiderbrain] t={s:5.1f}s dist={dist:.2f} bearing={math.degrees(bearing):+4.0f} "
                      f"turn={self.last_cmd.get('turn_ccw', 0):+.2f} fwd={self.last_cmd.get('forward', 0):.2f} "
                      f"gate={'open' if self._bc.pad.gate_ok else 'shut'}{' SEARCH' if self.last_cmd.get('search') else ''}", flush=True)
            st = self._bc.pad.state()
            self.rows.append((round(s, 3), round(dist, 3), round(math.degrees(bearing), 1), round(self.last_cmd.get('turn_ccw', 0), 3),
                              round(self.last_cmd.get('forward', 0), 3), int(bool(self.last_cmd.get('pounce'))), int(self._bc.pad.gate_ok),
                              round(rx_, 3), round(ry_, 3), round(px, 3), round(py, 3), round(math.degrees(yaw), 1), int(bool(self.last_cmd.get('search'))), round(st.ly, 3), round(st.rx, 3), int(bool(st.A))))
            self._t += 1
            return super()._person(now_us)                       # reads self._bc.pad.state()

    app_play.AppPlayer = BrainAppPlayer
    sys.argv = ["play.py"] + rest
    os.chdir(a.jumper)
    runpy.run_path(str(a.jumper / "scripts" / "play.py"), run_name="__main__")


if __name__ == "__main__":
    main()
