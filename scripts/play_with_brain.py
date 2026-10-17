"""Run Jumper's own `play --app` with the spider brain holding the gamepad.   *** UNTESTED against a live Jumper ***

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
    ap.add_argument("--log", default="brain_log.csv", help="per-tick CSV of what the brain saw and commanded")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
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
            self._bc = BrainController(conn, cfg, control_hz=app.control_hz, brain_hz=50.0, readout=readout, gaze_follow=False)
            super().__init__(app, env, pad=self._bc.pad)       # BrainPad has state() and connected
            self._env_ref, self._t, self._hz, self.last_cmd = env, 0, app.control_hz, {}
            self._pounces = 0
            print(f"[spiderbrain] Jumper control rate {app.control_hz:g} Hz; brain ticks every {self._bc.every} control steps", flush=True)
            self.rows = []
            import atexit
            atexit.register(self._flush)

        def _flush(self):
            import csv
            if not self.rows:
                return
            with open(a.log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "dist_m", "bearing_deg", "turn_cmd", "fwd_cmd", "pounce_cmd", "gate_open", "robot_x", "robot_y", "prey_x", "prey_y", "yaw_deg", "pad_ly", "pad_rx", "pad_A"])
                w.writerows(self.rows)
            print(f"[spiderbrain] wrote {len(self.rows)} ticks to {a.log}; pounce commands with gate open: {self._pounces}", flush=True)

        def _origin(self, u):
            try:
                return [float(v) for v in u.scene.env_origins[0][:2]]
            except Exception:
                return [0.0, 0.0]

        def _place_prey(self, u, x, y):
            import torch
            ball = u.scene[a.prey_entity]
            dev = ball.data.root_link_pos_w.device
            ox, oy = self._origin(u)
            z = float(ball.data.root_link_pos_w[0][2]) if self._t else 0.055
            st = torch.tensor([[ox + x, oy + y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=dev)
            ball.write_root_state_to_sim(st)

        def _person(self, now_us):
            u = self._env_ref.unwrapped
            s = self._t / self._hz
            if a.prey_entity and not a.no_puppet:
                self._place_prey(u, *prey_path(s, a.prey_start))
            if a.prey_entity:
                px, py = [float(v) for v in u.scene[a.prey_entity].data.root_link_pos_w[0][:2]]
            else:
                ox, oy = self._origin(u)
                px, py = (ox, oy)[0] + prey_path(s, a.prey_start)[0], (ox, oy)[1] + prey_path(s, a.prey_start)[1]
            robot = u.scene["robot"]
            rx_, ry_ = [float(v) for v in robot.data.root_link_pos_w[0][:2]]
            yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
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
                      f"gate={'open' if self._bc.pad.gate_ok else 'shut'}", flush=True)
            st = self._bc.pad.state()
            self.rows.append((round(s, 3), round(dist, 3), round(math.degrees(bearing), 1), round(self.last_cmd.get('turn_ccw', 0), 3),
                              round(self.last_cmd.get('forward', 0), 3), int(bool(self.last_cmd.get('pounce'))), int(self._bc.pad.gate_ok),
                              round(rx_, 3), round(ry_, 3), round(px, 3), round(py, 3), round(math.degrees(yaw), 1), round(st.ly, 3), round(st.rx, 3), int(bool(st.A))))
            self._t += 1
            return super()._person(now_us)                       # reads self._bc.pad.state()

    app_play.AppPlayer = BrainAppPlayer
    sys.argv = ["play.py"] + rest
    os.chdir(a.jumper)
    runpy.run_path(str(a.jumper / "scripts" / "play.py"), run_name="__main__")


if __name__ == "__main__":
    main()
