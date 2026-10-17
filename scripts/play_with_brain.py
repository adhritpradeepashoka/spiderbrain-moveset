"""Run Jumper's own `play --app` with the spider brain holding the gamepad.   *** UNTESTED against a live Jumper ***

    python scripts/play_with_brain.py --jumper /path/to/jumper --seed 0 -- \\
        --app out/bundle_<ts>/jumper.app --scene football --prey-entity ball --headless --steps 1500 --video clip.mp4

Everything after `--` goes to Jumper's scripts/play.py unchanged. We patch mjrl.app_play.AppPlayer so each control tick
(1) reads the robot pose and a prey position, (2) renders a synthetic camera frame of the prey, (3) steps the brain,
(4) hands the resulting pad state to the controller. Jumper's policies do all joint control.
Read Jumper's code before running it: this is third-party code (Apache-2.0), not vendored here.
"""
import argparse, math, os, runpy, sys
from pathlib import Path


def yaw_from_quat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True, type=Path, help="path to a KingKongRobotics/jumper checkout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--prey-entity", default=None, help="scene entity whose position is the prey (e.g. the football scene's ball)")
    ap.add_argument("--prey-xy", type=float, nargs=2, default=(1.0, 0.6), help="fixed virtual prey position if no entity is given")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [r for r in a.rest if r != "--"]

    sys.path[:0] = [str(a.jumper), str(a.jumper / "rl"), str(Path(__file__).resolve().parents[1] / "src")]
    from spiderbrain.config import load_config, with_profile
    from spiderbrain.generate import generate
    from spiderbrain.robot.loop import BrainController, render_prey_frame
    from spiderbrain.sim.readout import fit_ridge_readout

    cfg = with_profile(load_config(), a.profile)
    conn = generate(cfg, a.seed)
    readout, r2 = fit_ridge_readout(conn, cfg, episodes=60)
    print(f"[spiderbrain] seed {a.seed}: readout held-out R^2 turn/forward = {r2[0]:.2f}/{r2[1]:.2f}")

    from mjrl import app_play
    Orig = app_play.AppPlayer

    class BrainAppPlayer(Orig):
        def __init__(self, app, env, *, pad=None):
            self._bc = BrainController(conn, cfg, control_hz=app.control_hz, readout=readout, gaze_follow=False)
            super().__init__(app, env, pad=self._bc.pad)       # BrainPad has state() and connected
            self._env_ref = env
            self.last_cmd = {}

        def _prey_xy(self, scene):
            if a.prey_entity:
                return [float(v) for v in scene[a.prey_entity].data.root_link_pos_w[0][:2]]
            return a.prey_xy

        def _person(self, now_us):
            u = self._env_ref.unwrapped
            robot = u.scene["robot"]
            px, py = self._prey_xy(u.scene)
            rx_, ry_ = [float(v) for v in robot.data.root_link_pos_w[0][:2]]
            yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w[0])
            dx, dy = px - rx_, py - ry_
            dist = math.hypot(dx, dy)
            bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi      # + = left (CCW)
            _, self.last_cmd = self._bc.tick(render_prey_frame(bearing, dist), depth_m=dist,
                                             fovea_contrast=1.0 if abs(bearing) < 0.4 else 0.0)
            return super()._person(now_us)                       # reads self._bc.pad.state()

    app_play.AppPlayer = BrainAppPlayer
    sys.argv = ["play.py"] + rest
    os.chdir(a.jumper)
    runpy.run_path(str(a.jumper / "scripts" / "play.py"), run_name="__main__")


if __name__ == "__main__":
    main()
