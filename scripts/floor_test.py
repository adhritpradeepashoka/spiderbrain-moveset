"""Scripted pad test on one Jumper scene: stand -> walk -> stop -> crouch -> walk crouched -> raise.   (UNTESTED until run)

    python scripts/floor_test.py --jumper /kaggle/working/jumper --log ice.csv -- --app out/bundle_example/jumper.app --scene ice --headless --steps 2000 --video ice.mp4

Everything after `--` goes to Jumper's scripts/play.py. No brain: this only asks what Jumper's own walk and body-height
controls do on each floor, so a behaviour layer can be designed on measured facts.
"""
import argparse, math, os, runpy, sys
from pathlib import Path

# (end_time_s, ly, ry, R3_held)   ly<0 = forward, ry>0 = stick down (lowers body while R3 is held)
PLAN = [(0.5, 0, 0, False), (3.5, -1, 0, False), (4.0, 0, 0, False), (5.2, 0, 1, True), (5.6, 0, 0, False),
        (8.1, -1, 0, False), (8.4, 0, 0, False), (9.6, 0, -1, True), (10.5, 0, 0, False)]
LABEL = ["stand", "walk", "stop", "lower", "crouched", "walk_crouched", "stop2", "raise", "stand2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True, type=Path)
    ap.add_argument("--log", default="floor_log.csv")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [r for r in a.rest if r != "--"]
    sys.path[:0] = [str(a.jumper), str(a.jumper / "scripts"), str(a.jumper / "rl"), str(Path(__file__).resolve().parents[1] / "src")]
    from spiderbrain.robot.bridge import PadState
    from mjrl import app_play
    Orig = app_play.AppPlayer

    class Pad:
        connected = True
        def __init__(self): self.s = PadState()
        def state(self): return self.s

    def seg(t):
        for i, row in enumerate(PLAN):
            if t < row[0]:
                return i, row
        return len(PLAN) - 1, PLAN[-1]

    class FloorPlayer(Orig):
        def __init__(self, app, env, *, pad=None):
            self._pad = Pad()
            super().__init__(app, env, pad=self._pad)
            self._env_ref, self._t, self._hz, self.rows = env, 0, app.control_hz, []
            import atexit; atexit.register(self._flush)

        def _flush(self):
            import csv
            with open(a.log, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t_s", "phase", "x", "y", "z", "yaw_deg", "roll_deg", "pitch_deg"]); w.writerows(self.rows)
            print(f"[floor] wrote {len(self.rows)} rows to {a.log}", flush=True)

        def _person(self, now_us):
            u = self._env_ref.unwrapped
            r = u.scene["robot"]
            x, y, z = [float(v) for v in r.data.root_link_pos_w[0][:3]]
            w_, qx, qy, qz = [float(v) for v in r.data.root_link_quat_w[0]]
            roll = math.degrees(math.atan2(2 * (w_ * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
            pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w_ * qy - qz * qx)))))
            yaw = math.degrees(math.atan2(2 * (w_ * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            s = self._t / self._hz
            i, (_, ly, ry, r3) = seg(s)
            self._pad.s = PadState(ly=float(ly), ry=float(ry), buttons=frozenset({"R3"}) if r3 else frozenset())
            self.rows.append((round(s, 3), LABEL[i], round(x, 4), round(y, 4), round(z, 4), round(yaw, 1), round(roll, 1), round(pitch, 1)))
            self._t += 1
            return super()._person(now_us)

    app_play.AppPlayer = FloorPlayer
    sys.argv = ["play.py"] + rest
    os.chdir(a.jumper)
    runpy.run_path(str(a.jumper / "scripts" / "play.py"), run_name="__main__")


if __name__ == "__main__":
    main()
