"""Kinematic stand-in for Jumper: checks the brain->pad loop steers toward and pounces on prey.
NOT physics. Robot: forward speed 0.5 m/s * (-ly), yaw rate 1.5 rad/s * ang_vel_z(from rx)."""
import argparse, math
import numpy as np
from spiderbrain.config import load_config, with_profile
from spiderbrain.generate import generate
from spiderbrain.robot.loop import BrainController, render_prey_frame


def run(seed=0, profile="tiny", prey=(1.0, 0.6), steps=600, hz=50, verbose=True, readout=None):
    cfg = with_profile(load_config(), profile)
    conn = generate(cfg, seed)
    if readout is None:
        from spiderbrain.sim.readout import fit_ridge_readout
        readout, _ = fit_ridge_readout(conn, cfg, episodes=60, verbose=verbose)
    bc = BrainController(conn, cfg, control_hz=hz, readout=readout, gaze_follow=False)
    x = y = yaw = 0.0
    px, py = prey
    pounced_at = None
    trace = []
    for t in range(steps):
        wx, wy = px + 0.05 * math.sin(t / 5.0), py + 0.05 * math.cos(t / 5.0)   # prey fidgets (lateral eyes see motion)
        dx, dy = wx - x, wy - y
        dist = math.hypot(dx, dy)
        bearing = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi   # + = left
        st, cmd = bc.tick(render_prey_frame(bearing, dist), depth_m=dist, fovea_contrast=1.0 if abs(bearing) < 0.4 else 0.0)
        fwd = -st.ly
        ang = 0.0 if st.rx == 0 else -math.copysign((abs(st.rx) - 0.5) / 0.5, st.rx)   # ang_vel_z = -rx over travel [0.5,1]
        yaw += 1.5 * ang / hz
        x += 0.5 * fwd * math.cos(yaw) / hz
        y += 0.5 * fwd * math.sin(yaw) / hz
        trace.append((t / hz, dist, math.degrees(bearing), fwd, ang, bool(st.A)))
        if st.A and pounced_at is None:
            pounced_at = (t / hz, dist, math.degrees(bearing))
            break
    if verbose:
        for r in trace[::25]:
            print("t=%.2f dist=%.2f bearing=%+.0f fwd=%.2f turn=%+.2f A=%s" % r)
        print("pounce:", pounced_at)
    return trace, pounced_at


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="tiny")
    a = ap.parse_args()
    run(a.seed, a.profile)
