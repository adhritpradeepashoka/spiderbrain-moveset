"""Replay the stock jump's commands, with and without the forward bias, in plain MuJoCo on the CPU (no GPU, no policy).

    python scripts/directional_jump_check.py --jumper /path/to/jumper

Reports forward/sideways travel, rise and final tilt at kp = 20, 15, 10. It is a *different simulator* from Jumper's own
(no torque-speed curve, a plain floor, MuJoCo CPU) so use it to compare variants, then confirm with
`scripts/play_with_brain.py --jump-style directional` on Jumper's own sim.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumper", required=True, type=Path)
    ap.add_argument("--friction", type=float, default=None, help="cap every geom's sliding friction at this value")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import mujoco
    from spiderbrain.robot.directional_jump import biased_commands, KD

    J = a.jumper
    spec = mujoco.MjSpec.from_file(str(J / "assets/jumper/jumper.xml"))
    g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [5, 5, 0.1]; g.friction = [1.0, 0.005, 0.0001]
    m = spec.compile()
    if a.friction:
        m.geom_friction[:, 0] = np.minimum(m.geom_friction[:, 0], a.friction)
    ref = dict(np.load(J / "tasks/jumper/jump/ref/high_jump_flat.npz", allow_pickle=True))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
    order = json.loads((J / "out/bundle_example/jumper/high_jump_flat.trajectory.json").read_text())["joint_order"]
    assert order == names, "joint order differs from the recording's"
    qadr = [m.jnt_qposadr[i] for i in range(1, m.njnt)]
    vadr = [m.jnt_dofadr[i] for i in range(1, m.njnt)]

    def run(cmd, kp):
        d = mujoco.MjData(m)
        d.qpos[:3], d.qpos[3:7] = ref["base_pos"][0], ref["base_quat"][0]
        d.qpos[qadr] = ref["q"][0]
        mujoco.mj_forward(m, d)
        x0, y0, z0 = d.qpos[:3].copy()
        zmax = z0
        for s in range(int(2.2 / m.opt.timestep)):
            i = min(int(s * m.opt.timestep * 200.0), len(cmd) - 1)
            d.qfrc_applied[vadr] = np.clip(kp * (cmd[i] - d.qpos[qadr]) - KD * d.qvel[vadr], -1.7464, 1.7464)
            mujoco.mj_step(m, d)
            zmax = max(zmax, d.qpos[2])
        w, x, y, z = d.qpos[3:7]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        return d.qpos[0] - x0, d.qpos[1] - y0, zmax - z0, d.qpos[2], tilt

    for label, scale in (("stock commands", 0.0), ("forward bias", 1.0)):
        cmd = biased_commands(ref["q_cmd"], names, scale=scale)
        for kp in (20.0, 15.0, 10.0):
            fx, sy, rise, zend, tilt = run(cmd, kp)
            print(f"{label:15s} kp={kp:4.0f}: forward {fx:+.3f} m  sideways {sy:+.3f} m  rise {rise:.3f} m  end z {zend:.3f}  tilt {tilt:.1f} deg")


if __name__ == "__main__":
    main()
