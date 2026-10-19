# Directional (forward) jump

**Status: shown in a plain-MuJoCo CPU replay only. Not yet confirmed on Jumper's own simulator.**

## What it is
Jumper's stock jump is a trained policy tracking a symmetric recording (`tasks/jumper/jump/ref/high_jump_flat.npz`), so it
goes straight up. Replaying the recording's commanded joint angles open-loop (PD kp=20, kd=0.5, torque clip 1.7464 N*m)
reproduces that jump in plain MuJoCo: 0.157 m rise vs 0.141 m recorded, 0 cm travel, lands upright.

Adding a Gaussian bump (centre 0.72 s, sigma 0.08 s) to six joint pairs - rear-leg hip yaw / hip / knee, middle-leg
hip / knee, front-arm shoulder pitch - turns the same motion into a forward leap. Right side = mirror of left. Values are in
`src/spiderbrain/robot/directional_jump.py`; found by a small evolution-strategy search (CPU, ~2 min, 2 cores).

## Numbers (`scripts/directional_jump_check.py`, floor friction 1.0)
| | kp 20 | kp 15 | kp 10 (real robot) |
|---|---|---|---|
| stock commands | 0.0 m fwd, 0.157 m rise | -0.02 m, 0.139 m | -0.05 m, 0.093 m |
| forward bias | **+0.25 m fwd**, 0.082 m rise | +0.22 m | +0.17 m |

Sideways drift < 1 cm and final tilt < 0.5 deg in all rows. All commands stay inside the joint limits.

## Caveats (read these before claiming anything)
- Different simulator from Jumper's (no torque-speed curve; own floor). One search, one reference motion.
- Slippery floors defeat it: with every geom's friction capped at 0.4 the forward travel drops to 4-6 cm. Fits the ice result
  from `docs/floor-test-results.md`.
- Open-loop: no feedback, so a push or a bump mid-jump is not corrected. Landing on uneven ground is untested.
- The 0.25 m is straight ahead only; left/right jumps would need turning first (the stalk phase already does that).
- Real hardware (kp=10 and servo limits) not tested; 0.17 m is the kp=10 figure in this simulator.

## Use
    python scripts/directional_jump_check.py --jumper <jumper checkout>
    python scripts/play_with_brain.py --jumper <jumper> --jump-style directional --prey-entity ball -- --app out/bundle_example/jumper.app --scene football --headless --steps 2100 --video clip.mp4
