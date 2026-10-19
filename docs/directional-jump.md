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

## Verified in Jumper's own simulator (football scene, 1100 steps, kp 20 / kd 0.5, jump policy replaced by recorded commands + bias)
Forward = along the robot's heading at take-off; peak z is the body-centre height (start 0.105 m).

| config | forward | sideways | peak z | landing |
|---|---|---|---|---|
| stock commands replayed (no policy) | -2.5 cm | - | - | - |
| first bias (above) | +5.5 cm | <= 1.3 cm | - | upright |
| v3 (`configs/directional_jump_v3.json`, body height not constrained) | +14.6 cm | 7.1 cm | not logged (CPU model: ~8 cm rise) | upright (z 0.092) |
| v4a (`configs/directional_jump_v4a.json`, search required >=0.14 m rise) | **+38.7 cm** | 10.4 cm | 0.219 m | upright (z 0.094) |
| **v4c** (`configs/directional_jump_v4c.json`, rise >= 0.22 requested) | **+37.8 cm** | **1.7 cm** | 0.173 m | upright (z 0.091) |

v3 was a low dive: the search had rewarded distance and body height was not required. Requiring the rise fixed it. With the
rise constraint the CPU model (0.40-0.44 m) and Jumper's sim agree to within ~10%, so the earlier 2.4x over-prediction does not
carry over to these parameters. v4c is the default for `--jump-style directional`. One run per config, not repeated.

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
