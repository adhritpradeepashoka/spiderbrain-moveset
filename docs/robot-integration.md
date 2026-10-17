# Driving Jumper with the brain

Plan: the brain emits high-level intents; Jumper's own policies do the joint control.
1. Per tick: camera frame (+ dToF depth) → `EyeFrontEnd` → `RateNet.step` → `Readouts` → `BrainPad.update` → `BrainPad.state()`.
2. Give Jumper a pad-like object: `AppPlayer(pad=BrainPad(...))` (needs `state()` and `connected`). Jumper's `Fsm` / `GamepadState` live in its `controller/` package.
3. Pounce = pulse button `A` (jump mode). Forward = `ly < 0`. Turn = second half of right-stick travel (`rx`).

Untested: no Jumper install or MuJoCo here. Do not trust the bindings until verified; a stale binding means the robot does nothing. Jumper is Apache-2.0; this repo only imports its API, it does not vendor code. Treat the Jumper clone as untrusted code.
Calibrate readouts first: run the net unstimulated, feed frames to `Readouts.calibrate`.
