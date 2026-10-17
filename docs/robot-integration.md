# Driving Jumper with the brain

Plan: the brain emits high-level intents; Jumper's own policies do the joint control.
1. Per tick: camera frame (+ dToF depth) → `EyeFrontEnd` → `RateNet.step` → `Readouts` → `BrainPad.update` → `BrainPad.state()`.
2. Give Jumper a pad-like object: `AppPlayer(pad=BrainPad(...))` (needs `state()` and `connected`). Jumper's `Fsm` / `GamepadState` live in its `controller/` package.
3. Pounce = pulse button `A` (jump mode). Forward = `ly < 0`. Turn = second half of right-stick travel (`rx`).

Untested: no Jumper install or MuJoCo here. Do not trust the bindings until verified; a stale binding means the robot does nothing. Jumper is Apache-2.0; this repo only imports its API, it does not vendor code. Treat the Jumper clone as untrusted code.
Calibrate readouts first: run the net unstimulated, feed frames to `Readouts.calibrate`.

## Status (updated)
Verified by reading Jumper's code/docs (commit 78966f4, 10 Oct 2026): jump = pad `A` toggled from `locomotion`; forward = `Ly` negative (`lin_vel_x = -Ly`); turn CCW = `Rx` negative in the second half of travel; `AppPlayer(pad=...)` calls `pad.state()` and reads `connected`, with the d-pad hat sent as `dpad_y = -hat_y`. Pad input older than `command_timeout_ms` (500 ms in the jumper bundle) drops every latched mode, so `state()` must be polled every tick (the launcher does).

Tested here: the brain -> pad loop in `scripts/mock_arena.py` (kinematic stand-in, not physics): the brain orients toward moving prey, stalks, and presses A inside the 5-40 cm window. Because the random wiring does not say which neurons mean "left", a linear readout is fitted on the frozen network (`fit_ridge_readout`); how well it decodes varies by seed (report the ensemble).

NOT tested: `scripts/play_with_brain.py` against a live Jumper (needs mjlab/MuJoCo Warp, torch, onnxruntime, a built `.app` bundle). The prey is a position read from the scene (`--prey-entity`) or a fixed point, rendered as a synthetic blob, not from Jumper's camera, and it is not drawn in the video unless the scene shows it (e.g. the football scene's ball).
