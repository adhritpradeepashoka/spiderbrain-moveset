# Floor + crouch test on Jumper's stock walk (19 Oct 2026)

`scripts/floor_test.py`, Jumper pinned at 78966f4, bundle `out/bundle_example/jumper.app`, one run per scene, 2100 steps (10.5 s sim at 200 Hz).
Scripted pad: stand 0.5 s -> walk 3 s -> stop -> lower body 1.2 s -> walk crouched 2.5 s -> raise. No brain, no policy changes: Jumper's own walk and body-height controls.

| scene | walk 3 s (m) | crouched walk 2.5 s (m) | body height stand -> crouch (m) | max roll / pitch (deg) |
|---|---|---|---|---|
| plain | 2.246 | 1.815 | 0.105 -> 0.069 | 1.1 / 2.1 |
| ice | 0.954 | 1.092 | (stand value not read) -> 0.069 | 1.1 / 2.3 |
| soft | 1.994 | 1.583 | 0.093 -> 0.055 | 1.6 / 2.8 |
| rubber | 2.131 | 1.655 | 0.103 -> 0.067 | 1.2 / 1.9 |
| rough | 2.277 | 1.836 | 0.104 -> 0.069 | 1.3 / 2.4 |
| beach | 2.225 | 1.752 | 0.105 -> 0.068 | 1.0 / 2.3 |

Readings (single deterministic run each; not repeated):
- The body-height control works: about 35% lower (0.105 -> 0.069 m on plain) and back to standing afterwards.
- Nobody fell: roll and pitch stayed under 3 degrees on every floor.
- Ice is the only floor that clearly hurts the stock walk: 0.32 m/s standing vs 0.75 m/s on plain. Crouched on ice it did better per second (0.44 m/s) than standing on ice. One run, so suggestive only.
- Soft and rubber cost roughly 5-12%. Rough and beach look like plain in these numbers (not checked whether they change physics at all).
