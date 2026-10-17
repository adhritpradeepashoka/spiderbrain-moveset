# spiderbrain-moveset

A seeded generator of **synthetic jumping-spider (Salticidae) connectome priors**, a small rate-model simulator, and a bridge that lets the resulting "brain" steer the [KingKong Jumper](https://github.com/KingKongRobotics/jumper) crab robot through its existing movement sets.

> **This is not a measured connectome.** No jumping-spider connectome, EM volume or brain cell atlas exists (searched Oct 2026). Region *existence* and many pathways follow *Marpissa muscosa* (Steinhoff et al. 2020); every size and wiring density is a prior set in `configs/`, tagged with an evidence tier (A reported in a salticid, B inferred from other spiders/flies, C speculative). The output is an **ensemble** of plausible brains. Report results across seeds, not for one.

## Quick start
```bash
pip install -e ".[dev]"
spiderbrain generate --profile tiny --out out        # ~11k neurons, seconds
spiderbrain generate --profile default --out out     # ~100k neurons, ~5M edges, ~15 s/seed on CPU
pytest
```
Colab: open `notebooks/colab_quickstart.ipynb` (installs the package, generates, simulates; uses the GPU via torch sparse if present, otherwise numpy).

## What's inside
| Path | What |
|---|---|
| `configs/regions.yaml`, `projections.yaml` | neuron populations and out-degree priors with 90% ranges and evidence tiers |
| `configs/transmitters.yaml` | transmitter fractions (placeholders until the Hylyphantes atlas GSE241696 is processed) |
| `src/spiderbrain/generate.py` | degree-corrected stochastic wiring, retinotopy, Dale's law, NB synapse counts |
| `src/spiderbrain/sim/` | leaky rate network (numpy or torch) and readouts |
| `src/spiderbrain/sensors/eyes.py` | camera frame → AME strip (4 tiers, defocus depth cue) and motion-based secondary eyes |
| `src/spiderbrain/robot/bridge.py` | readouts → Jumper gamepad state (turn, forward, pounce = jump) |
| `src/spiderbrain/motion/` | swing-then-fling jump template; Dryad *Maratus* loader stub |
| `configs/movesets.yaml` | Jumper mode names/bindings and readout definitions |

## Honest limits
- Neurons within a region are exchangeable, so readouts are design choices, not discoveries. Left-eye → left-descending bias exists because projections are wired ipsilaterally.
- Photoreceptor sign inversion is absorbed in the rate model (light excites second-order cells).
- Not tested against a live Jumper sim. Mode bindings and the d-pad hat convention must be checked against Jumper's `deploy/manifests.json` and `docs/CONTROLS.md`.
- Motion template is a hypothesis, not recorded kinematics. Mapping spider legs IV/III onto Jumper's rear/middle legs is a guess.
- Jumper has one camera, not eight eyes; secondary eyes are emulated from image halves.

See `docs/robot-integration.md` and `docs/evidence.md`. Licence: AGPL-3.0.
