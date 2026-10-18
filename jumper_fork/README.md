# jumper_fork — a jump with a commanded landing offset

`files/ref_free_jump_dir/` is a **fork-by-wrapper** of KingKongRobotics/jumper's `jumper.ref_free_jump` (Apache-2.0, pinned commit 78966f4).
It builds the stock env config and edits it: the go-command becomes `DirJumpCommand` (samples a landing offset per episode, latches where the
jump touched down), the actor and critic gain `jump_target` as their last observation, and `horizontal_drift` is replaced by `landing_target`
(+6, σ 0.10 m, after touchdown) and `flight_target` (+2, σ 0.25 m, in flight). Curriculum: forward target 0 → 0.30 m over iterations 100–1500
of the run; 20 % of episodes keep (0,0).

`make_warmstart.py` pads Jumper's trained `model_5999.pt` with 3 zero input columns, so the new policy starts identical to the vertical jump
(checked numerically: max output difference 0.0). **Not validated in mjlab/GPU.** Unknowns: whether the servo/torque limits allow a 0.3 m hop
(my open-loop CPU search found none, see ../experiments), whether PPO escapes the vertical-jump optimum, and the reward weights (guesses).
Also: the deployed controller must supply (dx,dy) as an observation; Jumper's `scripts/export.py` whitelist will not know the term.
