# Forward-jump feasibility probe (CPU, approximate servo model)

`jumper_phys.py` loads Jumper's MJCF + reference jump into plain MuJoCo with a PD/torque-speed servo approximation (not mjlab, no latency/thermal).
It reproduces the stock jump's stats (rise 0.13 m, flight 0.31 s, net travel ~-0.02 m).

`forward_jump_search.py` = open-loop CMA-ES over an 8-knot x 22-joint edit of the reference command.

Results (kp=20, 200 generations, pop 24):
- First run scored net displacement and "found" 1.4 m -- an exploit: the robot slid/walked before takeoff (takeoff already at 1.4 m). Discarded.
- Fixed score (air distance only, takeoff within 5 cm of start, upright landing): best ~0.02 m in the air (0.05 m net). Search likely under-explored; this is NOT proof a forward jump is impossible.

Takeaway: tweaking the stock trajectory open-loop doesn't give a forward jump; a target-conditioned policy trained in mjlab is the real route.
