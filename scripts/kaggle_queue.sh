#!/usr/bin/env bash
# Re-create the survival-crab clips (new behaviours) and the failures (bloopers) in a Kaggle session that already ran
# `SETUP_ONLY=1 bash kaggle_setup_and_run.sh`.  Usage: bash kaggle_queue.sh A|B   (A = new behaviours, B = failures)
# Each run writes /kaggle/working/{dj,inst}<TAG>.{mp4,csv,log}; a line "QDONE <tag>" is appended to q_<stream>.out.
set -u
W=/kaggle/working; J=$W/jumper; SB=$W/spiderbrain-moveset; export MUJOCO_GL=egl
run() {  # run SBDIR TAG SCENE STEPS STYLE JPARAMS INSTINCT
  local D=$1 TAG=$2 SCENE=$3 STEPS=$4 STYLE=$5 JP=$6 INST=$7
  (cd $D && $J/.venv/bin/python scripts/play_with_brain.py --jumper $J --seed 0 $([ "$SCENE" = football ] || [ "$SCENE" = courtship ] || [ "$SCENE" = ledge ] && echo --prey-entity ball) $([ "$SCENE" = courtship ] && echo --mate-entity mate) $([ "$SCENE" = ledge ] && echo --step-entity step) \
    --jump-style $STYLE --jump-scale ${JSC:-1.0} ${JP:+--jump-params $JP} ${INST:+--instinct $INST --instinct-log $W/inst_$TAG.csv} \
    --log $W/dj_$TAG.csv -- --app out/bundle_example/jumper.app --scene $SCENE --headless --steps $STEPS --video $W/dj_$TAG.mp4) > $W/dj_$TAG.log 2>&1
  echo "QDONE $TAG rc=$? $(date +%T)"
}
if [ "${1:-A}" = C ]; then
  # genuinely different floors for the floor-adaptation scenario, and a real second crab for courtship
  run $SB court2 courtship 4000 directional "" court
  run $SB soft   soft      4000 directional "" ice
  run $SB rough  rough     4000 directional "" ice
elif [ "${1:-A}" = E ]; then
  # "how hard can we push the jump": the forward-bias amplitude scaled up, with the torque probe printing motor load per jump
  for k in 1 2 3 5; do JSC=$k.0 run $SB push$k football 2400 directional "" hunt; done
elif [ "${1:-A}" = D ]; then
  run $SB ledge  ledge     3000 directional "" ledge
elif [ "${1:-A}" = A ]; then
  run $SB hunt3  football 5000 directional "" hunt     # catch-fix check
  run $SB threat football 3200 directional "" threat
  run $SB tun    football 3400 directional "" tun
  run $SB court  football 4000 directional "" court
  run $SB ice    ice      4000 directional "" ice
else
  [ -d $W/sb_old ] || { git clone -q $SB $W/sb_old && (cd $W/sb_old && git checkout -q adfa27e); }   # code before the hunt fixes
  run $W/sb_old hunt_failed football 5000 directional "" hunt
  run $SB dive_v3   football 4000 directional configs/directional_jump_v3.json ""
  run $SB flip_back football 4000 directional configs/flip_try_back.json ""
  run $SB flip_fwd  football 4000 directional configs/flip_try_fwd.json ""
fi
