#!/usr/bin/env bash
# One-shot for a fresh Kaggle session (Internet On, GPU optional): installs Jumper (pinned) + this repo, then runs the
# brain with the forward-biased jump in Jumper's own simulator.  Logs: /kaggle/working/{setup,dj}.log
set -u
W=/kaggle/working; J=$W/jumper; SB=$W/spiderbrain-moveset; REV=78966f45de6f2b6de3f7cea017f32a39f8b993e6
exec > >(tee $W/setup.log) 2>&1
[ -d $J ] || git clone -q https://github.com/KingKongRobotics/jumper.git $J
(cd $J && git checkout -q $REV && git log -1 --format='jumper %h')
[ -d $SB ] || git clone -q https://github.com/adhritpradeepashoka/spiderbrain-moveset.git $SB
(cd $SB && git pull -q && git log -1 --format='spiderbrain %h %s')
pip install -q uv
if [ ! -x $J/.venv/bin/python ]; then
  uv venv --python "$(which python3)" $J/.venv
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  CUDA=cu126; python3 -c "import sys; sys.exit(0 if float('${CAP:-0}')>=8.9 else 1)" && CUDA=cu128
  echo "torch index: $CUDA (compute cap ${CAP:-none})"
  uv pip install --python $J/.venv/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/$CUDA 2>&1 | tail -2
  (cd $J && uv pip install --python .venv/bin/python -e . 2>&1 | tail -2)
fi
# sideload our extra scene(s) into Jumper's scene registry (idempotent)
cp $SB/scenes_extra/courtship.py $SB/scenes_extra/ledge.py $J/scenes/
grep -q 'id="courtship"' $J/scenes/__init__.py || cat >> $J/scenes/__init__.py <<'PY'

register(id="courtship", description="football pitch + a second crab to court", tags=("presentation",), use="watching")
PY
grep -q 'id="ledge"' $J/scenes/__init__.py || cat >> $J/scenes/__init__.py <<'PY'

register(id="ledge", description="football pitch + a 4 cm ledge with prey on it", tags=("presentation",), use="watching")
PY
echo SETUP_DONE
[ "${SETUP_ONLY:-0}" = 1 ] && exit 0
export MUJOCO_GL=egl
# knobs: INSTINCT (hunt|threat|tun|court|ice: survival-crab scenario; ice needs SCENE=ice), STEPS (200 Hz ticks), STYLE (directional|stock), JSCALE (0 = stock commands replayed open-loop), SCENE, TAG (output suffix)
STEPS=${STEPS:-4000}; STYLE=${STYLE:-directional}; JSCALE=${JSCALE:-1.0}; SCENE=${SCENE:-football}; TAG=${TAG:-}; INSTINCT=${INSTINCT:-}
(cd $SB && $J/.venv/bin/python scripts/play_with_brain.py --jumper $J --seed 0 $([ "$SCENE" = football ] && echo --prey-entity ball) --jump-style $STYLE --jump-scale $JSCALE ${JPARAMS:+--jump-params $JPARAMS} ${INSTINCT:+--instinct $INSTINCT --instinct-log $W/inst$TAG.csv} \
  --log $W/dj$TAG.csv -- --app out/bundle_example/jumper.app --scene $SCENE --headless --steps $STEPS --video $W/dj$TAG.mp4) > $W/dj$TAG.log 2>&1
echo RUN_DONE rc=$?
