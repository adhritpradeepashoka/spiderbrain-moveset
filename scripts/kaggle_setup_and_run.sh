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
echo SETUP_DONE
export MUJOCO_GL=egl
STEPS=${STEPS:-4000}; STYLE=${STYLE:-directional}
(cd $SB && $J/.venv/bin/python scripts/play_with_brain.py --jumper $J --seed 0 --prey-entity ball --jump-style $STYLE \
  --log $W/dj.csv -- --app out/bundle_example/jumper.app --scene football --headless --steps $STEPS --video $W/dj.mp4) > $W/dj.log 2>&1
echo RUN_DONE rc=$?
