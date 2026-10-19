import numpy as np
import pytest

from spiderbrain.config import load_config, with_profile
from spiderbrain.generate import generate
from spiderbrain.sensors.eyes import EyeFrontEnd
from spiderbrain.sim.rate import RateNet
from spiderbrain.sim.readout import Readouts
from spiderbrain.robot.bridge import BrainPad
from spiderbrain.motion.template import jump_template


@pytest.fixture(scope="module")
def cfg():
    return with_profile(load_config(), "tiny")


@pytest.fixture(scope="module")
def conn(cfg):
    return generate(cfg, 0)


def test_deterministic(cfg):
    a, b = generate(cfg, 3), generate(cfg, 3)
    assert a.edges.equals(b.edges)
    assert not a.edges.equals(generate(cfg, 4).edges)


def test_dale_and_no_self_loops(conn):
    e, n = conn.edges, conn.neurons
    assert (e.src != e.dst).all()
    # each source has a single sign (merged duplicates can only add same-sign counts)
    s = np.sign(e.syn.to_numpy())
    assert (s == n["sign"].to_numpy()[e.src]).all()


def test_pathway_ame_to_ab_exists(cfg):
    ok = 0
    for seed in range(5):
        c = generate(cfg, seed)
        W = c.weights()
        r = lambda x: c.region_ids(x)
        ok += W[r("am1")][:, r("ame_pr")].nnz > 0 and W[r("ab")][:, r("am2")].nnz > 0
    assert ok == 5


def test_contralateral_mb_bridge(conn):
    mb = conn.region_ids("mb_globuli")
    h = conn.neurons["hemi"].to_numpy()
    e = conn.edges
    m = e.src.isin(mb) & e.dst.isin(mb)
    sub = e[m & (e.proj == conn.proj_names.index("mb_bridge"))]
    assert (h[sub.src] != h[sub.dst]).mean() > 0.95


def test_left_stimulus_drives_left_dn(cfg, conn):
    net = RateNet(conn, backend="numpy", sensory_cfg=cfg["regions"]["regions"])
    eye = EyeFrontEnd(conn, cfg["eyes"], gain=5)
    for t in range(200):
        f = np.zeros((120, 160, 3), np.float32)
        f[50:70, 10 + t // 4:20 + t // 4] = 1
        net.step(eye(f))
    dn = lambda h: net.r[conn.region_ids("dn", h)].mean()
    assert dn(0) > dn(1)


def test_bridge_pounce_gate(cfg):
    pad = BrainPad(cfg["movesets"])
    pad.update({"pounce": True}, 0.02, distance_m=2.0)
    assert not pad.state().A
    pad.update({"pounce": True, "forward": 1.0}, 0.02, distance_m=0.2, fovea_contrast=0.5)
    assert not pad.state().A and pad.state().ly == 0          # freezes first, does not walk into the jump
    pressed = False
    for _ in range(int(1.2 / 0.02)):                           # ...then presses A once settled
        pad.update({"pounce": True, "forward": 1.0}, 0.02, distance_m=0.2, fovea_contrast=0.5)
        pressed |= pad.state().A
    assert pressed


def test_bridge_forward_is_negative_ly(cfg):
    pad = BrainPad(cfg["movesets"])
    pad.update({"forward": 1.0}, 0.02)
    assert pad.state().ly < 0


def test_template_shapes():
    t = jump_template()
    assert len(t["t"]) == len(t["LR"]) and t["LR"].max() > 0.9


def test_ridge_readout_decodes_turn(cfg):
    from spiderbrain.sim.readout import fit_ridge_readout
    c = generate(cfg, 0)
    _, r2 = fit_ridge_readout(c, cfg, episodes=40, verbose=False)
    assert r2[0] > 0.3          # bearing decodable from frozen random net (seed-dependent: report the ensemble)


def test_mock_arena_orients_and_pounces(cfg):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("mock_arena", pathlib.Path(__file__).parents[1] / "scripts" / "mock_arena.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    _, pounce = m.run(seed=0, verbose=False)
    # the gate (<= 0.40 m) is checked when the freeze starts; the jump fires 0.8 s later and the prey keeps wandering
    assert pounce is not None and 0.05 <= pounce[1] <= 0.45


def test_search_finds_prey_behind(cfg):
    """Prey starts out of view (100 deg left): the brain must search, find it, approach and pounce."""
    import importlib.util, pathlib, math
    spec = importlib.util.spec_from_file_location("mock_arena", pathlib.Path(__file__).parents[1] / "scripts" / "mock_arena.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    _, pounce = m.run(seed=0, prey=(-0.3, 1.2), steps=1500, verbose=False)
    assert pounce is not None


def test_directional_jump_commands():
    import numpy as np
    from spiderbrain.robot.directional_jump import biased_commands, DirectionalJump, LEFT, RIGHT, MIRROR
    names = [f"{s}_J{i}_joint" for s in ("LF", "RF", "LM", "RM", "LR", "RR") for i in range(3)]
    q = np.zeros((295, len(names)))
    assert np.allclose(biased_commands(q, names, scale=0.0), q)            # scale 0 = the stock commands
    b = biased_commands(q, names)
    for l, r, m in zip(LEFT, RIGHT, MIRROR):                               # right side mirrors the left
        assert np.allclose(b[:, names.index(r)], m * b[:, names.index(l)])
    assert np.abs(b).max() > 0.1 and np.abs(b[:20]).max() < 1e-3          # bias only around the push-off
    dj = DirectionalJump(q, names, blend_s=0.3)
    assert dj.target(0.0) is None
    dj.start(10.0, np.zeros(len(names)))
    assert dj.target(10.1) is not None and dj.target(10.0 + dj.duration + 0.01) is None


def test_bridge_directional_request():
    from spiderbrain.config import load_config
    from spiderbrain.robot.bridge import BrainPad
    cfg = load_config()
    pad = BrainPad(cfg["movesets"], jump_style="directional")
    for _ in range(6):
        pad.update({"turn_ccw": 0.0, "forward": 0.0, "pounce": True}, 0.25, distance_m=0.2, fovea_contrast=1.0)
        assert not pad.state().A                                          # directional never presses A
    assert pad.jump_request
