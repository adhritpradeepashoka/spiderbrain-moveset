import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spiderbrain.instinct import Personality, Percept, SurvivalAgent, looming_signal


def test_looming_distinguishes_approach_from_passing():
    approach = [(0.02 * i, 2 * math.atan2(0.11, 1.6 - 1.6 * 0.02 * i)) for i in range(20)]
    passing = [(0.02 * i, 2 * math.atan2(0.11, 1.0)) for i in range(20)]          # same distance, no expansion
    assert looming_signal(approach) > 0.1
    assert looming_signal(passing) < 0.01


def test_personality_seeded():
    assert Personality.from_seed(5) == Personality.from_seed(5)
    assert Personality.from_seed(5) != Personality.from_seed(6)


def test_tun_and_revive():
    ag = SurvivalAgent(Personality())
    t = 0.0
    for _ in range(200):
        it = ag.tick(Percept(t=t, stress=0.95), 0.02); t += 0.02
    assert it.state == "tun" and it.crouch and it.speed_scale == 0 and ag.metabolism < 0.05
    for _ in range(int(6 / 0.02)):
        it = ag.tick(Percept(t=t, stress=0.1), 0.02); t += 0.02
    assert it.state in ("revive", "wander")


def test_tun_protects_viability():
    a, b = SurvivalAgent(Personality()), SurvivalAgent(Personality())
    t = 0.0
    for _ in range(int(14 / 0.02)):
        a.tick(Percept(t=t, stress=0.95), 0.02)
        t += 0.02
    assert a.viability > a.shadow_viability + 0.3          # shutting down kept it alive; staying active would not have


def test_scenario_arc():
    from instinct_sim import run_scenario
    ag, _ = run_scenario(3)
    text = " ".join(e for _, e in ag.events)
    for needle in ("caught prey", "-> flee", "-> tun", "tun -> revive", "-> court", "courtship accepted"):
        assert needle in text, needle


def test_slippery_floor_disables_pounce_and_crouches():
    ag = SurvivalAgent(Personality())
    ag.energy = 0.3
    it = ag.tick(Percept(t=0.0, prey_dist=0.3, prey_bearing=0.0, friction=0.05), 0.02, {"turn_ccw": 0.0, "forward": 1.0, "pounce": True})
    assert it.state == "hunt" and not it.pounce and it.crouch and it.forward < 1.0
