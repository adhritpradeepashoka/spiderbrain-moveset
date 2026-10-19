import math
from spiderbrain.instinct.agent import Percept
from spiderbrain.instinct.scenarios import SCENARIOS, World, angular_size, bearing_to, make_agent


def test_threat_charges_and_stops_short():
    w = World("threat")
    d = None
    for k in range(600):
        o = w.step(k * 0.01, 0.01, (0.0, 0.0, 0.0))
    assert o.role == "threat"
    d = math.hypot(*o.ball)
    assert 0.3 < d < 0.45


def test_threat_absent_before_it_appears():
    assert World("threat").step(1.0, 0.01, (0, 0, 0)).role == "none"


def test_tun_stress_profile_rises_then_falls():
    w = World("tun")
    st = [w.step(t, 0.01, (0, 0, 0)).stress for t in (1, 4, 8, 15)]
    assert st[0] < 0.3 and 0.3 < st[1] < 0.95 and st[2] > 0.9 and st[3] < 0.2


def test_hunt_respawns_prey_ahead_of_the_robot():
    w = World("hunt")
    w.step(0.0, 0.01, (0, 0, 0))
    o = w.step(5.0, 0.01, (2.0, 1.0, 0.5), caught=True)
    d, _ = bearing_to((2.0, 1.0, 0.5), o.ball)
    assert 1.0 < d < 1.5 and w.respawns == 1


def test_court_world_has_receptive_mate():
    o = World("court").step(0.0, 0.01, (0, 0, 0))
    assert o.role == "mate" and o.mate_receptive


def test_angular_size_grows_as_it_nears():
    assert angular_size(0.3) > angular_size(1.0) > angular_size(2.0)


def test_each_scenario_agent_has_a_mood_to_match():
    assert make_agent("hunt").energy < 0.7 and make_agent("court").mate_urge > 0.6
    for n in SCENARIOS:
        assert make_agent(n).state == "wander"


def test_charging_threat_triggers_flee_within_a_few_seconds():
    """World + agent only (no robot): a robot standing still sees the threat charge and bolts."""
    w, ag = World("threat"), make_agent("threat")
    fled = False
    for k in range(1200):
        t = k * 0.02
        o = w.step(t, 0.02, (0.0, 0.0, 0.0))
        pc = Percept(t=t)
        if o.role == "threat":
            d, b = bearing_to((0.0, 0.0, 0.0), o.ball)
            pc.threat_dist, pc.threat_bearing, pc.threat_theta = d, b, angular_size(d)
        i = ag.tick(pc, 0.02)
        if i.state.startswith("flee"):
            fled = True
            break
    assert fled
