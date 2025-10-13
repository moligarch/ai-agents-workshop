import os, sys
TEST_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)


from thermostat_agent import ThermostatAgent, Config, Percept, Action

def make_agent(mode="auto", sp_h=21.0, sp_c=24.0, deadband=1.0, eco=2.0):
    return ThermostatAgent(Config(mode=mode, setpoint_h=sp_h, setpoint_c=sp_c,
                                  deadband=deadband, eco_offset=eco))

def test_heating_turns_on_below_band():
    ag = make_agent(mode="heating", sp_h=21.0, deadband=1.0)
    assert ag.decide(Percept(temperature=19.4, occupied=True)) == Action.HEAT_ON  # 21 - 0.5 = 20.5

def test_heating_turns_off_above_band():
    ag = make_agent(mode="heating", sp_h=21.0, deadband=1.0)
    assert ag.decide(Percept(temperature=21.6, occupied=True)) == Action.OFF      # 21 + 0.5 = 21.5

def test_cooling_turns_on_above_band():
    ag = make_agent(mode="cooling", sp_c=24.0, deadband=1.0)
    assert ag.decide(Percept(temperature=24.6, occupied=True)) == Action.COOL_ON  # 24 + 0.5 = 24.5

def test_eco_relaxes_heating_setpoint():
    ag = make_agent(mode="heating", sp_h=21.0, deadband=1.0, eco=2.0)
    # Unoccupied → effective sp_h = 19.0; band upper = 19.5
    assert ag.decide(Percept(temperature=20.0, occupied=False)) == Action.OFF

def test_auto_selects_side_with_stronger_violation():
    ag = make_agent(mode="auto", sp_h=21.0, sp_c=24.0, deadband=1.0)
    # Much colder than heating band, far from cooling band → should heat
    assert ag.decide(Percept(temperature=18.0, occupied=True)) == Action.HEAT_ON
