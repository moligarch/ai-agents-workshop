#!/usr/bin/env python3
"""
Smart Thermostat — Simple Reflex Agent

PURPOSE OF THIS SECTION
----------------------
This lab introduces intelligent agents via the classic Sense → Think → Act loop,
using a *simple reflex* (rule-based) thermostat. It shows how a minimal policy
(if-then rules) can still capture real trade-offs (stability vs. responsiveness),
and prepares participants for richer agents (ReAct, tools/function calling, RAG).

AGENT FRAMING
-------------
Percepts (Sense):
  - temperature: float (°C)
  - occupied: bool (is the space occupied?)

Policy (Think) — configuration:
  - mode: "heating" | "cooling" | "auto"
  - setpoint_h: comfort setpoint for heating (°C)
  - setpoint_c: comfort setpoint for cooling (°C)
  - deadband: total tolerance window around each setpoint (°C)
  - eco_offset: relax comfort when unoccupied (°C)

Actions (Act):
  - HEAT_ON, COOL_ON, OFF

KEY IDEAS
---------
1) Deadband:
   A temperature window around the setpoint where the system does NOT toggle.
   Example (heating): setpoint_h=21, deadband=1 → band = [20.5, 21.5].
   We only turn heat ON if temp < 20.5, and turn it OFF if temp > 21.5.

2) Hysteresis (stickiness):
   We remember the last action. When the temperature hovers around the boundary,
   we prefer to keep the current actuator ON until we clearly exit the band.
   This prevents rapid toggling (relay “chatter”).

3) Occupancy-aware Eco:
   When unoccupied, relax comfort to save energy:
     effective heating setpoint = setpoint_h - eco_offset
     effective cooling setpoint = setpoint_c + eco_offset

4) Modes:
   - "heating": consider only heating rules
   - "cooling": consider only cooling rules
   - "auto": decide which side is more strongly violated (heat vs. cool)

DECISION ALGORITHM (STEP BY STEP)
---------------------------------
Inputs:
  p.temperature (T), p.occupied (Occ), config (mode, setpoint_h, setpoint_c, deadband, eco_offset)

1) Compute effective setpoints based on occupancy:
     sp_h = setpoint_h - (0 if Occ else eco_offset)
     sp_c = setpoint_c + (0 if Occ else eco_offset)

2) Compute half-deadband:
     half = deadband / 2.0
   Bands:
     heating band = [sp_h - half, sp_h + half]
     cooling band = [sp_c - half, sp_c + half]

3) Evaluate rules per mode:

   HEATING MODE:
     if T < (sp_h - half):     return HEAT_ON
     elif T > (sp_h + half):   return OFF
     else:                     return HEAT_ON if last_action == HEAT_ON else OFF

   COOLING MODE (symmetric):
     if T > (sp_c + half):     return COOL_ON
     elif T < (sp_c - half):   return OFF
     else:                     return COOL_ON if last_action == COOL_ON else OFF

   AUTO MODE:
     need_heat = (T < (sp_h - half))
     need_cool = (T > (sp_c + half))

     if need_heat and not need_cool:
        return HEAT_ON
     elif need_cool and not need_heat:
        return COOL_ON
     elif need_heat and need_cool:
        # This indicates overlapping or ill-configured setpoints; fall back safely:
        return last_action
     else:
        # Inside both bands → prefer OFF unless hysteresis keeps one actuator on:
        if last_action == HEAT_ON:
            # Defer to heating band hysteresis rule:
            return HEAT_ON if T <= (sp_h + half) and T >= (sp_h - half) else (HEAT_ON if T < (sp_h - half) else OFF)
        elif last_action == COOL_ON:
            # Defer to cooling band hysteresis rule:
            return COOL_ON if T <= (sp_c + half) and T >= (sp_c - half) else (COOL_ON if T > (sp_c + half) else OFF)
        else:
            return OFF

4) Update last_action and return the decision.

WHY THIS WORKS
--------------
- Stability vs. responsiveness:
  Larger deadband → fewer toggles, slower return to comfort; smaller deadband → tighter comfort, more toggles.
- Hysteresis reduces toggling when T sits near a threshold.
- Eco offset trades off comfort recovery time vs. energy savings when unoccupied.
- Reflex design: there is NO dynamics model or prediction; we react only to the current percept.

EDGE CASES & GUARDS
-------------------
- Misconfigured setpoints (e.g., sp_h >= sp_c with small deadband) can cause
  ambiguous "both violated" outcomes in AUTO; code falls back to the last action.
- Extremely small deadband (e.g., 0.1°C) can still oscillate if sensors are noisy.
  Increase deadband or add sensor smoothing if needed.
- Occupancy flipping rapidly can cause frequent reconfiguration; in practice one
  may debounce occupancy events.

CLI & USAGE
-----------
Single decision:
  python thermostat_agent.py decide \
    --mode auto --setpoint_h 21 --setpoint_c 24 \
    --deadband 1.0 --eco_offset 2.0 --occupied true --temp 19.2

Simulate a sequence (JSON lines with {"temp": float, "occupied": bool}):
  python thermostat_agent.py simulate --scenarios scenarios.json

TESTING NOTES
-------------
- Unit tests assert key thresholds (turn on below band, off above band, eco behavior,
  auto-mode side selection).
- Run: `pytest -q` (tests are path-agnostic via sys.path adjustment in the test file).

EXTENSIONS
----------
- Add a goal/utility-based policy that optimizes energy cost vs. comfort.
- Introduce a simple thermal model (room heat capacity, HVAC power) to forecast T.
- Log decisions and plot temperature vs. actions for analysis.

"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import argparse, json, sys

class Action(str, Enum):
    HEAT_ON = "HEAT_ON"
    COOL_ON = "COOL_ON"
    OFF = "OFF"

@dataclass
class Config:
    mode: str = "auto"            # "heating" | "cooling" | "auto"
    setpoint_h: float = 21.0      # heating comfort setpoint (°C)
    setpoint_c: float = 24.0      # cooling comfort setpoint (°C)
    deadband: float = 1.0         # total width (°C)
    eco_offset: float = 2.0       # relax comfort when unoccupied (°C)

@dataclass
class Percept:
    temperature: float
    occupied: bool

class ThermostatAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_action: Action = Action.OFF

    def decide(self, p: Percept) -> Action:
        # derive effective setpoints
        sp_h = self.cfg.setpoint_h - (0 if p.occupied else self.cfg.eco_offset)
        sp_c = self.cfg.setpoint_c + (0 if p.occupied else self.cfg.eco_offset)
        half = self.cfg.deadband / 2.0

        def heat_rule(temp: float) -> Action:
            if temp < sp_h - half:
                return Action.HEAT_ON
            if temp > sp_h + half:
                return Action.OFF
            # within band: stickiness (hysteresis)
            return self._last_action if self._last_action == Action.HEAT_ON else Action.OFF

        def cool_rule(temp: float) -> Action:
            if temp > sp_c + half:
                return Action.COOL_ON
            if temp < sp_c - half:
                return Action.OFF
            return self._last_action if self._last_action == Action.COOL_ON else Action.OFF

        mode = self.cfg.mode.lower()
        if mode == "heating":
            action = heat_rule(p.temperature)
        elif mode == "cooling":
            action = cool_rule(p.temperature)
        elif mode == "auto":
            # Choose the stronger violation; break ties using last action to avoid flipping
            need_heat = (p.temperature < sp_h - half)
            need_cool = (p.temperature > sp_c + half)
            if need_heat and not need_cool:
                action = Action.HEAT_ON
            elif need_cool and not need_heat:
                action = Action.COOL_ON
            elif need_heat and need_cool:
                # Impossible if setpoints overlap properly; fall back to last action
                action = self._last_action
            else:
                # inside both bands → prefer to turn off, unless hysteresis keeps one on
                if self._last_action == Action.HEAT_ON:
                    action = heat_rule(p.temperature)
                elif self._last_action == Action.COOL_ON:
                    action = cool_rule(p.temperature)
                else:
                    action = Action.OFF
        else:
            raise ValueError("mode must be one of: heating|cooling|auto")

        self._last_action = action
        return action

# ------------- CLI -------------
def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--mode", default="auto", choices=["heating", "cooling", "auto"])
    base.add_argument("--setpoint_h", type=float, default=21.0)
    base.add_argument("--setpoint_c", type=float, default=24.0)
    base.add_argument("--deadband", type=float, default=1.0)
    base.add_argument("--eco_offset", type=float, default=2.0)

    d = sub.add_parser("decide", parents=[base])
    d.add_argument("--temp", type=float, required=True)
    d.add_argument("--occupied", type=_bool, required=True)

    s = sub.add_parser("simulate", parents=[base])
    s.add_argument("--scenarios", type=str, required=True, help="path to scenarios.json")
    return p

def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = Config(
        mode=args.mode,
        setpoint_h=args.setpoint_h,
        setpoint_c=args.setpoint_c,
        deadband=args.deadband,
        eco_offset=args.eco_offset,
    )
    agent = ThermostatAgent(cfg)

    if args.cmd == "decide":
        action = agent.decide(Percept(temperature=args.temp, occupied=args.occupied))
        print(action.value)
        return 0

    if args.cmd == "simulate":
        with open(args.scenarios, "r", encoding="utf-8") as f:
            data = json.load(f)
        for step in data:
            p = Percept(temperature=float(step["temp"]), occupied=bool(step["occupied"]))
            a = agent.decide(p)
            print(f't={p.temperature:>4.1f}°C  occ={p.occupied!s:<5}  -> {a.value}')
        return 0

if __name__ == "__main__":
    sys.exit(main())
