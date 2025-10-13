# Foundations — Reflex / Rule-Based Agents (Smart Thermostat Lab)

## Why this section

This module introduces **intelligent agents** through the classical **perception–reasoning–action** (sense–think–act) loop and implements a **simple reflex agent** with deterministic rules (a smart thermostat). Reflex agents map percepts to actions via if–then rules—great to illustrate the agent loop without LLMs.

## Learning objectives

By the end, participants can:

* Explain **agent**, **environment**, and the **sense–think–act** loop.
* Differentiate **simple reflex** vs. model/goal/utility-based agents (at a glance).
* Implement and test a small **rule-based agent** with **hysteresis** and **occupancy-aware eco rules**.
* Run repeatable simulations and reason about **oscillation** and **deadband** trade-offs.

## Repository layout (this section)

```
ai-agents-workshop/
  01-foundations/
    README.md
    smart_thermostat/
      thermostat_agent.py
      scenarios.json
      tests/
        test_thermostat_agent.py
      requirements.txt
```

## What you’ll build

A **portable** Python 3.10+ thermostat agent:

* Input percepts: `temperature`, `occupied` (bool).
* Config: `mode` (`heating|cooling|auto`), `setpoint_h`, `setpoint_c`, `deadband`, `eco_offset`.
* Output actions: `HEAT_ON | COOL_ON | OFF`.
* Features: **hysteresis** (to avoid rapid toggling), **eco setpoints when unoccupied**, CLI, and tests.

## Theory (quick primer)

* **Agent:** an entity that perceives its environment and **acts** to achieve goals; **rational** behavior aims to maximize expected goal achievement given evidence.
* **Sense–Think–Act:** perception → (state/rule evaluation) → action; a minimal but universal scaffold for agent design.
* **Simple Reflex Agent:** action = f(current percept) via if–then rules; no internal state. Pros: fast, predictable. Cons: brittle to unseen contexts; can **oscillate** without hysteresis. The thermostat is the canonical example.

## Prerequisites

* Python **3.10 or 3.11**.
* No external APIs. Only `pytest` for tests.

## Setup

```bash
cd agent-workshop/01-foundations/smart_thermostat
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Single decision from CLI:

```bash
python thermostat_agent.py decide \
  --mode auto --setpoint_h 21 --setpoint_c 24 \
  --deadband 1.0 --eco_offset 2.0 --occupied true --temp 19.2
```

Simulate a scenario file:

```bash
python thermostat_agent.py simulate --scenarios scenarios.json
```

Run tests:

```bash
pytest -q
```

## How it works (design notes)

* **Rules:**

  * *Heating:* turn heat **on** if `temp < setpoint_h - deadband/2`; turn **off** if `temp > setpoint_h + deadband/2`.
  * *Cooling:* symmetric for `setpoint_c`.
  * *Auto:* choose whichever side violates its band more.
* **Hysteresis:** we remember the **last action** (`HEAT_ON`/`COOL_ON`/`OFF`) so we don’t thrash near the boundary.
* **Occupancy:** when `occupied = false`, we relax comfort using `eco_offset`:

  * heating setpoint becomes `setpoint_h - eco_offset`
  * cooling setpoint becomes `setpoint_c + eco_offset`

## What to look for in the demo

* If you set `deadband` too small (e.g., 0.1°C), the agent flips often. Increase to 1.0–1.5°C for stability.
* Toggle `occupied` to see eco logic change the decision.
* In `auto` mode, check that only **one** actuator engages at a time.

## Troubleshooting

* If nothing happens: verify CLI subcommand (`decide | simulate`).
* If oscillation is visible: increase `deadband` and ensure the previous action is remembered.

## Further reading

* Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach* (Intelligent Agents).
* Open University — overview of agents and the sense–think–act loop.
* HF NLP course — concise recap of acting rationally & agent loop.
* Classic overview of simple reflex agents and thermostat examples.
