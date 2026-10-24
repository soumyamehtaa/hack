"""
SIMULATED input data for UNBOUND.

Every number in this file is an illustrative placeholder chosen to make the
model's behaviour legible. None of it is a measurement of the real
Massachusetts grid, utility queues, or data-center pipeline. Replace with
sourced data (ISO-NE CELT, utility hosting-capacity maps, interconnection
queues, EIA-860, vendor lead times) before drawing real-world conclusions.
"""
from __future__ import annotations

import copy

import numpy as np

SIM_LABEL = (
    "SIMULATED DATA: all inputs are illustrative placeholders, "
    "not real-world measurements of the Massachusetts grid."
)

# Load zones are generic regional groupings, not ISO-NE / utility zone definitions.
ZONES = [
    dict(name="Greater Boston", grid_headroom_mw=220, substation_headroom_mw=140, pipeline_mw=600),
    dict(name="MetroWest", grid_headroom_mw=180, substation_headroom_mw=120, pipeline_mw=450),
    dict(name="Central MA", grid_headroom_mw=260, substation_headroom_mw=90, pipeline_mw=400),
    dict(name="Pioneer Valley", grid_headroom_mw=150, substation_headroom_mw=110, pipeline_mw=250),
    dict(name="Southeastern MA", grid_headroom_mw=200, substation_headroom_mw=70, pipeline_mw=350),
]

DEFAULT_PARAMS = dict(
    base_year=2026,
    target_year=2030,
    # Interconnection: throughput = engineers x MW cleared per engineer-year x years
    ic_engineers=12,
    ic_mw_per_engineer_year=15.0,
    ic_bonus_mw=0.0,
    # Transformers: new pool = units/yr x years x MVA/unit x power factor
    xfmr_units_per_year=2.0,
    xfmr_mva_per_unit=60.0,
    power_factor=0.95,
    xfmr_existing_multiplier=1.0,
    # Grid
    grid_multiplier=1.0,
    dlr_uplift=0.0,
    # Peak / resource adequacy margin available to new large loads
    peak_margin_mw=800.0,
    # Flexible compute
    flex_share=0.30,        # max share of DC load that is flexible (training/batch)
    peak_curtailment=0.50,  # share of flexible load that can be shed during system peak
    btm_storage_mw=0.0,     # behind-the-meter batteries at data centers
    zone_grid_bonus={},
    zone_xfmr_bonus={},
    zones=ZONES,
)

PRESETS = {
    "Base case": {},
    "Consensus: build generation": {"peak_margin_mw": 1300.0},
    "Supply-chain shock": {"xfmr_units_per_year": 0.5, "xfmr_existing_multiplier": 0.8},
    "UNBOUND: flex + automation": {
        "ic_mw_per_engineer_year": 30.0,
        "flex_share": 0.5,
        "peak_curtailment": 0.9,
    },
}


def make_params(preset: str) -> dict:
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update(copy.deepcopy(PRESETS.get(preset, {})))
    return p


# --------------------------------------------------------------------------
# Demand modelling (simulated)
# --------------------------------------------------------------------------
DEMAND_BASELINE_MW_2024 = 300.0  # simulated statewide DC load in 2024
DEMAND_SCENARIOS = {"Low growth": 0.08, "Base growth": 0.15, "AI surge": 0.25}


def demand_projection(start=2024, end=2035) -> dict:
    years = np.arange(start, end + 1)
    return {
        name: (years, DEMAND_BASELINE_MW_2024 * (1 + g) ** (years - start))
        for name, g in DEMAND_SCENARIOS.items()
    }


def hourly_profile(firm_mw: float, flex_mw: float, curtailment: float,
                   peak_hours=(16, 17, 18, 19, 20)):
    """24h data-center load: firm load is flat; flexible load is curtailed
    during the system peak window and the deferred energy is shifted evenly
    into overnight hours (energy-conserving)."""
    hours = np.arange(24)
    firm = np.full(24, firm_mw)
    flex = np.full(24, flex_mw)
    shed = flex_mw * curtailment
    flex[list(peak_hours)] -= shed
    night = [0, 1, 2, 3, 4, 5]
    flex[night] += shed * len(peak_hours) / len(night)
    # Simulated normalized system load shape (evening peak), for context only
    system_shape = 0.62 + 0.25 * np.exp(-((hours - 18) ** 2) / 10) + 0.08 * np.exp(-((hours - 9) ** 2) / 6)
    return hours, firm, flex, system_shape, list(peak_hours)
