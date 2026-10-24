"""climate_intelligence: auditable climate-impact estimates for DIVERGE experiments.

Standalone package. It does not import or modify `diverge/`, `engine/` or `unbound/`.
All results are estimates computed from caller-supplied inputs; nothing here
measures electricity or demonstrates energy savings.
"""
from __future__ import annotations

from .impact import (
    STATUSES,
    ImpactInputError,
    compare_scenarios,
    emissions_kg_co2e,
    estimate_scenario,
    facility_energy_kwh,
    gpu_energy_kwh,
)

VERSION = "cie-0.1-experimental"

__all__ = [
    "VERSION",
    "STATUSES",
    "ImpactInputError",
    "gpu_energy_kwh",
    "facility_energy_kwh",
    "emissions_kg_co2e",
    "estimate_scenario",
    "compare_scenarios",
]
