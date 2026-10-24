"""
Climate impact calculator (auditable, assumption-explicit).

Formulas
--------
    GPU energy (kWh)      = GPU power (W) x runtime (h) / 1000
    Facility energy (kWh) = GPU energy (kWh) x PUE
    Emissions (kg CO2e)   = Facility energy (kWh) x grid intensity (kg CO2e/kWh)

Measurement status
------------------
Every input carries a source label: "measured" (read from instrumentation or a
real run) or "assumed" (a user/literature assumption). Outputs get a status:

    measured     every input it depends on was measured
    estimated    at least one input was assumed
    hypothetical the scenario itself is hypothetical (e.g. an assumed
                 efficiency improvement not shown by an experiment)

Status is propagated conservatively: an output is never stronger than its
weakest input. PUE and grid intensity default to "assumed".

Scope
-----
Results describe only the GPU workload passed in. They do not represent
whole-data-center electricity use or savings: lower training compute does not
automatically lower total facility consumption (utilization, rebound effects,
inference load and idle power are not modeled).
"""
from __future__ import annotations

import math
from numbers import Real

STATUSES = ("measured", "estimated", "hypothetical")  # strongest -> weakest
SOURCES = ("measured", "assumed")

SCOPE = ("Covers only the GPU workload described by the inputs. Not a whole-data-center "
         "estimate; reduced training compute does not automatically reduce total facility "
         "electricity (utilization, rebound, inference and idle power are not modeled).")

FORMULAS = {
    "gpu_energy_kwh": "gpu_power_w * runtime_hours / 1000",
    "facility_energy_kwh": "gpu_energy_kwh * pue",
    "emissions_kg_co2e": "facility_energy_kwh * grid_intensity_kg_per_kwh",
}

UNITS = {
    "gpu_power_w": "W",
    "runtime_hours": "h",
    "pue": "dimensionless (facility energy / IT energy)",
    "grid_intensity_kg_per_kwh": "kg CO2e/kWh",
    "gpu_energy_kwh": "kWh",
    "facility_energy_kwh": "kWh",
    "emissions_kg_co2e": "kg CO2e",
}


class ImpactInputError(ValueError):
    """Invalid calculator input. `field` names the offending input."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message

    def to_dict(self) -> dict:
        return {"error": {"code": "invalid_input", "message": self.message, "field": self.field}}


# ---------------------------------------------------------------- validation
def _number(field: str, value, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ImpactInputError(field, f"must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ImpactInputError(field, f"must be finite, got {value!r}")
    if v < minimum:
        raise ImpactInputError(field, f"must be >= {minimum:g}, got {value!r}")
    return v


def _source(field: str, value: str) -> str:
    if value not in SOURCES:
        raise ImpactInputError(field, f"source must be one of {SOURCES}, got {value!r}")
    return value


def weakest(*statuses: str) -> str:
    for s in statuses:
        if s not in STATUSES:
            raise ValueError(f"unknown status {s!r}")
    return max(statuses, key=STATUSES.index)


# ---------------------------------------------------------------- formulas
def gpu_energy_kwh(gpu_power_w: float, runtime_hours: float) -> float:
    """GPU energy (kWh) = power (W) x runtime (h) / 1000."""
    w = _number("gpu_power_w", gpu_power_w)
    h = _number("runtime_hours", runtime_hours)
    return w * h / 1000.0


def facility_energy_kwh(gpu_kwh: float, pue: float) -> float:
    """Facility energy (kWh) = GPU energy (kWh) x PUE. PUE must be >= 1.0."""
    e = _number("gpu_energy_kwh", gpu_kwh)
    p = _number("pue", pue, minimum=1.0)
    return e * p


def emissions_kg_co2e(facility_kwh: float, grid_intensity_kg_per_kwh: float) -> float:
    """Emissions (kg CO2e) = facility energy (kWh) x grid intensity (kg CO2e/kWh)."""
    e = _number("facility_energy_kwh", facility_kwh)
    g = _number("grid_intensity_kg_per_kwh", grid_intensity_kg_per_kwh)
    return e * g


# ---------------------------------------------------------------- scenario
def _record(value, unit, status, source, note=None, formula=None):
    r = {"value": value, "unit": unit, "status": status, "source": source}
    if formula:
        r["formula"] = formula
    if note:
        r["note"] = note
    return r


def estimate_scenario(name: str, gpu_power_w: float, runtime_hours: float, pue: float,
                      grid_intensity_kg_per_kwh: float, *,
                      power_source: str = "assumed", runtime_source: str = "assumed",
                      pue_source: str = "assumed", grid_source: str = "assumed",
                      hypothetical: bool = False, notes: dict | None = None) -> dict:
    """Energy and emissions for one scenario, with every input, formula and status.

    Set `hypothetical=True` when the scenario is not backed by a real run (for
    example, an assumed compute reduction). Returns a JSON-serializable dict.
    """
    if not isinstance(name, str) or not name.strip():
        raise ImpactInputError("name", "must be a non-empty string")
    sources = {
        "gpu_power_w": _source("power_source", power_source),
        "runtime_hours": _source("runtime_source", runtime_source),
        "pue": _source("pue_source", pue_source),
        "grid_intensity_kg_per_kwh": _source("grid_source", grid_source),
    }
    notes = notes or {}

    gpu = gpu_energy_kwh(gpu_power_w, runtime_hours)
    fac = facility_energy_kwh(gpu, pue)
    co2 = emissions_kg_co2e(fac, grid_intensity_kg_per_kwh)

    def status_of(*inputs):
        s = "estimated" if any(sources[i] == "assumed" for i in inputs) else "measured"
        return weakest(s, "hypothetical") if hypothetical else s

    raw_inputs = {"gpu_power_w": gpu_power_w, "runtime_hours": runtime_hours, "pue": pue,
                  "grid_intensity_kg_per_kwh": grid_intensity_kg_per_kwh}
    inputs = {k: _record(float(v), UNITS[k], sources[k], "caller input", notes.get(k))
              for k, v in raw_inputs.items()}
    outputs = {
        "gpu_energy_kwh": _record(gpu, UNITS["gpu_energy_kwh"],
                                  status_of("gpu_power_w", "runtime_hours"), "computed",
                                  formula=FORMULAS["gpu_energy_kwh"]),
        "facility_energy_kwh": _record(fac, UNITS["facility_energy_kwh"],
                                       status_of("gpu_power_w", "runtime_hours", "pue"), "computed",
                                       formula=FORMULAS["facility_energy_kwh"]),
        "emissions_kg_co2e": _record(co2, UNITS["emissions_kg_co2e"],
                                     status_of(*sources), "computed",
                                     formula=FORMULAS["emissions_kg_co2e"]),
    }
    assumptions = [k for k, s in sources.items() if s == "assumed"]
    return {
        "name": name,
        "hypothetical": bool(hypothetical),
        "inputs": inputs,
        "outputs": outputs,
        "assumptions": assumptions,
        "measurement_status": outputs["emissions_kg_co2e"]["status"],
        "scope": SCOPE,
    }


def compare_scenarios(baseline: dict, improved: dict) -> dict:
    """Compare two `estimate_scenario` results.

    difference = improved - baseline (negative means the improved scenario uses less);
    reduction  = baseline - improved; percent = reduction / baseline x 100
    (None when the baseline value is 0). The comparison status is the weakest of
    the two scenarios' output statuses, so a hypothetical scenario always yields
    a hypothetical comparison. This is not a demonstrated saving unless both
    scenarios are measured.
    """
    for label, sc in (("baseline", baseline), ("improved", improved)):
        if not isinstance(sc, dict) or "outputs" not in sc:
            raise ImpactInputError(label, "must be a result of estimate_scenario()")
    rows = {}
    for key in ("gpu_energy_kwh", "facility_energy_kwh", "emissions_kg_co2e"):
        b, i = baseline["outputs"][key], improved["outputs"][key]
        reduction = b["value"] - i["value"]
        rows[key] = {
            "baseline": b["value"], "improved": i["value"],
            "difference": i["value"] - b["value"], "reduction": reduction,
            "percent_reduction": (100.0 * reduction / b["value"]) if b["value"] > 0 else None,
            "unit": b["unit"], "status": weakest(b["status"], i["status"]),
        }
    status = weakest(baseline["measurement_status"], improved["measurement_status"])
    return {
        "baseline": baseline["name"],
        "improved": improved["name"],
        "comparison": rows,
        "measurement_status": status,
        "demonstrated_saving": status == "measured",
        "note": ("Both scenarios measured: the difference reflects these runs only."
                 if status == "measured" else
                 f"Comparison is {status}: it depends on assumptions and is not a demonstrated saving."),
        "scope": SCOPE,
    }
