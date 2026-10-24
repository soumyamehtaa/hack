"""Unit tests for climate_intelligence.impact (no models, no network)."""
from __future__ import annotations

import json
import math

import pytest

from climate_intelligence.impact import (
    SCOPE,
    ImpactInputError,
    compare_scenarios,
    emissions_kg_co2e,
    estimate_scenario,
    facility_energy_kwh,
    gpu_energy_kwh,
    weakest,
)


def scenario(name="baseline", w=300.0, h=2.0, pue=1.2, gi=0.4, **kw):
    return estimate_scenario(name, w, h, pue, gi, **kw)


# ---------------------------------------------------------------- formulas
class TestFormulas:
    def test_gpu_energy(self):
        assert gpu_energy_kwh(300, 2) == pytest.approx(0.6)
        assert gpu_energy_kwh(700, 10) == pytest.approx(7.0)

    def test_facility_energy(self):
        assert facility_energy_kwh(0.6, 1.2) == pytest.approx(0.72)
        assert facility_energy_kwh(5.0, 1.0) == pytest.approx(5.0)

    def test_emissions(self):
        assert emissions_kg_co2e(0.72, 0.4) == pytest.approx(0.288)

    def test_scenario_chain_matches_formulas(self):
        out = scenario()["outputs"]
        assert out["gpu_energy_kwh"]["value"] == pytest.approx(0.6)
        assert out["facility_energy_kwh"]["value"] == pytest.approx(0.72)
        assert out["emissions_kg_co2e"]["value"] == pytest.approx(0.288)
        assert out["facility_energy_kwh"]["value"] >= out["gpu_energy_kwh"]["value"]

    def test_units_and_formulas_reported(self):
        r = scenario()
        assert r["outputs"]["gpu_energy_kwh"]["unit"] == "kWh"
        assert r["outputs"]["emissions_kg_co2e"]["unit"] == "kg CO2e"
        assert r["inputs"]["gpu_power_w"]["unit"] == "W"
        assert r["inputs"]["grid_intensity_kg_per_kwh"]["unit"] == "kg CO2e/kWh"
        assert r["outputs"]["gpu_energy_kwh"]["formula"] == "gpu_power_w * runtime_hours / 1000"
        assert r["scope"] == SCOPE


# ---------------------------------------------------------------- zero values
class TestZeros:
    def test_zero_power_or_runtime(self):
        assert gpu_energy_kwh(0, 5) == 0.0
        assert gpu_energy_kwh(400, 0) == 0.0

    def test_zero_grid_intensity(self):
        r = scenario(gi=0.0)
        assert r["outputs"]["emissions_kg_co2e"]["value"] == 0.0
        assert r["outputs"]["facility_energy_kwh"]["value"] == pytest.approx(0.72)

    def test_zero_baseline_percent_is_none(self):
        c = compare_scenarios(scenario(h=0.0), scenario("improved", h=0.0))
        assert c["comparison"]["gpu_energy_kwh"]["percent_reduction"] is None
        assert c["comparison"]["gpu_energy_kwh"]["reduction"] == 0.0


# ---------------------------------------------------------------- invalid inputs
class TestInvalid:
    @pytest.mark.parametrize("field,kwargs", [
        ("gpu_power_w", dict(w=-1)), ("runtime_hours", dict(h=-0.5)),
        ("grid_intensity_kg_per_kwh", dict(gi=-0.1)), ("pue", dict(pue=0.9)),
        ("pue", dict(pue=-1.2)),
    ])
    def test_negative_or_out_of_range_rejected(self, field, kwargs):
        with pytest.raises(ImpactInputError) as e:
            scenario(**kwargs)
        assert e.value.field == field

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "300", None, True, [300]])
    def test_non_numeric_or_non_finite_rejected(self, bad):
        with pytest.raises(ImpactInputError) as e:
            gpu_energy_kwh(bad, 1.0)
        assert e.value.field == "gpu_power_w"

    def test_error_is_value_error_with_structured_dict(self):
        with pytest.raises(ValueError):
            gpu_energy_kwh(100, -1)
        try:
            facility_energy_kwh(1.0, 0.5)
        except ImpactInputError as e:
            assert e.to_dict()["error"] == {"code": "invalid_input", "message": e.message, "field": "pue"}

    def test_bad_source_label_and_name(self):
        with pytest.raises(ImpactInputError) as e:
            scenario(power_source="guessed")
        assert e.value.field == "power_source"
        with pytest.raises(ImpactInputError):
            estimate_scenario("  ", 1, 1, 1, 1)

    def test_compare_rejects_non_scenarios(self):
        with pytest.raises(ImpactInputError):
            compare_scenarios({"foo": 1}, scenario())


# ---------------------------------------------------------------- status & transparency
class TestStatus:
    def test_defaults_are_assumptions_so_estimated(self):
        r = scenario()
        assert r["measurement_status"] == "estimated"
        assert set(r["assumptions"]) == {"gpu_power_w", "runtime_hours", "pue", "grid_intensity_kg_per_kwh"}

    def test_measured_gpu_inputs_but_assumed_pue(self):
        r = scenario(power_source="measured", runtime_source="measured")
        assert r["outputs"]["gpu_energy_kwh"]["status"] == "measured"
        assert r["outputs"]["facility_energy_kwh"]["status"] == "estimated"
        assert r["outputs"]["emissions_kg_co2e"]["status"] == "estimated"

    def test_all_measured(self):
        r = scenario(power_source="measured", runtime_source="measured",
                     pue_source="measured", grid_source="measured")
        assert r["measurement_status"] == "measured"
        assert r["assumptions"] == []

    def test_hypothetical_dominates(self):
        r = scenario(power_source="measured", runtime_source="measured",
                     pue_source="measured", grid_source="measured", hypothetical=True)
        assert all(o["status"] == "hypothetical" for o in r["outputs"].values())

    def test_weakest_ordering(self):
        assert weakest("measured", "estimated") == "estimated"
        assert weakest("estimated", "hypothetical", "measured") == "hypothetical"
        with pytest.raises(ValueError):
            weakest("synthetic-ish")

    def test_json_serializable_round_trip(self):
        r = scenario(notes={"pue": "industry-average assumption"})
        assert json.loads(json.dumps(r)) == r
        assert r["inputs"]["pue"]["note"] == "industry-average assumption"


# ---------------------------------------------------------------- comparisons
class TestCompare:
    def test_reduction_and_difference(self):
        base = scenario(h=10.0)
        better = scenario("improved", h=8.0)
        c = compare_scenarios(base, better)
        g = c["comparison"]["gpu_energy_kwh"]
        assert g["baseline"] == pytest.approx(3.0)
        assert g["improved"] == pytest.approx(2.4)
        assert g["reduction"] == pytest.approx(0.6)
        assert g["difference"] == pytest.approx(-0.6)
        assert g["percent_reduction"] == pytest.approx(20.0)
        e = c["comparison"]["emissions_kg_co2e"]
        assert e["reduction"] == pytest.approx(0.6 * 1.2 * 0.4)

    def test_worse_scenario_gives_negative_reduction(self):
        c = compare_scenarios(scenario(h=2.0), scenario("improved", h=3.0))
        assert c["comparison"]["facility_energy_kwh"]["reduction"] < 0

    def test_swap_negates_difference(self):
        a, b = scenario(h=4.0), scenario("improved", h=3.0)
        d1 = compare_scenarios(a, b)["comparison"]["emissions_kg_co2e"]["difference"]
        d2 = compare_scenarios(b, a)["comparison"]["emissions_kg_co2e"]["difference"]
        assert math.isclose(d1, -d2)

    def test_assumed_inputs_never_demonstrated_saving(self):
        c = compare_scenarios(scenario(h=10.0), scenario("improved", h=8.0))
        assert c["measurement_status"] == "estimated"
        assert c["demonstrated_saving"] is False
        assert "not a demonstrated saving" in c["note"]

    def test_hypothetical_improvement_is_hypothetical(self):
        all_measured = dict(power_source="measured", runtime_source="measured",
                            pue_source="measured", grid_source="measured")
        base = scenario(h=10.0, **all_measured)
        hypo = scenario("assumed 20% less compute", h=8.0, hypothetical=True, **all_measured)
        c = compare_scenarios(base, hypo)
        assert c["measurement_status"] == "hypothetical"
        assert c["demonstrated_saving"] is False
        assert all(r["status"] == "hypothetical" for r in c["comparison"].values())

    def test_fully_measured_comparison(self):
        m = dict(power_source="measured", runtime_source="measured",
                 pue_source="measured", grid_source="measured")
        c = compare_scenarios(scenario(h=1.0, **m), scenario("improved", h=0.5, **m))
        assert c["measurement_status"] == "measured"
        assert c["demonstrated_saving"] is True
        assert c["scope"] == SCOPE
        assert json.loads(json.dumps(c)) == c
