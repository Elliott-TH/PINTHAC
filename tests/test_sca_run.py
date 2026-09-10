"""
Permanent tests for pinthac.sca.run, added per docs/PHASE5_BRIEF.md section 4.

Annular cases here always pass a small N and a loose outer_iter/tol through
run_channel's **solver_kwargs (see run.py's docstring and tests/test_sca_annular.py's
module docstring for why) so the suite stays fast; rod cases use a small n for the same
reason. No asserted number was obtained by running the code under test -- these check
run_channel's own dispatch/validation/reporting behavior (which correlation names are
accepted, which requests get honored, where a bad case's failure is located), not a
physics value.
"""
import warnings

import pandas as pd
import pytest

from pinthac.ranges import RangeWarning

with warnings.catch_warnings():
    # See tests/test_sca_annular.py's module docstring: sca.annular's module-level
    # UO2 conductivity-integral build warns on import, unrelated to this test file.
    warnings.simplefilter("ignore", RangeWarning)
    from pinthac.sca import run


ROD_GEOMETRY = {"type": "rod", "pitch": 0.0125, "rco": 0.0045, "tc": 0.00063,
                 "delta": 5e-4, "kc": 24}
ROD_CONDITIONS = {"G": 1200, "pval": 25, "Tin": 300 + 273.15, "q0": 25e3, "N": 20}


def test_module_imports():
    assert run.run_channel is not None and run.run_batch is not None


# ------------------------------------------------------------------- name selection
def test_select_raises_with_valid_options_on_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        run._select("not_a_real_model", run.HTC_MODELS, "htc")


def test_select_error_lists_the_actual_valid_names():
    with pytest.raises(ValueError) as excinfo:
        run._select("bogus", run.FRICTION_MODELS, "friction")
    for name in run.FRICTION_MODELS:
        assert name in str(excinfo.value)


def test_select_passes_none_through():
    assert run._select(None, run.BUNDLE_MODELS, "bundle") is None


# ------------------------------------------------------------------- rod dispatch
def test_run_channel_rod_dispatch():
    out = run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS)
    assert out["geom_type"] == "rod"
    assert out["convergence"]["ok"] is True
    assert out["notes"] == []
    assert "T_fuel_max" in out["result"]


def test_run_channel_reports_unhonored_rod_selection():
    out = run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, friction="wu")
    assert len(out["notes"]) == 1
    assert "friction" in out["notes"][0] and "wu" in out["notes"][0]


# ------------------------------------------------------------------- annular dispatch
def test_run_channel_annular_dispatch():
    out = run.run_channel(geometry={"type": "annular"}, conditions={"N": 5},
                           outer_iter=2, tol=1.0e6)
    assert out["geom_type"] == "annular"
    assert out["convergence"]["outer_iters_used"] >= 1
    assert "q_i" in out["result"]


def test_run_channel_reports_unhonored_annular_bundle_and_conductivity():
    out = run.run_channel(geometry={"type": "annular"}, conditions={"N": 5},
                           outer_iter=2, tol=1.0e6)
    # defaults are bundle="presser", fuel_conductivity="klimenko" -- annular has no
    # bundle correction at all (Q42) and is hardcoded to "nfi", so both are unhonored.
    assert len(out["notes"]) == 2
    joined = " ".join(out["notes"])
    assert "bundle" in joined and "fuel_conductivity" in joined


# ------------------------------------------------------------------- error handling
def test_unknown_geometry_type_raises():
    with pytest.raises(ValueError, match="rod.*annular"):
        run.run_channel(geometry={"type": "bogus"}, conditions={})


def test_missing_required_rod_geometry_key_raises():
    with pytest.raises(ValueError, match="missing geometry keys"):
        run.run_channel(geometry={"type": "rod", "pitch": 0.01}, conditions=ROD_CONDITIONS)


def test_two_phase_htc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="subcooled-boiling"):
        run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="bjorge")


# ------------------------------------------------------------------- known-bad case
def test_known_bad_case_gives_a_located_report_not_a_silent_nan():
    """docs/PHASE5_BRIEF.md section 4: "a known-bad case produces a readable convergence
    report rather than a silent NaN". Pitch too small for the rod OD gives a negative
    flow area, which propagates to NaN through the friction factor / Reynolds number --
    a genuine bad case, not a fabricated one."""
    bad_geometry = {"type": "rod", "pitch": 0.005, "rco": 0.0045, "tc": 0.00063,
                     "delta": 5e-4, "kc": 24}
    out = run.run_channel(geometry=bad_geometry, conditions=ROD_CONDITIONS)
    convergence = out["convergence"]
    assert convergence["ok"] is False
    assert convergence["node"] is not None
    assert convergence["field"] is not None
    assert "non-finite" in convergence["message"]


# ------------------------------------------------------------------- batch
def test_run_batch_isolates_one_bad_case_from_the_rest():
    cases = [
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="not_a_real_model"),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, friction="wu"),
    ]
    results = run.run_batch(cases)
    assert len(results) == 3
    assert results[0]["case_index"] == 0 and "error" not in results[0]
    assert results[1]["case_index"] == 1 and "error" in results[1]
    assert "unknown model" in results[1]["error"]
    assert results[2]["case_index"] == 2 and "error" not in results[2]
    assert len(results[2]["notes"]) == 1


def test_run_batch_one_case_across_several_correlations():
    cases = [
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="swenson"),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="chen_scw"),
    ]
    results = run.run_batch(cases)
    # Neither htc selection is honored by rod.py today (it is hardcoded to Swenson --
    # see run.py's module docstring), so both produce the same unhonored-htc note
    # unless the requested name matches the fixed one.
    assert "error" not in results[0] and "error" not in results[1]
    assert results[0]["notes"] == []            # "swenson" matches _ROD_FIXED
    assert len(results[1]["notes"]) == 1         # "chen_scw" does not


# ------------------------------------------------------------------- spreadsheet reader
def test_read_cases_csv_round_trips_rod_cases(tmp_path):
    csv_path = tmp_path / "rod_cases.csv"
    df = pd.DataFrame([
        {**{k: v for k, v in ROD_GEOMETRY.items() if k != "type"},
         **ROD_CONDITIONS, "N": 15},
        {**{k: v for k, v in ROD_GEOMETRY.items() if k != "type"},
         **ROD_CONDITIONS, "N": 15, "pitch": 0.0130},
    ])
    df.to_csv(csv_path, index=False)

    cases = run.read_cases_csv(str(csv_path), "rod")
    assert len(cases) == 2
    assert cases[0]["geometry"]["type"] == "rod"
    assert cases[0]["geometry"]["pitch"] == pytest.approx(0.0125)
    assert cases[1]["geometry"]["pitch"] == pytest.approx(0.0130)

    results = run.run_batch(cases)
    assert all("error" not in r for r in results)
    assert all(r["convergence"]["ok"] for r in results)


def test_read_cases_csv_rejects_bad_geom_type(tmp_path):
    csv_path = tmp_path / "cases.csv"
    pd.DataFrame([{"pitch": 0.01}]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="rod.*annular"):
        run.read_cases_csv(str(csv_path), "bogus")
