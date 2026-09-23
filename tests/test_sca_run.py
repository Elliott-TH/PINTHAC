"""Permanent tests for pinthac.sca.run,

Annular cases here always pass a small N and a loose outer_iter/tol through
run_channel's **solver_kwargs (see run.py's docstring and tests/test_sca_annular.py's
module docstring for why) so the suite stays fast; rod cases use a small n for the same
reason. No asserted number was obtained by running the code under test -- these check
run_channel's own dispatch/validation/reporting behavior (which correlation names are
accepted, which requests get honored, where a bad case's failure is located), not a
physics value.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from pinthac.ranges import RangeWarning

with warnings.catch_warnings():
    # See tests/test_sca_annular.py's module docstring: sca.annular's module-level
    # UO2 conductivity-integral build warns on import, unrelated to this test file.
    warnings.simplefilter("ignore", RangeWarning)
    from pinthac.sca import run


def _np(v):
    """Solver outputs come back as torch tensors or numpy arrays depending on the
    geometry; normalize for comparison.
    """
    return np.asarray(v.detach().cpu() if torch.is_tensor(v) else v)


ROD_GEOMETRY = {"type": "rod", "pitch": 0.0125, "rco": 0.0045, "tc": 0.00063,
                 "delta": 5e-4, "kc": 24}
ANNULAR_GEOMETRY = {"type": "annular", "ri": 0.0035, "ro": 0.0055, "tci": 0.0006,
                    "tco": 0.0006, "delta_i": 1.0e-4, "delta_o": 1.0e-4,
                    "Pitch": 0.0130, "Gas": "He"}
ANNULAR_CONDITIONS = {"L": 4.27, "N": 5, "Tin_i": 623.15, "Tin_o": 623.15,
                      "Pnom": 25.0, "mdot_i": 0.010, "mdot_o": 0.060, "q0": 5.0e3}


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


def test_rod_selection_is_honored_not_merely_reported():
    """Every selection now reaches the solver. Before the sca rework, rod.py hardcoded
    its own physics and run_channel could only validate a name and note that it had been
    ignored -- so two different htc names returned bit-identical profiles. Asserting the
    results differ is what catches a regression back to that.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        swenson = _np(run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS,
                                       htc="swenson")["result"]["T_fuel_max"])
        chen = _np(run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS,
                                    htc="chen_scw")["result"]["T_fuel_max"])
    assert np.isfinite(swenson).all() and np.isfinite(chen).all()
    assert not np.allclose(swenson, chen)


def test_rod_bundle_selection_is_honored():
    presser = _np(run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS,
                                   bundle="presser")["result"]["T_fuel_max"])
    none_ = _np(run.run_channel(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS,
                                 bundle=None)["result"]["T_fuel_max"])
    assert not np.allclose(presser, none_)


# ------------------------------------------------------------------- annular dispatch
def test_run_channel_annular_dispatch():
    out = run.run_channel(geometry=ANNULAR_GEOMETRY, conditions=ANNULAR_CONDITIONS,
                           annular_method="picard", use_lut=True, outer_iter=2, tol=1.0e6)
    assert out["geom_type"] == "annular"
    assert out["convergence"]["outer_iters_used"] >= 1
    assert "q_i" in out["result"]


def test_annular_requires_every_key_rather_than_defaulting():
    """The solver used to carry a module-level Inputs_ann default case, so an omitted key
    silently became someone else's pin. It is required now, and the error names every
    missing key at once rather than failing on the first.
    """
    with pytest.raises(ValueError, match="missing keys"):
        run.run_channel(geometry={"type": "annular"}, conditions={"N": 5})


def test_annular_selection_is_honored():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        swenson = _np(run.run_channel(geometry=ANNULAR_GEOMETRY,
                                       conditions=ANNULAR_CONDITIONS, htc="swenson",
                                       annular_method="picard", use_lut=True, outer_iter=2, tol=1.0e6)["result"]["Tfo_o"])
        chen = _np(run.run_channel(geometry=ANNULAR_GEOMETRY,
                                    conditions=ANNULAR_CONDITIONS, htc="chen_scw",
                                    annular_method="picard", use_lut=True, outer_iter=2, tol=1.0e6)["result"]["Tfo_o"])
    assert np.isfinite(swenson).all() and np.isfinite(chen).all()
    assert not np.allclose(swenson, chen)


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
def test_known_bad_case_is_rejected_before_wall_solve():
    bad_geometry = {"type": "rod", "pitch": .005, "rco": .0045, "tc": .00063,
                    "delta": .0005, "kc": 24}
    with pytest.raises(ValueError, match="pitch must exceed"):
        run.run_channel(geometry=bad_geometry, conditions=ROD_CONDITIONS)



def test_run_batch_isolates_one_bad_case_from_the_rest():
    cases = [
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="not_a_real_model"),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, friction="wu"),
    ]
    # Wu is capped at G = 1000 kg/m^2-s and ROD_CONDITIONS runs at 1200, so case 2
    # legitimately warns. That is the range checker doing its job, not a batch failure --
    # what this test is about is that the one genuinely bad case (an unknown model name)
    # does not take the others down with it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        results = run.run_batch(cases)
    assert len(results) == 3
    assert results[0]["case_index"] == 0 and "error" not in results[0]
    assert results[1]["case_index"] == 1 and "error" in results[1]
    assert "unknown model" in results[1]["error"]
    # Case 2 runs and is honored: the friction selection reaches the solver now, so it
    # carries no note. Before the sca rework this asserted a note saying it was ignored.
    assert results[2]["case_index"] == 2 and "error" not in results[2]
    assert results[2]["notes"] == []


def test_run_batch_one_case_across_several_correlations():
    cases = [
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="swenson"),
        dict(geometry=ROD_GEOMETRY, conditions=ROD_CONDITIONS, htc="chen_scw"),
    ]
    # Chen's range check fires at the cosine tails -- see the note on the selection
    # tests above.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        results = run.run_batch(cases)
    assert "error" not in results[0] and "error" not in results[1]
    # Both selections are honored now, so neither carries a note and the two runs give
    # genuinely different answers. Before the sca rework rod.py was hardcoded to Swenson
    # and the second case came back flagged as unhonored.
    assert results[0]["notes"] == [] and results[1]["notes"] == []
    assert not np.allclose(_np(results[0]["result"]["T_fuel_max"]),
                           _np(results[1]["result"]["T_fuel_max"]))


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
