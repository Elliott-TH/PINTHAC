"""
sca/run.py -- the top-level single-channel-analysis driver.

Why this module exists: sca/rod.py and sca/annular.py are each a complete axial solver
for one geometry, but a user (or a figure-generation script, or a batch data-gen sweep)
should not have to know which module a given case dispatches to, or repeat the
boilerplate of assembling that module's own inputs dict. run_channel() is the one
function that takes a geometry, a set of operating conditions, and a correlation
selection, and returns the axial solution plus a convergence report.

Correlation selection, honestly: docs/PHASE5_BRIEF.md asks for "correlation selection by
name" via plain dict literals mapping name -> function, and that mechanism is built below
(HTC_MODELS, FRICTION_MODELS, BUNDLE_MODELS, FUEL_CONDUCTIVITY_MODELS, and _select() to
look a name up with a clear error on an unknown one). But sca/rod.py and sca/annular.py
do not themselves accept an injected correlation -- their per-node physics calls
Swenson/Filonenko/etc. by name internally, hardcoded (rod.py additionally applies
Bundle.Presser and UO2.Theta_Klimenko/k_Klimenko; annular.py applies no bundle correction
at all, see docs/OPEN_QUESTIONS.md Q42, and is hardcoded to UO2.k_NFI). Rewiring either
solver's internals to accept an injected correlation is a real restructuring of a working
module (CLAUDE.md section 9.2: "never rewrite a working module without asking"), not
something to do silently inside a driver script. So run_channel() validates every
requested name against the tables below (a typo or an unsupported name fails immediately,
listing the valid options -- the actual "selection by name" contract), dispatches to
whichever solver the geometry needs, and reports which requested selections the
dispatched solver did NOT actually honor, rather than silently running different physics
than what was asked for or pretending the selection did something it did not.

Two-phase (PWR/BWR): correlations/htc.py has Chen, Bjorge and Schrock-Grossman for
two-phase flow, and their names are in HTC_MODELS below -- so they are importable and
selectable through the same mechanism. But neither sca/rod.py nor sca/annular.py is a
boiling-channel solver: both march a single-phase supercritical-water enthalpy balance
with no onset-of-nucleate-boiling correlation, no quality/void-fraction tracking, and no
subcooled-boiling bookkeeping anywhere in pinthac/sca/. Per docs/PHASE5_BRIEF.md's own
instruction ("if subcooled-boiling bookkeeping is not already present somewhere, say so
rather than inventing it"), run_channel() raises NotImplementedError for a two-phase htc
selection rather than pretending to run a boiling channel that does not exist.
"""
import numpy as np
import pandas as pd

from pinthac.correlations import bundle as bundle_mod
from pinthac.correlations import friction as fric
from pinthac.correlations import htc
from pinthac.properties import matmod
from pinthac.sca import annular, rod


# =============================================================================
# Correlation selection: a plain dict literal per category, name -> function (or, for
# fuel_conductivity, a (k_func, Theta_func) pair -- pin.cylindrical.Cyl_T and
# pin.annular.Ann_flux_split both need the conductivity and its integral together).
# No plugin registry, no entry points, no auto-discovery, no DSL, no config parser --
# docs/PHASE5_BRIEF.md forbids all of these by name. This is the whole mechanism.
# =============================================================================
HTC_MODELS = {
    "swenson": htc.SCW.Swenson_dT,
    "chen_scw": htc.SCW.Chen_SCW_dT,
    # Two-phase (PWR/BWR) -- selectable and importable, but see the module docstring:
    # neither solver below has the subcooled-boiling bookkeeping to actually run one.
    "chen_h2o": htc.Water.Chen_H2O_dT,
    "bjorge": htc.Water.Bjorge_dT,
    "schrock_grossman": htc.Water.SchrockGrossman,
}

FRICTION_MODELS = {
    "filonenko": fric.f_SCW.Filonenko,
    "wu": fric.f_SCW.Wu,
    "blasius": fric.f_water.Blasius,
    "mcadams": fric.f_water.McAdams,
    "colebrook": fric.f_water.Colebrook,
}

BUNDLE_MODELS = {
    "presser": bundle_mod.Bundle.Presser,
    "weissman": bundle_mod.Bundle.Weissman,
}

FUEL_CONDUCTIVITY_MODELS = {
    "klimenko": (matmod.UO2.k_Klimenko, matmod.UO2.Theta_Klimenko),
    "nfi": (matmod.UO2.k_NFI, matmod.UO2.Theta_NFI),
}

# What each solver's own per-node physics actually is today (see the module docstring).
# bundle=None means "this channel applies no bundle correction at all" (annular.py, Q42),
# not "unspecified".
_ROD_FIXED = dict(htc="swenson", friction="filonenko", bundle="presser",
                   fuel_conductivity="klimenko")
_ANNULAR_FIXED = dict(htc="swenson", friction="filonenko", bundle=None,
                       fuel_conductivity="nfi")

_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")

_ROD_GEOM_KEYS = ("pitch", "rco", "tc", "delta", "kc")
_ROD_COND_KEYS = ("G", "pval", "Tin", "q0", "L", "N")


def _select(name, table, category):
    """
    Look up a correlation by name, raising immediately with the valid options on a miss.

    Why this model is here:
        The single point every correlation-selection keyword in this module passes
        through -- the entire "selection by name" mechanism docs/PHASE5_BRIEF.md asks
        for, and nothing more (no registry, no fallback guessing, no partial matching).

    Inputs:
        name     : the requested model name, or None (passes through unchanged -- used
                   for the "bundle" category, where None means "no correction applies")
        table    : one of the module-level *_MODELS dicts above
        category : category name, used only in the error message
    Returns:
        table[name], or None if name is None
    """
    if name is None:
        return None
    if name not in table:
        raise ValueError(
            f"{category}: unknown model {name!r} -- valid options are {sorted(table)}"
        )
    return table[name]


def _unhonored_notes(requested, fixed):
    """
    Human-readable notes for every requested correlation the dispatched solver's
    hardcoded per-node physics does not actually match -- see the module docstring for
    why this is a note, not a silent no-op or a forced error.

    Inputs:
        requested : dict, the four correlation-selection kwargs as passed to run_channel
        fixed     : one of _ROD_FIXED / _ANNULAR_FIXED
    Returns:
        list of strings, empty if every requested name matches what the solver does
    """
    notes = []
    for key, fixed_name in fixed.items():
        req_name = requested.get(key)
        if req_name == fixed_name:
            continue
        if fixed_name is None:
            notes.append(
                f"{key}={req_name!r} was requested, but this channel applies no {key} "
                f"correction at all -- see docs/OPEN_QUESTIONS.md Q42."
            )
        else:
            notes.append(
                f"{key}={req_name!r} was requested, but this solver's per-node physics "
                f"is hardcoded to {key}={fixed_name!r}; the name was recognized but not "
                f"honored -- see this module's docstring."
            )
    return notes


def _scan_for_nonfinite(result, z_key, fields):
    """
    Find the first axial node where any tracked field is non-finite (NaN or +/-inf).

    Why this model is here:
        docs/PHASE5_BRIEF.md section 4 asks for "a known-bad case produces a readable
        convergence report rather than a silent NaN". Neither sca/rod.py's bisect_newton
        solves nor sca/annular.py's closure() raise on a physically nonsensical case --
        a bad bracket or a diverging Picard iterate just comes out as NaN or inf in the
        output arrays. This turns that into a specific, located failure instead of a
        result the caller has to notice is broken by inspecting every field by hand.

    Inputs:
        result : a run_SCA()/solve_field()-shaped dict of 1-D arrays
        z_key  : the key holding the axial position array ('Z' for rod, 'z' for annular)
        fields : field names to scan, in order
    Returns:
        dict: ok (bool), node (int or None), z (float or None), field (str or None),
        message (str)
    """
    z = np.asarray(result[z_key], dtype=float)
    for field in fields:
        values = np.asarray(result[field], dtype=float)
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            i = int(bad[0])
            return dict(ok=False, node=i, z=float(z[i]), field=field,
                        message=(f"{field} is non-finite (NaN or inf) at node {i} "
                                 f"(z = {z[i]:.4f} m) -- first occurrence; there may be "
                                 f"more."))
    return dict(ok=True, node=None, z=None, field=None,
                message="every tracked field is finite at every axial node")


def _rod_inputs(geometry, conditions):
    """Assemble sca/rod.py's inputs dict and run_SCA() keyword arguments from a
    geometry dict (pitch, rco, tc, delta, kc) and a conditions dict (G, pval, Tin, q0,
    and optionally L, N). Raises ValueError listing what is missing rather than letting
    rod.run_SCA fail on a KeyError with no context."""
    missing_geom = [k for k in _ROD_GEOM_KEYS if k not in geometry]
    missing_cond = [k for k in ("G", "pval", "Tin", "q0") if k not in conditions]
    if missing_geom or missing_cond:
        raise ValueError(
            f"rod channel: missing geometry keys {missing_geom}, missing condition keys "
            f"{missing_cond} -- required geometry: {_ROD_GEOM_KEYS}, required "
            f"conditions: ('G', 'pval', 'Tin', 'q0')"
        )
    inputs = {k: geometry[k] for k in _ROD_GEOM_KEYS}
    inputs["G"] = conditions["G"]
    run_kwargs = dict(
        pval=conditions["pval"],
        Tscw_in=conditions["Tin"],
        q0=conditions["q0"],
        L=conditions.get("L", 3.0),
        n=int(conditions.get("N", 400)),
    )
    return inputs, run_kwargs


def _annular_inputs(geometry, conditions):
    """Assemble sca/annular.py's Inputs_ann-shaped dict: that solver takes one flat dict
    carrying both geometry and operating conditions, so this starts from
    annular.Inputs_ann's own defaults (so an omitted key falls back to a documented
    default rather than a bare KeyError deep inside solve_field) and overlays whatever
    the caller supplied."""
    merged = dict(annular.Inputs_ann)
    merged.update(geometry)
    merged.update(conditions)
    return merged


def run_channel(geometry, conditions, htc="swenson", friction="filonenko",
                 bundle="presser", fuel_conductivity="klimenko"):
    """
    Run one single-channel-analysis case, dispatching to sca/rod.py or sca/annular.py.

    Why this model is here:
        The one entry point docs/PHASE5_BRIEF.md section 3 asks for: geometry and
        operating conditions as plain dicts, correlation selection by name, and a
        convergence report a person can read when a case fails. See the module
        docstring for exactly what the correlation-selection keywords do and do not
        change today.

    Formulation:
        Not a physical model -- a dispatcher. geometry['type'] selects rod.run_SCA (for
        'rod') or annular.solve_field (for 'annular'); geometry/conditions are reshaped
        into that solver's own input format by _rod_inputs/_annular_inputs.

    Inputs:
        geometry   : dict with a 'type' key, 'rod' or 'annular', plus that geometry's
                     own keys -- rod: pitch, rco, tc, delta, kc (m, m, m, m, W/m-K);
                     annular: any of Inputs_ann's geometry keys (ri, ro, tci, tco,
                     delta_i, delta_o, Gas, Pitch), defaulting to Inputs_ann's values
                     for anything omitted
        conditions : dict of operating conditions -- rod: G (kg/m^2-s), pval (MPa),
                     Tin (K), q0 (W/m), optionally L (m), N; annular: any of Inputs_ann's
                     condition keys (Tin_i, Tin_o, Pnom, mdot_i, mdot_o, q0, L, N),
                     defaulting the same way
        htc, friction, bundle, fuel_conductivity : correlation selection by name (see
                     HTC_MODELS/FRICTION_MODELS/BUNDLE_MODELS/FUEL_CONDUCTIVITY_MODELS
                     for the valid names). bundle=None means "apply no bundle
                     correction" and is always accepted.
    Returns:
        dict: result (the underlying run_SCA()/solve_field() output), geom_type,
        requested (the four correlation names as given), notes (list of strings --
        which requested selections the dispatched solver did not honor, see
        _unhonored_notes), convergence (dict, see _scan_for_nonfinite --
        for 'annular' this also carries outer_converged/outer_residual/outer_iters_used,
        see annular.solve_field's docstring)
    """
    geom_type = geometry.get("type")
    if geom_type not in ("rod", "annular"):
        raise ValueError(
            f"run_channel: geometry['type'] must be 'rod' or 'annular', got {geom_type!r}"
        )

    requested = dict(htc=htc, friction=friction, bundle=bundle,
                      fuel_conductivity=fuel_conductivity)

    # Validate every requested name is at least recognized, regardless of whether the
    # dispatched solver can honor it yet -- this is the actual "unknown name raises
    # immediately" contract, independent of which solver the case lands on.
    _select(htc, HTC_MODELS, "htc")
    _select(friction, FRICTION_MODELS, "friction")
    _select(bundle, BUNDLE_MODELS, "bundle")
    _select(fuel_conductivity, FUEL_CONDUCTIVITY_MODELS, "fuel_conductivity")

    if htc in _TWO_PHASE_HTC:
        raise NotImplementedError(
            f"htc={htc!r} is a two-phase correlation, but neither sca/rod.py nor "
            f"sca/annular.py has subcooled-boiling bookkeeping (onset-of-nucleate-boiling, "
            f"quality/void-fraction tracking) -- see this module's docstring. Not invented "
            f"here; use a single-phase htc selection ('swenson' or 'chen_scw')."
        )

    geom_only = {k: v for k, v in geometry.items() if k != "type"}

    if geom_type == "rod":
        inputs, run_kwargs = _rod_inputs(geom_only, conditions)
        result = rod.run_SCA(inputs, **run_kwargs)
        convergence = _scan_for_nonfinite(result, "Z", ("T_i", "T_fuel_max", "dP"))
        notes = _unhonored_notes(requested, _ROD_FIXED)
    else:
        inp = _annular_inputs(geom_only, conditions)
        result = annular.solve_field(inp)
        convergence = _scan_for_nonfinite(
            result, "z", ("Tm_i", "Tm_o", "Tfo_i", "Tfo_o", "q_i", "q_o")
        )
        convergence["outer_converged"] = bool(result["outer_converged"])
        convergence["outer_residual"] = result["outer_residual"]
        convergence["outer_iters_used"] = result["outer_iters_used"]
        if not result["outer_converged"]:
            convergence["ok"] = False
            convergence["message"] += (
                f" -- also, the outer enthalpy Picard loop did not converge: residual "
                f"{result['outer_residual']:.4g} J/kg after {result['outer_iters_used']} "
                f"iterations."
            )
        notes = _unhonored_notes(requested, _ANNULAR_FIXED)

    return dict(result=result, geom_type=geom_type, requested=requested,
                convergence=convergence, notes=notes)


def run_batch(cases):
    """
    Run many cases, or one case repeated across several correlation selections, in one
    call.

    Why this model is here:
        docs/PHASE5_BRIEF.md section 3 asks for exactly this -- "run many cases, or one
        case across several correlations, in one call", which is what the comparison
        figures and DeepONet training-data generation both need. One bad case (a typo'd
        correlation name, a geometry missing a required key) is reported and skipped
        rather than stopping every other case in the batch -- deliberate error isolation
        at this one boundary, not a substitute for checking values inline elsewhere in
        this module (CLAUDE.md section 2 forbids try/except as control flow; this is
        catching an actual exception at a batch boundary to isolate one case's failure
        from the rest, which is what run_channel already raises to signal).

    Formulation:
        Not a physical model -- calls run_channel(**case) once per entry in `cases`.

    Inputs:
        cases : list of dicts, each with keys 'geometry' and 'conditions' (as
                run_channel expects) plus, optionally, any of 'htc', 'friction',
                'bundle', 'fuel_conductivity' to override run_channel's defaults for
                that case -- so the same geometry/conditions repeated across several
                cases with only the correlation keys varying is "one case across
                several correlations" in one call.
    Returns:
        list of dicts, same length and order as `cases`. Each entry is either
        run_channel's normal return with an added 'case_index' key, or, if that case
        raised, {'case_index': i, 'error': str(exception)}.
    """
    results = []
    for i, case in enumerate(cases):
        kwargs = {k: v for k, v in case.items() if k not in ("geometry", "conditions")}
        try:
            out = run_channel(case["geometry"], case["conditions"], **kwargs)
            out["case_index"] = i
        except (ValueError, NotImplementedError, KeyError) as exc:
            out = dict(case_index=i, error=str(exc))
        results.append(out)
    return results


def read_cases_csv(path, geom_type):
    """
    Read many cases from a CSV file, in the spirit of
    docs/reference_code/SCA_Example.py's spreadsheet-driven input.

    Why this model is here:
        docs/PHASE5_BRIEF.md section 3 asks for "a small spreadsheet reader in the
        spirit of docs/reference_code/SCA_Example.py". openpyxl is not installed, and
        docs/DECISIONS.md/docs/PHASE5_BRIEF.md do not approve adding it, so this reads
        .csv only (pandas.read_csv) -- an .xlsx file needs to be exported to .csv first,
        or openpyxl added as an explicit, owner-approved dependency before a
        pandas.read_excel path is added here.

    Formulation:
        Not a physical model. Each row becomes one case dict for run_batch(): columns
        named 'htc'/'friction'/'bundle'/'fuel_conductivity' (if present) become that
        row's correlation-selection override, and every other column becomes a
        geometry/condition value, keyed by _rod_inputs'/_annular_inputs' own key names
        (a column with a name neither function recognizes is silently unused by
        run_channel's dispatch, exactly as an unused dict entry would be -- this reader
        does no key validation of its own beyond what run_channel already does).

    Inputs:
        path      : path to a .csv file
        geom_type : 'rod' or 'annular' -- every row in one file is the same geometry
                    family; mixing geometries in one file is not supported here (it
                    would need a per-row 'type' column and is not built)
    Returns:
        list of case dicts, each with keys 'geometry', 'conditions', and any of
        'htc'/'friction'/'bundle'/'fuel_conductivity' present as columns -- suitable
        for run_batch() directly.
    """
    if geom_type not in ("rod", "annular"):
        raise ValueError(f"read_cases_csv: geom_type must be 'rod' or 'annular', got {geom_type!r}")

    df = pd.read_csv(path)
    corr_cols = [c for c in ("htc", "friction", "bundle", "fuel_conductivity") if c in df.columns]
    value_cols = [c for c in df.columns if c not in corr_cols]

    cases = []
    for _, row in df.iterrows():
        values = {col: row[col] for col in value_cols}
        if geom_type == "rod":
            geometry = {"type": "rod", **{k: values[k] for k in _ROD_GEOM_KEYS if k in values}}
            conditions = {k: values[k] for k in _ROD_COND_KEYS if k in values}
        else:
            geometry = {"type": "annular", **values}
            conditions = {}
        case = dict(geometry=geometry, conditions=conditions)
        for col in corr_cols:
            case[col] = row[col]
        cases.append(case)
    return cases


if __name__ == "__main__":
    example = run_channel(
        geometry={"type": "rod", "pitch": 0.0125, "rco": 0.0045, "tc": 0.00063,
                  "delta": 5e-4, "kc": 24},
        conditions={"G": 1200, "pval": 25, "Tin": 300 + 273.15, "q0": 25e3, "N": 400},
    )
    print(f"rod case: converged={example['convergence']['ok']}  "
          f"peak T_fuel_max={max(example['result']['T_fuel_max']):.1f} K")
    print("notes:", example["notes"] or "(none)")

    batch = run_batch([
        dict(geometry={"type": "annular"}, conditions={}, htc="swenson"),
        dict(geometry={"type": "annular"}, conditions={}, htc="not_a_real_model"),
    ])
    for entry in batch:
        if "error" in entry:
            print(f"case {entry['case_index']}: FAILED -- {entry['error']}")
        else:
            print(f"case {entry['case_index']}: converged={entry['convergence']['ok']}  "
                  f"outer_converged={entry['convergence']['outer_converged']}")
