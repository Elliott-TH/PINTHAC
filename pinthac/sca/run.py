"""sca/run.py -- the top-level single-channel-analysis driver, and the only file most
callers need.

run_channel(geometry, conditions, htc=..., friction=..., bundle=...,
fuel_conductivity=...) takes plain dicts, dispatches on geometry['type'] to sca/rod.py
or sca/annular.py, and returns the axial solution plus a convergence report. run_batch
runs many cases and isolates a failing one; read_cases_csv builds cases from a
spreadsheet.

Correlation selection is four dict literals below mapping name -> function, looked up by
_select(), which fails immediately on an unknown name and lists the valid ones. No
registry, no auto-discovery, no config language -- that is the whole mechanism. Whether
a heat transfer correlation needs an implicit wall-temperature solve is a property of
the correlation, not of the solver, so it is recorded in each solver's own
_HTC_DISPATCH (implicit) / _EXPLICIT_HTC (explicit) tables rather than here.

Two things this driver will accept and should not, both flagged at their definitions
below:

  - Two-phase (PWR/BWR) htc names. Chen, Bjorge and Schrock-Grossman are real,
    implemented correlations, but neither solver has onset-of-nucleate-boiling
    detection, quality/void-fraction tracking, or subcooled-boiling bookkeeping -- both
    march a single-phase enthalpy balance. run_channel raises NotImplementedError rather
    than pretending to run a boiling channel that does not exist.

  - A correlation that does not match the coolant. Both solvers now take coolant=, so
    running a sodium correlation against water (or Swenson against sodium) is a real
    mistake rather than the only thing available, and is rejected by name. See
    HTC_FAMILY.
"""
import numpy as np
import pandas as pd

from pinthac.correlations import bundle as bundle_mod
from pinthac.correlations import friction as fric
from pinthac.correlations import htc
from pinthac.properties import matmod
from pinthac.sca import annular, coolant as coolant_mod, rod


# One dict literal per category, name -> function. fuel_conductivity maps to a
# (k_func, Theta_func) pair because pin.cylindrical.Cyl_T needs the conductivity and its
# integral together -- Theta to state the problem, k as its analytic derivative for the
# Newton polish.
HTC_MODELS = {
    "swenson": htc.SCW.Swenson_dT,
    "chen_scw": htc.SCW.Chen_SCW_dT,
    "chen_h2o": htc.Water.Chen_H2O_dT,
    "bjorge": htc.Water.Bjorge_dT,
    "schrock_grossman": htc.Water.SchrockGrossman,
    "dittus":htc.Water.Dittus,
    "gnielinski":htc.Water.Gnielinski,
    "petukhov":htc.Water.Petchukov,
    "lyon":htc.Sodium.Lyon,
    "seban":htc.Sodium.SebanShimazaki,
    "lead_shen":htc.Lead.Shen,
    "mikityuk":htc.Sodium.Mikityuk,
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

_TWO_PHASE_HTC = ("chen_h2o", "bjorge", "schrock_grossman")

# Which coolant family each correlation was fitted for. This is what stops a liquid-metal
# correlation being run against water, which used to be selectable and silently produced
# a plausible-looking number: Lyon on a water rod returned a peak fuel temperature within
# 1.5 K of Swenson's, with no error and no warning.
#
# The split is not arbitrary. Dittus-Boelter and its relatives are validated for
# 0.7 < Pr < 160; a liquid metal sits near Pr = 0.005, three orders of magnitude below
# the bottom of that range, where the thermal boundary layer is far thicker than the
# velocity one and the Nusselt number stops following Re^0.8 Pr^0.4 at all. The
# liquid-metal correlations are the Peclet-number fits built for that regime, and they
# are equally wrong applied to water.
HTC_FAMILY = {
    "swenson": "water", "chen_scw": "water",
    "dittus": "water", "gnielinski": "water", "petukhov": "water",
    "chen_h2o": "water", "bjorge": "water", "schrock_grossman": "water",
    "lyon": "liquid_metal", "seban": "liquid_metal",
    "mikityuk": "liquid_metal", "lead_shen": "liquid_metal",
}

# Filonenko and Wu are supercritical-water fits; the other three are Reynolds-number
# correlations for any single-phase turbulent flow in a smooth channel.
FRICTION_FAMILY = {
    "filonenko": "water", "wu": "water",
    "blasius": "any", "mcadams": "any", "colebrook": "any",
}

# What htc= and friction= resolve to when left as None, per coolant family.
FAMILY_DEFAULTS = {
    "water": dict(htc="swenson", friction="filonenko"),
    "liquid_metal": dict(htc="lyon", friction="blasius"),
}

_ROD_GEOM_KEYS = ("pitch", "rco", "tc", "delta", "kc")
# Every physical key is required, with no default to fall back on -- see run_channel.
_ANNULAR_GEOM_KEYS = ("ri", "ro", "tci", "tco", "delta_i", "delta_o", "Pitch")
_ANNULAR_COND_KEYS = ("L", "Tin_i", "Tin_o", "Pnom", "mdot_i", "mdot_o", "q0")
_ROD_COND_KEYS = ("G", "pval", "Tin", "q0", "L", "N")


def _select(name, table, category):
    """Look up a correlation by name, raising immediately with the valid options on a miss.

    The single point every correlation-selection keyword passes through. No fallback
    guessing, no partial matching -- an unknown name raises here, listing the options,
    rather than becoming a silently different physics model three figures later.

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


def _scan_for_nonfinite(result, z_key, fields):
    """Find the first axial node where any tracked field is non-finite (NaN or +/-inf).

    Neither solver raises on a physically nonsensical case: bisect_newton runs a fixed
    iteration count and returns whatever it has, and a diverging Picard iterate comes out
    as inf. Both propagate quietly into the output arrays. This turns that into a
    located failure -- which field, which node, which axial position -- instead of a
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
    first_bad = None
    for field in fields:
        values = np.asarray(result[field], dtype=float)
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            i = int(bad[0])
            if first_bad is None or i < first_bad[0]:
                first_bad = (i, field)
    if first_bad is not None:
        i, field = first_bad
        return dict(ok=False, node=i, z=float(z[i]), field=field,
                    message=f"{field} is non-finite (NaN or inf) at node {i} "
                            f"(z = {z[i]:.4f} m) -- first occurrence; there may be more.")
    return dict(ok=True, node=None, z=None, field=None,
                message="every tracked field is finite at every axial node")


def _rod_inputs(geometry, conditions):
    """Assemble sca/rod.py's inputs dict and run_SCA() keyword arguments from a
    geometry dict (pitch, rco, tc, delta, kc) and a conditions dict (G, pval, Tin, q0,
    and optionally L, N). Raises ValueError listing what is missing rather than letting
    rod.run_SCA fail on a KeyError with no context.
    """
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
    """Assemble the flat dict sca/annular.py's solve_field takes, which carries both
    geometry and operating conditions in one mapping. Every physical key is required --
    see _ANNULAR_GEOM_KEYS above for why there is no default case to fall back on. Only
    the two discretization/material choices with an obvious default (N, Gas) are filled
    in.
    """
    merged = dict(geometry)
    merged.update(conditions)
    missing = [k for k in _ANNULAR_GEOM_KEYS + _ANNULAR_COND_KEYS if k not in merged]
    if missing:
        raise ValueError(
            f"annular channel: missing keys {missing} -- required geometry: "
            f"{_ANNULAR_GEOM_KEYS}, required conditions: {_ANNULAR_COND_KEYS}"
        )
    merged.setdefault("N", 100)
    merged.setdefault("Gas", "He")
    return merged


def run_channel(geometry, conditions, htc=None, friction=None,
                 bundle="presser", fuel_conductivity="klimenko", coolant="scw", annular_method="march",
                 **solver_kwargs):
    """Run one single-channel-analysis case. The module's main entry point.

    Not a physical model -- a dispatcher. geometry['type'] selects rod.run_SCA or
    annular.solve_field, geometry/conditions are reshaped into that solver's own input
    format by _rod_inputs/_annular_inputs, and the four correlation names are resolved
    against the tables above and passed straight through.

    Inputs:
        geometry   : dict with 'type' ('rod' or 'annular') plus that geometry's keys --
                     rod: pitch, rco, tc, delta, kc [m, m, m, m, W/m-K]
                     annular: ri, ro, tci, tco, delta_i, delta_o, Pitch [m], plus
                     optional Gas (default "He"). All are required; a missing key raises
                     naming it, because a defaulted geometry silently standing in for a
                     forgotten one is how a run reports somebody else's pin.
        conditions : rod: G [kg/m^2-s], pval [MPa], Tin [K], q0 [W/m], optional L [m]
                     and N; annular: L, Tin_i, Tin_o, Pnom, mdot_i, mdot_o, q0, optional
                     N. q0 is the peak of the default cosine axial shape.
        coolant    : coolant name -- "scw" (default), "water", "sodium", "lead", "lbe".
                     See sca/coolant.py. Both channels of an annular case carry the same
                     coolant. Pressure is unused for a liquid metal.
        htc, friction, bundle, fuel_conductivity : selection by name; see the tables
                     above for valid names. htc and friction default to None, meaning
                     "the standard choice for this coolant" -- swenson/filonenko for
                     water, lyon/blasius for a liquid metal. A correlation that does not
                     match the coolant is rejected, see HTC_FAMILY.
                     bundle=None means no bundle correction and is always accepted.
        **solver_kwargs : passed straight through to the dispatched solver -- rod:
                     scw_table, device; annular: q_p, outer_iter, tol, progress, and
                     use_lut/lut for the property lookup table. This is why run_channel
                     does not need to name every solver knob itself.
        annular_method : 'march' (default) solves the radial balance cell by cell;
                     'picard' retains the whole-field solver for comparisons.
    Returns:
        dict: result (the underlying run_SCA()/solve_field() output), geom_type,
        requested (the four correlation names as given), notes (list of strings, always
        empty today -- kept in the return shape in case a future selection category
        again can't be fully honored by the dispatched solver), convergence (dict, see
        _scan_for_nonfinite). Annular march adds radial_converged/radial_residual;
        Picard adds outer_converged/outer_residual/outer_iters_used.
        result includes axial convection HTC [W/m²/K]: htc_conv for rods,
        htc_conv_i/htc_conv_o for annular channels. Annular gap HTC profiles
        are htc_gap_i/htc_gap_o; each profile aligns with result['z'].
    """
    geom_type = geometry.get("type")
    if geom_type not in ("rod", "annular"):
        raise ValueError(
            f"run_channel: geometry['type'] must be 'rod' or 'annular', got {geom_type!r}"
        )

    family = coolant_mod.resolve(coolant)["family"]
    defaults = FAMILY_DEFAULTS[family]
    htc = defaults["htc"] if htc is None else htc
    friction = defaults["friction"] if friction is None else friction

    requested = dict(htc=htc, friction=friction, bundle=bundle,
                      fuel_conductivity=fuel_conductivity, coolant=coolant)
    if geom_type == "annular":
        requested["annular_method"] = annular_method

    # Validate every requested name is at least recognized (a typo or an unsupported
    # name fails immediately, listing the valid options) and resolve it to the actual
    # function(s) rod.py/annular.py need.
    _select(htc, HTC_MODELS, "htc")   # rod/annular dispatch on the name itself
    friction_func = _select(friction, FRICTION_MODELS, "friction")
    bundle_func = _select(bundle, BUNDLE_MODELS, "bundle")
    k_func, Theta_func = _select(fuel_conductivity, FUEL_CONDUCTIVITY_MODELS,
                                  "fuel_conductivity")

    if htc in HTC_FAMILY and HTC_FAMILY[htc] != family:
        raise ValueError(
            f"htc={htc!r} is a {HTC_FAMILY[htc]} correlation but coolant={coolant!r} is "
            f"{family} -- the two do not go together (see HTC_FAMILY for why). Valid "
            f"htc names for this coolant: "
            f"{sorted(k for k, v in HTC_FAMILY.items() if v == family and k not in _TWO_PHASE_HTC)}."
        )
    if FRICTION_FAMILY.get(friction, "any") not in ("any", family):
        raise ValueError(
            f"friction={friction!r} is a {FRICTION_FAMILY[friction]} correlation but "
            f"coolant={coolant!r} is {family}. Valid friction names for this coolant: "
            f"{sorted(k for k, v in FRICTION_FAMILY.items() if v in ('any', family))}."
        )

    if htc in _TWO_PHASE_HTC:
        raise NotImplementedError(
            f"htc={htc!r} is a two-phase correlation, but neither sca/rod.py nor "
            f"sca/annular.py has subcooled-boiling bookkeeping (onset-of-nucleate-"
            f"boiling, quality/void-fraction tracking) -- see this module's docstring. "
            f"Not invented here; use a single-phase htc selection ('swenson' or "
            f"'chen_scw')."
        )

    geom_only = {k: v for k, v in geometry.items() if k != "type"}
    # Kept as an always-present, normally-empty channel for anything a caller should know
    # about a run that is not an error. Every correlation selection is honored, so nothing
    # writes to it today.
    notes = []

    if geom_type == "rod":
        inputs, run_kwargs = _rod_inputs(geom_only, conditions)
        run_kwargs.update(solver_kwargs)
        result = rod.run_SCA(inputs, coolant=coolant, htc_name=htc,
                                      friction_func=friction_func,
                                      bundle_func=bundle_func, k_func=k_func,
                                      Theta_func=Theta_func, **run_kwargs)
        convergence = _scan_for_nonfinite(result, "Z", ("Tm", "htc_conv", "Tco", "Tci", "Tfo", "Tf_max", "dP"))
    else:
        inp = _annular_inputs(geom_only, conditions)
        if annular_method not in ('march', 'picard'):
            raise ValueError("annular_method must be 'march' or 'picard'")
        if annular_method == 'march':
            from pinthac.sca.annular_march import solve_channel
            solver = solve_channel
        else:
            solver = annular.solve_field
        result = solver(inp, coolant=coolant, htc_name=htc,
                                              friction_func=friction_func,
                                              bundle_func=bundle_func, k_func=k_func,
                                              **solver_kwargs)
        convergence = _scan_for_nonfinite(
            result, "z", ("Tm_i", "Tm_o", "Tci_i", "Tco_i", "Tci_o", "Tco_o", "Tfo_i", "Tfo_o",
                         "htc_conv_i", "htc_conv_o", "Tf_max", "C1", "C2", "q_i", "q_o", "dP_i", "dP_o")
        )
        if annular_method == 'march':
            convergence['radial_converged'] = bool(np.all(result['radial_converged']))
            convergence['radial_residual'] = float(np.max(np.abs(result['radial_residual'])))
            convergence['ok'] &= convergence['radial_converged']
        else:
            convergence['outer_converged'] = bool(result['outer_converged'])
            convergence['outer_residual'] = result['outer_residual']
            convergence['outer_iters_used'] = result['outer_iters_used']
            convergence['radial_converged'] = bool(result['radial_converged'])
            convergence['ok'] &= convergence['outer_converged'] and convergence['radial_converged']
            if not convergence['outer_converged']:
                convergence['message'] += ' -- outer enthalpy Picard loop did not converge'

    return dict(result=result, geom_type=geom_type, requested=requested,
                convergence=convergence, notes=notes)


def run_batch(cases):
    """Run many cases, or one case repeated across several correlation selections, in one
    call.

    One bad case -- a typo'd correlation name, a geometry missing a key -- is recorded
    and skipped rather than killing every other case in a sweep. This is the module's
    only try/except, and the documented exception to CONTRIBUTING.md section 2's ban on
    exceptions as control flow: it catches an actual exception at a batch boundary to
    isolate one case, which is exactly what run_channel raises to signal.

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
        except (ValueError, NotImplementedError, KeyError, RuntimeError) as exc:
            out = dict(case_index=i, error=str(exc))
        results.append(out)
    return results


def read_cases_csv(path, geom_type):
    """Read many cases from a CSV file, in the spirit of
    spreadsheet-driven input.

    Excel files must be exported to CSV before reading.

    Each row becomes one case dict for run_batch(): columns
        named 'htc'/'friction'/'bundle'/'fuel_conductivity'/'coolant' (if present) become that
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
        'htc'/'friction'/'bundle'/'fuel_conductivity'/'coolant' present as columns -- suitable
        for run_batch() directly.
    """
    if geom_type not in ("rod", "annular"):
        raise ValueError(f"read_cases_csv: geom_type must be 'rod' or 'annular', got {geom_type!r}")

    df = pd.read_csv(path)
    corr_cols = [c for c in ("htc", "friction", "bundle", "fuel_conductivity", "coolant") if c in df.columns]
    value_cols = [c for c in df.columns if c not in corr_cols]

    cases = []
    for _, row in df.iterrows():
        values = {col: row[col] for col in value_cols if pd.notna(row[col])}
        if geom_type == "rod":
            geometry = {"type": "rod", **{k: values[k] for k in _ROD_GEOM_KEYS if k in values}}
            conditions = {k: values[k] for k in _ROD_COND_KEYS if k in values}
        else:
            geometry = {"type": "annular", **values}
            conditions = {}
        case = dict(geometry=geometry, conditions=conditions)
        for col in corr_cols:
            if pd.notna(row[col]):
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
