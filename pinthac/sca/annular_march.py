"""Axially marched, dual-cooled annular fuel channel.

Each cell solves the radial heat split at its entering coolant state, then
advances both enthalpies by q_j*dz/mdot_j. No whole-channel Picard iteration.
Radial fields and Tm_i/o share the entering state; T_out_i/o and h_out_i/o are
cell-exit states. z labels the power sampling midpoint of each cell.
"""
import warnings

import numpy as np
import torch
import torchsolve as ts
from scipy.optimize import brentq

from pinthac import pin
from pinthac.properties.matmod import Gas, UO2, Zircalloy
from pinthac.ranges import RangeWarning
from pinthac.correlations import bundle, friction
from pinthac.sca import coolant as coolant_mod, film


def _float(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def radial_node(Ti, To, q, inp, geom, props_at, theta, *, htc_name='swenson',
                bundle_func=bundle.Bundle.Presser, anchor=None, wall_hi=None, wall_lo=None,
                split_tol=1e-4):
    """Bracket the heat split using the Kirchhoff boundary mismatch [W/m].

    Signed heat transfer permits heat exchange between channels near an unheated
    end. Cooling applies the same correlations on the lower-temperature branch;
    this is extrapolation for correlations fitted only to heating (notably Chen).
    All returned surface fields use the same converged heat split.
    """
    ri = inp['ri']
    ro = inp['ro']
    fuel_area = np.pi * (ro**2 - ri**2)
    q3 = q / fuel_area
    if bundle_func:
        psi = bundle_func(inp['Pitch'], 2 * geom['R_clad_o_OD'])
    else:
        psi = 1.
    properties = film.numpy_adapter(props_at)

    def kgas(T):
        return Gas.k(inp.get('Gas', 'He'), T)

    def surface(Tb, power, side):
        # Step 1: convection from the bulk coolant to the wet clad surface.
        inner = side == 'i'
        Rlo = geom[f'R_clad_{side}_ID']
        Rhi = geom[f'R_clad_{side}_OD']
        qpp_wet = power / geom[f'Per_{side}']
        bundle_factor = 1. if inner else psi
        state = film.solve(
            properties, Tb, geom[f'G_{side}'], geom[f'D_{side}'], qpp_wet,
            name=htc_name, psi=bundle_factor, pitch=inp['Pitch'],
            anchor=anchor, hi=wall_hi, lo=wall_lo,
        )
        wet = _float(state['Tw'])
        hc = _float(state['htc'])

        # Step 2: conduction through the clad, using its mean temperature.
        clad_heat = power * np.log(Rhi / Rlo) / (2 * np.pi)

        def clad_res(Tdry):
            Tmean = (Tdry + wet) / 2
            kc = Zircalloy.k(Tmean)
            return (Tdry - wet) * kc - clad_heat

        if power == 0:
            dry = wet
        else:
            lo, hi = sorted((wet, wet + np.sign(power) * 500.))
            dry = brentq(clad_res, max(lo, 285.), min(hi, 1770.), xtol=1e-8)

        # Step 3: conduction and radiation across the gas gap.
        radius = ri if inner else ro
        delta = inp[f'delta_{side}']
        qpp_fuel = power / (2 * np.pi * radius)

        def gap_res(Tfuel):
            htc_gap = pin.htc_gap(Tfuel, dry, delta, kgas)
            return htc_gap * (Tfuel - dry) - qpp_fuel

        if power == 0:
            fuel = dry
        else:
            lo = max(300., 600. - dry)
            hi = min(6000., 5000. - dry)
            fuel = brentq(gap_res, lo, hi, xtol=1e-8)
        htc_gap = pin.htc_gap(fuel, dry, delta, kgas)

        # The inner channel wets the clad ID; the outer channel wets the OD.
        if inner:
            Tci = wet
            Tco = dry
        else:
            Tci = dry
            Tco = wet
        return dict(Tci=Tci, Tco=Tco, Tfo=fuel, htc_conv=hc, htc_gap=htc_gap)

    def evaluate(qi):
        # Step 4: compare the assumed heat split with fuel conduction.
        qi = float(np.asarray(qi))
        qo = q - qi
        si = surface(Ti, qi, 'i')
        so = surface(To, qo, 'o')
        Theta_i = theta(si['Tfo'])
        Theta_o = theta(so['Tfo'])
        C1, C2 = pin.Ann_HT(ri, ro, q3, Theta_i, Theta_o)
        predicted = -2 * np.pi * ri * pin.Ann_qpp(ri, q3, C1)
        residual = predicted - qi
        return float(residual), si, so, float(C1), float(C2)

    if q < 0:
        raise ValueError('annular march requires nonnegative total fuel power')
    # Adjust the assumed inner heat rate until both fuel boundaries agree.
    lo = 0.
    hi = q
    if q == 0 and Ti == To:
        qi = 0.
    else:
        def split_bracket(left, right):
            try:
                fl, fr = evaluate(left)[0], evaluate(right)[0]
                if fl*fr <= 0:
                    return left, right
            except ts.SolverFailure:
                pass
            # An all-inner/all-outer trial can exceed a film model's domain
            # even when a feasible interior heat split exists.
            previous = None
            for trial in np.linspace(left, right, 33):
                try:
                    value = evaluate(trial)[0]
                except ts.SolverFailure:
                    previous = None
                    continue
                if previous is not None and previous[1]*value <= 0:
                    return previous[0], trial
                previous = trial, value
            return None

        bracket = split_bracket(lo, hi)
        if bracket is None:
            # Allow inter-channel heat exchange even at zero generated power.
            width = max(.05*q, 100.)
            for _ in range(12):
                bracket = split_bracket(-width, q+width)
                if bracket is not None:
                    break
                width *= 2
        if bracket is None:
            raise ValueError("no feasible sign-changing annular heat-split bracket")
        lo, hi = bracket
        def split_residual(x):
            mismatch = evaluate(x[0])[0]
            return np.array([mismatch])

        result = ts.solve_numpy(split_residual,
                                bracket=(np.array([lo]), np.array([hi])),
                                backend='brentq', xtol=1e-7, ftol=split_tol, strict=True)
        qi = float(result.root[0])
    residual, si, so, C1, C2 = evaluate(qi)
    # The only interior stationary point is r²=2*C1/q3; include both surfaces
    # so reversed flux and zero generation have the correct boundary maximum.
    candidates = [ri, ro]
    if q3 > 0 and C1 > 0:
        candidates.append(float(np.clip(np.sqrt(2*C1/q3), ri, ro)))
    transformed = []
    for r in candidates:
        Theta_r = -q3 * r**2 / 4 + C1 * np.log(r) + C2
        transformed.append(Theta_r)
    index = int(np.argmax(transformed))
    target = transformed[index]
    def fuel_residual(T):
        return float(theta(T)) - target

    Tmax = brentq(fuel_residual, 300., 6000., xtol=1e-7)
    out = {}
    for side, state in (('i', si), ('o', so)):
        for key, value in state.items():
            out[f'{key}_{side}'] = value
    out.update(Tm_i=Ti, Tm_o=To, q_i=qi, q_o=q-qi, q3=q3, C1=C1, C2=C2,
               Tf_max=Tmax, r_Tf_max=candidates[index], radial_residual=residual)
    out.update(Tcldi_ID=out['Tci_i'], Tcldi_OD=out['Tco_i'],
               Tcldo_ID=out['Tci_o'], Tcldo_OD=out['Tco_o'])
    return out


def solve_channel(Inputs, q_p=None, *, htc_name='swenson',
                  friction_func=friction.f_SCW.Filonenko,
                  bundle_func=bundle.Bundle.Presser, k_func=UO2.k_Klimenko,
                  coolant='scw', use_lut=True, lut=None, progress=False,
                  split_tol=1e-4):
    """March N cells, retaining local balances and radial profile coefficients.

    Theta(T(r)) = -q3*r²/4 + C1*log(r) + C2, with r in metres. The returned
    theta_T/theta_values table defines the transform zero and allows inversion
    during workup without serializing a callable. Temperatures are in kelvin,
    powers q_i/o in W/m, htc in W/m²/K, enthalpies in J/kg, pressure drops in Pa.
    """
    from pinthac.sca import annular
    inp = dict(Inputs)
    N = inp.get('N', 100)
    L = inp['L']
    if int(N) != N or N < 1 or L <= 0:
        raise ValueError('N must be a positive integer and L must be positive')
    N = int(N)
    geom = annular.geometry(inp)
    if inp['mdot_i'] <= 0 or inp['mdot_o'] <= 0:
        raise ValueError('co-current annular march requires positive mass flows')
    entry = coolant_mod.resolve(coolant)
    if entry['tabulated'] and not use_lut:
        from pinthac.properties import getprop
        props_at = lambda T: getprop._getprop(entry['substance'], T, inp['Pnom'])
        T_from_h = lambda h: annular._T_hp_fast(np.asarray(h)/1000, inp['Pnom'])
    else:
        props_at, T_from_h = coolant_mod.make_lookups(coolant, inp['Pnom'], table=lut)
    anchor = film.pseudocritical(props_at) if htc_name in ('swenson', 'chen_scw') else None
    if use_lut and entry['tabulated'] and lut is not None:
        wall_lo, wall_hi = float(lut['T'][0]), float(lut['T'][-1])
    else:
        wall_lo, wall_hi = entry['T_range']
    # As in the existing field solver, construct the transform beyond the fit's
    # validation range; warn only if the solved fuel state exceeds that range.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RangeWarning)
        theta = pin.Ann_Theta(k_func, T_max=6000., n=8000)
    dz = L/N
    z = -L/2 + (np.arange(N)+.5)*dz
    if q_p is None:
        power = inp['q0'] * np.cos(np.pi * z / L)
    else:
        power = np.broadcast_to(q_p(z), z.shape)

    # Initialize both channels at the inlet.
    h_i = float(props_at(inp['Tin_i'])['h'])
    h_o = float(props_at(inp['Tin_o'])['h'])
    rows = []
    htc_conv_i = []
    htc_conv_o = []
    htc_gap_i = []
    htc_gap_o = []
    iterator = range(N)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc='Annular axial march')
    for j in iterator:
        # Step 1: coolant temperatures entering this cell.
        Ti = float(T_from_h(h_i))
        To = float(T_from_h(h_o))
        qp = float(power[j])

        # Step 2: solve the heat split and radial temperatures at this state.
        try:
            row = radial_node(Ti, To, qp, inp, geom, props_at, theta,
                              htc_name=htc_name, bundle_func=bundle_func,
                              anchor=anchor, wall_hi=wall_hi, wall_lo=wall_lo, split_tol=split_tol)
        except (ValueError, ts.SolverFailure) as exc:
            raise RuntimeError(f'annular radial solve failed at cell {j}, z={z[j]:.6g} m: {exc}') from exc
        # Step 3: add each channel's heat to its entering enthalpy.
        h_out_i = h_i + row['q_i'] * dz / inp['mdot_i']
        h_out_o = h_o + row['q_o'] * dz / inp['mdot_o']
        T_out_i = float(T_from_h(h_out_i))
        T_out_o = float(T_from_h(h_out_o))

        # Step 4: save this cell, then use its exits as the next cell's inlets.
        row['h_i'] = h_i
        row['h_o'] = h_o
        row['h_out_i'] = h_out_i
        row['h_out_o'] = h_out_o
        row['T_out_i'] = T_out_i
        row['T_out_o'] = T_out_o
        rows.append(row)
        htc_conv_i.append(row['htc_conv_i'])
        htc_conv_o.append(row['htc_conv_o'])
        htc_gap_i.append(row['htc_gap_i'])
        htc_gap_o.append(row['htc_gap_o'])
        h_i = h_out_i
        h_o = h_out_o

    # Collect the saved cell values into axial arrays.
    out = dict(
        htc_conv_i=np.asarray(htc_conv_i),
        htc_conv_o=np.asarray(htc_conv_o),
        htc_gap_i=np.asarray(htc_gap_i),
        htc_gap_o=np.asarray(htc_gap_o),
    )
    for key in rows[0]:
        if key in out:
            continue
        values = []
        for row in rows:
            values.append(row[key])
        out[key] = np.asarray(values)
    out.update(z=z, qp=power, z_in=z-dz/2, z_out=z+dz/2,
               theta_T=theta.x, theta_values=theta.y, ri=inp['ri'], ro=inp['ro'],
               radial_converged=np.abs(out['radial_residual']) <= split_tol)
    for side in ('i', 'o'):
        # Include the inlet-to-first-outlet segment in cumulative pressure drop.
        temperatures = np.r_[inp[f'Tin_{side}'], out[f'T_out_{side}']]
        out[f'dP_{side}'] = annular.pressure_drop(temperatures, geom[f'G_{side}'],
                            geom[f'D_{side}'], props_at, friction_func, dz)[1:]
    k_func(out['Tf_max'])  # validate the actual final fuel temperatures
    return out


def fuel_profile(result, radii):
    """Reconstruct T_fuel(z,r) from the saved coefficients and transform table."""
    from scipy.interpolate import PchipInterpolator
    r = np.asarray(radii, dtype=float)
    if np.any((r < result['ri']) | (r > result['ro'])):
        raise ValueError('radii must lie within the fuel annulus')
    target = (-result['q3'][:, None]*r**2/4
              + result['C1'][:, None]*np.log(r) + result['C2'][:, None])
    inverse = PchipInterpolator(result['theta_values'], result['theta_T'], extrapolate=False)
    return inverse(target)
