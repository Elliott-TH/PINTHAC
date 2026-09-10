"""
Shared physics-residual terms for the ml/ surrogates.

Both pinthac/ml/deeponet.py (single-stream rod DeepONet-PINN) and
pinthac/ml/pinn.py (two-stream annular PINN) add a physics loss -- the
coolant-channel energy balance evaluated at autograd collocation points --
on top of their data loss. Before this module existed, each wrote that
residual out by hand inline, once in deeponet.py's temperature form and
once in pinn.py's enthalpy form, with no guarantee the two were checking
the same equation. This module gives the energy balance, its
non-dimensionalized form, and a network-output positivity constraint one
definition each, so the two models import the same physics rather than
maintaining two hand-typed copies of it.

Reference: docs/reference/Annular_Heat_Transfer_Final.pdf, section 2.1.
Equation (4) there (pressure-drop and friction neglected) is

    G * dh/dz = (1/A_z) * q'(z)   =>   mdot * dh/dz = q'(z)

which is coolant_energy_residual_h below. Non-dimensionalizing
(h -> h_in + h* h0, z -> z* L) gives

    dh*/dz* = (<q'> L)/(mdot h0) * Fq(z* L)

-- the same residual, just rescaled by a characteristic enthalpy/length
pair so the loss magnitude does not depend on the raw SI units. That
rescaling is what normalized_residual_loss does generically, given
whatever scale a caller supplies.

These are plain functions of torch tensors, matching the flat-namespace /
function-level style the rest of the library uses (CLAUDE.md section 2) --
there is no state to carry, so a class would add nothing.
"""
import torch


def coolant_energy_residual_h(mdot, dh_dz, qp):
    """
    Coolant channel energy balance, enthalpy form.

    Why this model is here:
        The governing ODE every axial march or autograd collocation point
        in this library's coolant channels must satisfy, with pressure
        drop and friction heating neglected (a small correction to the
        enthalpy rise itself -- pin/rod.py's momentum equation, not this
        energy balance, is where pressure drop is tracked when it
        matters). Used as a physics-loss term: driven to zero at random
        collocation points during PINN/DeepONet training, never solved
        for h directly here.

    Formulation:
        mdot * dh/dz = q'(z)
        res = mdot*dh_dz - qp

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 2.1, eq. (4).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        mdot  : coolant mass flow rate, kg/s
        dh_dz : axial enthalpy gradient (autograd d/dz of the network's h
                output), J/kg-m
        qp    : linear heat generation rate q'(z), W/m

    Returns:
        res : residual mdot*dh_dz - qp, W/m, same type as the inputs.
              Zero when the balance holds exactly.
    """
    return mdot * dh_dz - qp


def coolant_energy_residual_T(mdot, cp, dT_dz, qp):
    """
    Coolant channel energy balance, temperature form.

    Why this model is here:
        Identical physics to coolant_energy_residual_h, written for a
        network whose output is bulk temperature T rather than enthalpy h
        (pinthac/ml/deeponet.py's trunk/branch combination predicts T_i
        directly). dh/dz = cp*dT/dz with cp evaluated at the network's
        current bulk state and held fixed with respect to autograd (a
        frozen coefficient, not itself part of the residual's gradient
        path) -- see the call site for why cp is detached there.

    Formulation:
        mdot * cp * dT/dz = q'(z)
        res = mdot*cp*dT_dz - qp

    Reference:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 2.1, eq. (4),
        with dh = cp dT.

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        mdot  : coolant mass flow rate, kg/s
        cp    : coolant isobaric specific heat at the bulk state, J/kg-K
        dT_dz : axial temperature gradient, K/m
        qp    : linear heat generation rate q'(z), W/m

    Returns:
        res : residual mdot*cp*dT_dz - qp, W/m, same type as the inputs.
    """
    return mdot * cp * dT_dz - qp


def normalized_residual_loss(res, scale):
    """
    Mean-squared residual, non-dimensionalized by a characteristic scale.

    Why this model is here:
        A raw physics residual is expressed in whatever SI unit the
        underlying balance carries (W/m for the energy balance above),
        which is not O(1) and would either dominate or vanish next to an
        already-normalized data loss purely as an artifact of unit choice
        -- not because the physics term should count for more or less.
        Both deeponet.py and pinn.py divide their residual by a
        characteristic scale before squaring and averaging; this is that
        shared step, matching the PDF's own non-dimensionalization
        (section 2.1: h -> h*, z -> z*) in spirit, generalized to
        whatever scale a caller supplies (a per-point |q'(z)| clamp in
        deeponet.py, a fixed q0 in pinn.py).

    Formulation:
        loss = mean( (res/scale)^2 )

    Inputs (torch tensors, broadcastable):
        res   : residual, any shape, any unit
        scale : characteristic scale of res's unit, same or broadcastable
                shape, same unit as res, > 0

    Returns:
        loss : scalar tensor
    """
    return torch.mean((res / scale) ** 2)


def positivity_penalty(y, floor=0.0):
    """
    Soft penalty keeping a network output above a floor.

    Why this model is here:
        docs/reference/Annular_Heat_Transfer_Final.pdf section 2.1 states
        the requirement directly: "it is ensured that proper constraints
        are made to ensure positivity of the network outputs" -- needed
        wherever a predicted enthalpy/temperature/LHGR-shape feeds a
        downstream property-table lookup (e.g. torch.searchsorted into an
        increasing h/T table), which is only well-defined above the
        table's lower bound. Provided here as a reusable soft constraint
        for future PINN/DeepONet work; pinthac/ml/pinn.py enforces its
        own positivity a different way (H()'s softplus multiplicative
        form hard-constrains h > h_in by construction, so it carries no
        loss term for this), and pinthac/ml/deeponet.py's target (bulk
        temperature in kelvin) does not need one at the current training
        range -- neither call site is changed by adding this function.

    Formulation:
        violation = max(floor - y, 0)
        loss = mean(violation^2)

    Inputs:
        y     : network output tensor, any shape
        floor : scalar, the value y must stay above, same unit as y

    Returns:
        loss : scalar tensor, 0 wherever y >= floor
    """
    violation = torch.clamp(floor - y, min=0.0)
    return torch.mean(violation ** 2)
