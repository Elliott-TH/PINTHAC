"""Shared physics-residual terms for the ml/ surrogates.

The rod DeepONet and annular PINN share the coolant energy balance,
residual normalization, and network-output positivity constraints below.

Reference: Annular_Heat_Transfer_Final.pdf, section 2.1.
Equation (4) there (pressure-drop and friction neglected) is

    G * dh/dz = (1/A_z) * q'(z)   =>   mdot * dh/dz = q'(z)

which is coolant_energy_residual_h below. Non-dimensionalizing
(h -> h_in + h* h0, z -> z* L) gives

    dh*/dz* = (<q'> L)/(mdot h0) * Fq(z* L)

-- the same residual, just rescaled by a characteristic enthalpy/length
pair so the loss magnitude does not depend on the raw SI units. That
rescaling is what normalized_residual_loss does generically, given
whatever scale a caller supplies.
"""
import torch


def coolant_energy_residual_h(mdot, dh_dz, qp):
    """Coolant channel energy balance, enthalpy form.

    Formulation:
        mdot * dh/dz = q'(z)
        res = mdot*dh_dz - qp

    Reference:
        Annular_Heat_Transfer_Final.pdf section 2.1, eq. (4).

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
    """Coolant channel energy balance, temperature form.

    Formulation:
        mdot * cp * dT/dz = q'(z)
        res = mdot*cp*dT_dz - qp

    Reference:
        Annular_Heat_Transfer_Final.pdf section 2.1, eq. (4),
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
    """Mean-squared residual, non-dimensionalized by a characteristic scale.

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
    """Soft penalty keeping a network output above a floor.

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
