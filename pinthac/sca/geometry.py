"""Channel-geometry helpers shared by the single-channel solvers.

Exact unit-cell geometry, not fitted correlations, so there is no valid range,
uncertainty or reference to state for either function. sca/rod.py's coolant channel and
sca/annular.py's outer channel are both square_pitch_cell; sca/annular.py's inner
channel is circular_channel.
"""
import math


def square_pitch_cell(pitch, R):
    """Flow area, wetted perimeter and hydraulic diameter of a square-pitch rod-bundle unit
    cell: one rod of outer radius R inside a pitch x pitch square.

        A_flow = pitch^2 - pi*R^2
        Per    = 2*pi*R
        Dh     = 4*A_flow/Per

    The wetted perimeter counts only the rod surface. The four square boundaries are
    symmetry planes between adjacent rods -- no shear, no heat flux -- not walls.

    Inputs (float, numpy array, or torch tensor; broadcastable):
        pitch : rod pitch, m
        R     : rod (or clad) outer radius, m
    Returns:
        dict with A_flow [m^2], Per [m], Dh [m], each the same type as the inputs
    """
    A_flow = pitch**2 - math.pi*R**2
    Per = 2*math.pi*R
    Dh = 4*A_flow/Per
    return dict(A_flow=A_flow, Per=Per, Dh=Dh)


def circular_channel(R):
    """Flow area, wetted perimeter and hydraulic diameter of a circular channel of radius R.

        A_flow = pi*R^2
        Per    = 2*pi*R
        Dh     = 4*A_flow/Per = 2*R

    Dh is trivially the channel's own diameter; written in the same A/Per/Dh shape as
    square_pitch_cell so both channels read identically at the call site.

    Inputs (float, numpy array, or torch tensor):
        R : channel radius, m
    Returns:
        dict with A_flow [m^2], Per [m], Dh [m], each the same type as R
    """
    A_flow = math.pi*R**2
    Per = 2*math.pi*R
    Dh = 4*A_flow/Per
    return dict(A_flow=A_flow, Per=Per, Dh=Dh)
