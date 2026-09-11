"""
Shared channel-geometry helpers for the single-channel solvers.

Why this module exists: sca/rod.py's single rod-bundle channel and
sca/annular.py's outer (rod-bundle) channel both compute the same square-pitch
unit-cell hydraulic diameter from a pitch and a rod (or clad) outer radius, by
hand, in two different files; annular.py's inner channel separately computes
the same circular-tube hydraulic diameter (its own diameter) from a channel
radius. docs/brief/PHASE5_BRIEF.md section 3 asks for a shared geometry module --
this is that module, factored out of the two call sites it replaces rather
than written from scratch, so it carries no new physics or formula.
"""
import math


def square_pitch_cell(pitch, R):
    """
    Flow area, wetted perimeter and hydraulic diameter of a square-pitch rod-bundle
    unit cell, for one rod of outer radius R inside a pitch x pitch square.

    Why this model is here:
        The rod-bundle channel geometry shared by sca/rod.py's single coolant channel
        (rod_node) and sca/annular.py's outer channel (geometry()) -- both are a rod (or
        cladding) sitting in a square-pitch unit cell.

    Formulation:
        A_flow = pitch^2 - pi*R^2
        Per    = 2*pi*R
        Dh     = 4*A_flow/Per

    Valid range:
        Not applicable -- exact unit-cell geometry, not a fitted correlation.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable -- standard square-pitch bundle unit-cell geometry (see e.g.
        Todreas & Kazimi, Nuclear Systems Volume 1, chapter 9).

    Inputs (float, numpy array, or torch tensor; broadcastable against each other):
        pitch : rod pitch, m
        R     : rod (or clad) outer radius, m
    Returns:
        dict with keys A_flow (flow area, m^2), Per (wetted perimeter, m),
        Dh (hydraulic diameter, m) -- each the same type as the inputs
    """
    A_flow = pitch**2 - math.pi*R**2
    Per = 2*math.pi*R
    Dh = 4*A_flow/Per
    return dict(A_flow=A_flow, Per=Per, Dh=Dh)


def circular_channel(R):
    """
    Flow area, wetted perimeter and hydraulic diameter of a circular channel of radius R.

    Why this model is here:
        sca/annular.py's inner channel is a plain circular tube bored through the
        cladding. The hydraulic diameter of a circular channel is trivially its own
        diameter, but writing it as A_flow/Per/Dh here keeps both channels' geometry the
        same shape at the call site, alongside square_pitch_cell.

    Formulation:
        A_flow = pi*R^2
        Per    = 2*pi*R
        Dh     = 4*A_flow/Per = 2*R

    Valid range:
        Not applicable -- exact circular-tube geometry, not a fitted correlation.

    Uncertainty:
        Not applicable.

    Reference:
        Not applicable -- standard circular-tube geometry.

    Inputs (float, numpy array, or torch tensor):
        R : channel radius, m
    Returns:
        dict with keys A_flow (flow area, m^2), Per (wetted perimeter, m),
        Dh (hydraulic diameter, m) -- each the same type as R
    """
    A_flow = math.pi*R**2
    Per = 2*math.pi*R
    Dh = 4*A_flow/Per
    return dict(A_flow=A_flow, Per=Per, Dh=Dh)
