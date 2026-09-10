"""
PINTHAC pin layer: radial heat transfer through a fuel pin.

The layer is split by physical mechanism -- gap, clad, solid pellet, annular pellet --
because those are the pieces a reader looks for, and because a solver usually needs two
or three of them rather than all four. The names are re-exported here so callers can
reach the whole layer with one import, which is how the pre-cleanup code used PinHT.py
and what the single-channel solvers still expect.
"""
from pinthac.pin.gap import htc_gap
from pinthac.pin.clad import T_ci
from pinthac.pin.cylindrical import Cyl_HT
from pinthac.pin.annular import Ann_Theta, Ann_HT, Ann_qpp

__all__ = ["htc_gap", "T_ci", "Cyl_HT", "Ann_Theta", "Ann_HT", "Ann_qpp"]
