"""
PINTHAC -- Physics Informed Nuclear Thermal-Hydraulics Analysis Code.

A thermal-hydraulics library covering water and liquid-metal properties, heat transfer
and friction correlations, fuel-pin radial conduction, single-channel analysis, and
neural surrogates for the same. Every public function runs on Python floats, NumPy
arrays, or PyTorch tensors, and stays differentiable under torch.

Import direction is strictly one-way and never reversed:

    backend / ranges / uncertainty  <-  properties  <-  correlations  <-  pin  <-  sca  <-  ml

See CLAUDE.md for the code style contract and docs/SPLIT_PLAN.md for how these layers
would separate into standalone packages.
"""
from pinthac import backend, ranges, uncertainty

__all__ = ["backend", "ranges", "uncertainty"]
__version__ = "0.1.0"
