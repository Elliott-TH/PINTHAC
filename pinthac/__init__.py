"""PINTHAC -- Physics Informed Nuclear Thermal-Hydraulics Analysis Code.

A thermal-hydraulics library covering water and liquid-metal properties, heat transfer
and friction correlations, fuel-pin radial conduction, single-channel analysis, and
neural surrogates for the same. Property and correlation functions support Python floats, NumPy arrays,
and PyTorch tensors; gradient support varies by solver.

Import direction is strictly one-way and never reversed:

    backend / ranges / uncertainty  <-  properties  <-  correlations  <-  pin  <-  sca  <-  ml

See CONTRIBUTING.md for development conventions.
"""
from pinthac import backend, ranges, uncertainty

__all__ = ["backend", "ranges", "uncertainty"]
__version__ = "0.1.0"
