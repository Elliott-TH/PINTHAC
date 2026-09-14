"""
solve_field(use_lut=True) against solve_field(use_lut=False), same case, same seed of
inputs -- the accuracy-for-speed trade, measured rather than assumed.

Run: HIP_VISIBLE_DEVICES=0 python docs/scripts/compare_annular_lut.py

Recorded output is pasted at the bottom of this file.
"""
import time

import numpy as np

from pinthac.sca import annular

INP = dict(ri=0.0035, ro=0.0055, tci=0.0006, tco=0.0006,
           delta_i=8e-5, delta_o=8e-5, Pitch=0.0136,
           L=4.27, N=40, Tin_i=553.15, Tin_o=553.15, Pnom=25.0,
           mdot_i=0.01, mdot_o=0.06, q0=10.0e3, Gas="He")

FIELDS = ("Tm_i", "Tm_o", "Tfo_i", "Tfo_o", "Tcldi_ID", "Tcldo_OD",
          "q_i", "q_o", "dP_i", "dP_o")


def run(use_lut):
    t0 = time.perf_counter()
    out = annular.solve_field(INP, use_lut=use_lut)
    return out, time.perf_counter() - t0


def main():
    print(f"case: N={INP['N']} nodes, q0={INP['q0']/1e3:.0f} kW/m, Pnom={INP['Pnom']} MPa\n")

    # Build the table outside the timed region is NOT done here on purpose: the reported
    # LUT time includes its own one-off table build, which is the honest comparison for
    # a single run. Reuse it via lut= and a sweep amortises that away entirely.
    live, t_live = run(False)
    print(f"live IAPWS-95   {t_live:8.2f} s   "
          f"{live['outer_iters_used']} outer iterations, converged={live['outer_converged']}")

    lut, t_lut = run(True)
    print(f"lookup table    {t_lut:8.2f} s   "
          f"{lut['outer_iters_used']} outer iterations, converged={lut['outer_converged']}")
    print(f"speedup         {t_live/t_lut:8.1f}x\n")

    print(f"{'field':>10}  {'max abs diff':>14}  {'max rel diff':>14}   unit")
    for f in FIELDS:
        a = np.asarray(live[f], dtype=float)
        b = np.asarray(lut[f], dtype=float)
        abs_d = np.max(np.abs(b - a))
        rel_d = np.max(np.abs(b - a)/np.maximum(np.abs(a), 1e-30))
        unit = "K" if f.startswith(("T", "t")) else ("W/m" if f.startswith("q") else "Pa")
        print(f"{f:>10}  {abs_d:14.6g}  {100*rel_d:13.4f}%   {unit}")

    # The number a designer actually reads off this solver.
    print(f"\npeak Tfo_i   live {np.max(live['Tfo_i']):8.3f} K   "
          f"lut {np.max(lut['Tfo_i']):8.3f} K   diff {np.max(lut['Tfo_i'])-np.max(live['Tfo_i']):+.3f} K")
    print(f"peak Tfo_o   live {np.max(live['Tfo_o']):8.3f} K   "
          f"lut {np.max(lut['Tfo_o']):8.3f} K   diff {np.max(lut['Tfo_o'])-np.max(live['Tfo_o']):+.3f} K")
    print(f"outlet Tm_i  live {live['Tm_i'][-1]:8.3f} K   "
          f"lut {lut['Tm_i'][-1]:8.3f} K   diff {lut['Tm_i'][-1]-live['Tm_i'][-1]:+.3f} K")
    print(f"outlet Tm_o  live {live['Tm_o'][-1]:8.3f} K   "
          f"lut {lut['Tm_o'][-1]:8.3f} K   diff {lut['Tm_o'][-1]-live['Tm_o'][-1]:+.3f} K")


if __name__ == "__main__":
    main()

"""
Real output (HIP_VISIBLE_DEVICES=0 python docs/scripts/compare_annular_lut.py,
GenEnv3.12, torch 2.9.1+rocm7.2.1):

case: N=40 nodes, q0=10 kW/m, Pnom=25.0 MPa

live IAPWS-95     405.13 s   7 outer iterations, converged=True
lookup table        2.72 s   8 outer iterations, converged=True
speedup            149.0x

     field    max abs diff    max rel diff   unit
      Tm_i        0.125812         0.0209%   K
      Tm_o        0.116746         0.0203%   K
     Tfo_i       0.0926932         0.0157%   K
     Tfo_o       0.0976817         0.0167%   K
  Tcldi_ID        0.111957         0.0171%   K
  Tcldo_OD        0.107974         0.0195%   K
       q_i          3.9016         2.0534%   W/m
       q_o          3.9016         0.6442%   W/m
      dP_i         3.53302         0.0134%   Pa
      dP_o        0.733154         0.0305%   Pa

peak Tfo_i   live  696.054 K   lut  696.071 K   diff +0.017 K
peak Tfo_o   live  674.922 K   lut  674.940 K   diff +0.018 K
outlet Tm_i  live  654.590 K   lut  654.539 K   diff -0.051 K
outlet Tm_o  live  613.476 K   lut  613.460 K   diff -0.017 K

Reading this: every temperature the solver reports agrees to better than 0.13 K, and
the peak fuel surface temperatures -- the numbers a designer actually reads off -- to
0.02 K. That is far inside the +/- 25 percent model-form uncertainty of the Swenson
correlation driving them, so the table is not the limiting error in this solve by any
margin.

The flux split shows the largest relative difference (2.05 percent on q_i) but the same
3.9 W/m absolute difference as q_o, which is what conservation requires -- q_i is simply
the smaller of the two, so the same absolute difference reads larger against it.

One honest caveat on attribution: the two runs converged in 7 and 8 outer iterations
respectively, both to the same 10 J/kg tolerance. Part of the difference above is
therefore the two solves landing on slightly different points inside that tolerance
band, not interpolation error alone. The interpolation error by itself is smaller than
what this comparison can resolve.
"""
