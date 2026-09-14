"""Count and time the property-evaluation calls inside one annular solve.
Small case (N=20, outer_iter=3) so it finishes quickly; the per-call cost and the
call count per (node, outer iteration) are what scale, not the absolute time."""
import time, numpy as np
from pinthac.properties import getprop as gp
from pinthac.sca import annular

stats = {"getprop_calls": 0, "getprop_points": 0, "getprop_time": 0.0,
         "Thp_calls": 0, "Thp_points": 0, "Thp_time": 0.0}

_real_getprop = gp._getprop
def counting_getprop(substance, T, P, *a, **k):
    t0 = time.perf_counter()
    out = _real_getprop(substance, T, P, *a, **k)
    stats["getprop_time"] += time.perf_counter() - t0
    stats["getprop_calls"] += 1
    stats["getprop_points"] += int(np.size(T))
    return out
gp._getprop = counting_getprop
annular.gp._getprop = counting_getprop

_real_Thp = annular._T_hp_fast
def counting_Thp(h, p, **k):
    t0 = time.perf_counter()
    out = _real_Thp(h, p, **k)
    stats["Thp_time"] += time.perf_counter() - t0
    stats["Thp_calls"] += 1
    stats["Thp_points"] += int(np.size(h))
    return out
annular._T_hp_fast = counting_Thp

INP = dict(ri=0.0035, ro=0.0055, tci=0.0006, tco=0.0006,
           delta_i=8e-5, delta_o=8e-5, Pitch=0.0136,
           L=4.27, N=20, Tin_i=553.15, Tin_o=553.15, Pnom=25.0,
           mdot_i=0.01, mdot_o=0.06, q0=10.0e3, Gas="He")

t0 = time.perf_counter()
out = annular.solve_field(INP, outer_iter=3, tol=10.0)
total = time.perf_counter() - t0

print(f"N = {INP['N']} nodes, 3 outer Picard iterations")
print(f"total wall clock                  {total:8.2f} s")
print(f"  _T_hp_fast   {stats['Thp_calls']:5d} calls  "
      f"{stats['Thp_points']:8d} pts  {stats['Thp_time']:8.2f} s  "
      f"({100*stats['Thp_time']/total:5.1f}%)")
print(f"  _getprop     {stats['getprop_calls']:5d} calls  "
      f"{stats['getprop_points']:8d} pts  {stats['getprop_time']:8.2f} s  "
      f"({100*stats['getprop_time']/total:5.1f}%)")
tot_eos = stats['Thp_time'] + stats['getprop_time']
print(f"  EOS total                       {tot_eos:8.2f} s  ({100*tot_eos/total:5.1f}%)")
print(f"  everything else                 {total-tot_eos:8.2f} s")
print()
print(f"_getprop calls per outer iteration: {stats['getprop_calls']/3:.0f}")
print(f"mean points per _getprop call:      {stats['getprop_points']/max(stats['getprop_calls'],1):.1f}")
print(f"mean cost per _getprop call:        {1e3*stats['getprop_time']/max(stats['getprop_calls'],1):.2f} ms")
