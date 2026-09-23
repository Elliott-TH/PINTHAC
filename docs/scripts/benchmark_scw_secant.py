"""Compare the Swenson wall solve with an external q,Tw CSV (Tw in Celsius)."""
import argparse
import time

import numpy as np
import torch

from pinthac.correlations import htc
from pinthac.properties import getprop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', help='CSV with q [W/m²] and Tw [°C] columns')
    parser.add_argument('--pressure', type=float, default=27., help='MPa')
    parser.add_argument('--bulk-c', type=float, default=380.)
    parser.add_argument('--mass-flux', type=float, default=800.)
    parser.add_argument('--diameter', type=float, default=.007)
    parser.add_argument('--max-wall-c', type=float, default=1400.)
    parser.add_argument('--scan-points', type=int, default=500)
    args = parser.parse_args()
    table = np.genfromtxt(args.table, delimiter=',', names=True)
    q = np.atleast_1d(table['q'])
    Tb = np.full_like(q, args.bulk_c+273.15)
    properties = lambda T: getprop._getprop('SCW', T, args.pressure)
    original = htc._bracketed_secant
    recorded = []

    def measured(residual, lo, hi, **kwargs):
        result = original(residual, lo, hi, **kwargs)
        recorded.append((residual, lo.clone(), hi.clone(), result))
        return result

    start = time.perf_counter()
    htc._bracketed_secant = measured
    try:
        state = htc.SCW.Swenson(
            properties(Tb), properties, args.mass_flux, args.diameter, q, Tb,
            hi=args.max_wall_c+273.15, branch_n=args.scan_points, return_state=True)
    finally:
        htc._bracketed_secant = original
    elapsed = time.perf_counter()-start
    residual, lo, hi, result = recorded[0]
    with torch.no_grad():
        fl = residual(lo)
        for _ in range(36):
            middle = (lo+hi)/2
            fm = residual(middle)
            same = torch.signbit(fl) == torch.signbit(fm)
            lo, fl = torch.where(same, middle, lo), torch.where(same, fm, fl)
            hi = torch.where(same, hi, middle)
        difference = float((result.root-(lo+hi)/2).abs().max())
    error = np.abs(state['Tw']-273.15-table['Tw'])
    iterations, counts = np.unique(result.iterations.cpu().numpy(), return_counts=True)
    print(f'Cases: {len(q)}; solve including scan: {elapsed:.3f} s')
    print('Iterations (count):', dict(zip(iterations.tolist(), counts.tolist())))
    print(f'Table temperature error: mean {error.mean():.6g} K; max {error.max():.6g} K')
    print(f'Secant/bisection maximum difference: {difference:.6g} K')
    print('Maximum normalized flux residual:',
          np.max(np.abs(state['residual'])/np.maximum(np.abs(q), 1.)))


if __name__ == '__main__':
    main()
