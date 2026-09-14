import numpy as np
import pandas as pd
import time
from pinthac.properties import getprop as prop
from pinthac.paths import data_file

# Generates the 2-D (T, P) supercritical-water property table that sca/lut.py reads.
# Legacy: nothing in the live solvers uses it. sca/annular.py's use_lut builds its own
# 1-D table in T at a fixed pressure instead (annular.build_scw_lut).


def build_table():
    """
    Generate the supercritical-water property lookup table and write it to the data
    directory.

    501 temperatures x 9 pressures. The pressure points exist so sca/lut.py can look
    properties up at the local pressure as it drops along the channel -- the one thing
    neither live solver does.

    Behind a function and a __main__ guard because this used to run at module scope,
    where merely importing the module regenerated 4509 rows over the existing table.
    Regeneration is reproducible in physics but not byte-for-byte (GPU reduction ordering
    moves the last significant figure or two), so nothing should assert on an exact value.

    Returns:
        path : absolute path of the CSV written, string
    """
    T_min, T_max, dT = 500.0, 1000.0, 1.0     # [K]
    # NOTE: IAPWS95.rho_Tp's bisection loses the liquid-density root below
    # ~T=480K at these pressures (converges to a spurious root near rho_c
    # instead) -- T_min=500 stays clear of that with margin. Fine for SCA
    # use (Tin ~660K), but don't lower T_min without first checking rho_Tp.
    P_list = np.arange(23.0, 27.5, 0.5)       # [MPa], brackets Pnom=25 MPa

    T = np.arange(T_min, T_max + dT, dT)

    rows = []
    t0 = time.time()
    for P in P_list:
        props = prop._getprop('SCW', T, float(P))
        rows.append(pd.DataFrame({
            'T': T,
            'P': np.full_like(T, P),
            'rho': props['rho'],
            'h': props['h'],
            'cp': props['cp'],
            'mu': props['mu'],
            'k': props['k'],
        }))
        print(f'  P = {P:.2f} MPa done ({time.time()-t0:.1f} s elapsed)')

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(data_file('SCW_Prop_Table.csv'), index=False)
    print(f'Wrote {len(df)} rows ({len(T)} T points x {len(P_list)} P points) '
          f'to SCW_Prop_Table.csv in {time.time()-t0:.1f} s')
    return data_file('SCW_Prop_Table.csv')


if __name__ == '__main__':
    build_table()
