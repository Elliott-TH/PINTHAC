import numpy as np
import pandas as pd
import time
from pinthac.properties import getprop as prop
from pinthac.paths import data_file

# =============================================================
# Generates a property lookup table for supercritical water
# (SCW), used by SCA_LUT.py so the channel loop can interpolate
# properties instead of calling the IAPWS-95 solver at every
# axial node. A handful of pressure points bracket the nominal
# operating pressure so the table still resolves properties as
# pressure drops along the channel.
# =============================================================


def build_table():
    """
    Generate the supercritical-water property lookup table and write it to the data
    directory.

    Why this model is here:
        pinthac/sca/lut.py interpolates this table instead of calling the IAPWS-95 solver
        at every axial node, which is what makes the table-driven channel solver fast
        enough to generate surrogate training data. A handful of pressure points bracket
        the nominal operating pressure so the table still resolves properties as the
        pressure drops along the channel.

        This used to run at module scope, which meant merely *importing* the module
        regenerated 4500 rows and overwrote the existing table -- so it now sits behind a
        function and a __main__ guard. Note that the regeneration is reproducible in
        physics but not byte-for-byte: GPU reduction ordering moves the last one or two
        significant figures, so nothing downstream should assert on an exact table value.

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
