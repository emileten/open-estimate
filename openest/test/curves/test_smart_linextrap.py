import pytest
import numpy as np
import pandas as pd
import xarray as xr

from openest.generate.smart_curve import ZeroInterceptPolynomialCurve
from openest.curves.smart_linextrap import LinearExtrapolationCurve

## 1-D orthotope
def test_1d_orthotope():
    basecurve1 = ZeroInterceptPolynomialCurve([1, 1], ['tas', 'tas-poly-2'])

    bounds1 = {'tas': (0, 1)}
    margins1 = {'tas': .1}
    clipcurve = LinearExtrapolationCurve(basecurve1, ['tas'], bounds1, margins1, 1)

    ds0 = xr.Dataset({'tas': (['time'], [0, .5, 1]),
                      'tas-poly-2': (['time'], np.array([0, .5, 1]) ** 2)},
                     coords={'time': pd.date_range('1800-01-01', periods=3)})
    yy0 = basecurve1(ds0)

    ds1 = xr.Dataset({'tas': (['time'], [-.2, -.1, .5, 1.2, 1.3]),
                      'tas-poly-2': (['time'], np.array([-.2, -.1, .5, 1.2, 1.3]) ** 2)},
                     coords={'time': pd.date_range('1800-01-01', periods=5)})

    slope0 = -1.1
    slope1 = 2.9

    yy1 = clipcurve(ds1)
    desired = [yy0[0] + .2 * slope0, yy0[0] + .1 * slope0, yy0[1], yy0[2] + .2 * slope1, yy0[2] + .3 * slope1]
    np.testing.assert_allclose(yy1, desired)

    clipcurve = LinearExtrapolationCurve(basecurve1, ['tas'], bounds1, margins1, .1)
    yy1 = clipcurve(ds1)
    desired = [yy0[0] + .2 * slope0 / 10, yy0[0] + .1 * slope0 / 10, yy0[1], yy0[2] + .2 * slope1 / 10, yy0[2] + .3 * slope1 / 10]
    np.testing.assert_allclose(yy1, desired)
