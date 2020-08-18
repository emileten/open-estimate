import time
import numpy as np
from openest.curves import basic, ushape_analytic, ushape_numeric

curve = basic.ZeroInterceptPolynomialCurve([-10, 30], [1, 1, -.03]) # u-shaped to about 22

fillxxs = np.arange(-10, 30)
fillyys = curve(fillxxs)

ucurve_old = ushape_numeric.UShapedCurve(curve, 0, lambda x: x, ordered=False, fillxxs=fillxxs, fillyys=fillyys)
ucurve_new = ushape_numeric.UShapedCurve(curve, 0, lambda x: x, ordered='maintain', fillxxs=fillxxs, fillyys=fillyys)

#tas = 40 * np.random.sample(100) - 10 # -10 to 30
#tas = np.array([1, 30, 2, 35, 3, 40])
tas = 40 * np.random.sample(100) - 10 # -10 to 30

results_old = ucurve_old(tas)
results_new = ucurve_new(tas)

## Check that same results from old version
np.testing.assert_equal(sorted(results_old), sorted(results_new))

## Check that all values match up to plateau
hiplateau = None
loplateau = None
for ii in range(len(tas)):
    if results_new[ii] != curve(tas[ii]):
        if tas[ii] > 0:
            if hiplateau is None:
                hiplateau = results_new[ii]
            else:
                np.testing.assert_equal(results_new[ii], hiplateau)
        else:
            if loplateau is None:
                loplateau = results_new[ii]
            else:
                np.testing.assert_equal(results_new[ii], loplateau)

ucurveclip_old = ushape_numeric.UShapedClipping(ucurve_old, curve, 0, lambda x: x, ordered=False)
ucurveclip_new = ushape_numeric.UShapedClipping(ucurve_new, curve, 0, lambda x: x, ordered='maintain')

results_old = ucurveclip_old(tas)
results_new = ucurveclip_new(tas)

## Check that same results from old version
np.testing.assert_equal(sorted(results_old), sorted(results_new))

## Check that all values match up to plateau
hiplateau = None
for ii in range(len(tas)):
    if curve(tas[ii]) < 0:
        np.testing.assert_equal(results_new[ii], 0)
    elif results_new[ii] != curve(tas[ii]):
        if hiplateau is None:
            hiplateau = results_new[ii]
        else:
            np.testing.assert_equal(results_new[ii], hiplateau)

