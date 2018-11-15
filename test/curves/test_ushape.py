import numpy as np
import pandas as pd
from openest.curves.basic import ZeroInterceptPolynomialCurve, ClippedCurve, ShiftedCurve, MinimumCurve
from openest.curves import ushape_numeric, ushape_analytic

# Read polymins
polymins = pd.read_csv("polymins.csv")
allcalcs = pd.read_csv("allcalcs.csv")

tas = np.arange(-40, 50, .01)
fillins = np.arange(-40, 50, .1)
    
for ii in range(polymins.shape[0]):
    print ii
    
    curve = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [allcalcs['tas'][ii], allcalcs['tas2'][ii], allcalcs['tas3'][ii], allcalcs['tas4'][ii]])
    mintemp = polymins['analytic'][ii]
    orig = curve(tas)

    # Try without clipping
    ucurve_numeric = ushape_numeric.UShapedCurve(curve, mintemp, lambda xs: tas, True, fillins=fillins)
    unum = ucurve_numeric(tas)

    ucurve_analytic = ushape_analytic.UShapedCurve(curve.ccs, mintemp)
    uana = ucurve_analytic(tas)

    np.testing.assert_allclose(unum, uana, atol=.1)
    
    # Try with clipping
    ucurve_numeric = ushape_numeric.UShapedCurve(ClippedCurve(ShiftedCurve(curve, -curve(mintemp))), mintemp, lambda xs: tas, True, fillins=fillins)
    unum = ucurve_numeric(tas)

    ucurve_analytic = ClippedCurve(ShiftedCurve(ushape_analytic.UShapedCurve(curve.ccs, mintemp), -curve(mintemp)))
    uana = ucurve_analytic(tas)

    np.testing.assert_allclose(unum, uana, atol=.1)
    
    # Try with minimum
    jj = (ii + 1000) % polymins.shape[0]
    curve2 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [allcalcs['tas'][jj], allcalcs['tas2'][jj], allcalcs['tas3'][jj], allcalcs['tas4'][jj]])

    # Try with good money and clipping
    ucurve_numeric = ushape_numeric.UShapedCurve(ClippedCurve(MinimumCurve(ShiftedCurve(curve, -curve(mintemp)), ShiftedCurve(curve2, -curve2(mintemp)))), mintemp, lambda xs: tas, True, fillins=fillins)
    unum = ucurve_numeric(tas)
    
    ucurve_analytic = ushape_analytic.UShapedMinimumCurve(curve, curve.ccs, curve2, curve2.ccs, mintemp)
    uana = ucurve_analytic(tas)
    
    pd.DataFrame(dict(tas=tas, orig=orig - curve(mintemp), good=curve2(tas) - curve2(mintemp), unum=unum, uana=uana))

    np.testing.assert_allclose(unum, uana, atol=.1)
    
    max(abs(uana - unum))
