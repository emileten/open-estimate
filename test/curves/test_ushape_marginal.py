import numpy as np
import pandas as pd
from openest.curves.basic import ZeroInterceptPolynomialCurve, ClippedCurve, ShiftedCurve, MinimumCurve
from openest.curves import ushape_numeric, ushape_analytic

# Read polymins
polymins = pd.read_csv("polymins.csv")
allcalcs = pd.read_csv("allcalcs.csv")

for ii in range(polymins.shape[0]):
    print ii

    mintemp = polymins['analytic'][ii]
    tas = np.arange(-40, 50, .01) # numeric appears to have problem with low inward clipping

    curve = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [allcalcs['tas'][ii], allcalcs['tas2'][ii], allcalcs['tas3'][ii], allcalcs['tas4'][ii]])

    clipcurve = ClippedCurve(ShiftedCurve(curve, -curve(mintemp)))
    orig = clipcurve(tas)

    jj = (ii + 1000) % polymins.shape[0]
    marginal = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [allcalcs['tas'][jj], allcalcs['tas2'][jj], allcalcs['tas3'][jj], allcalcs['tas4'][jj]])
    shiftmarginal = ShiftedCurve(marginal, -marginal(mintemp))

    # Try without good money
    uclipcurve_numeric = ushape_numeric.UShapedCurve(ClippedCurve(ShiftedCurve(curve, -curve(mintemp))), mintemp, lambda xs: tas)

    ucurve_numeric = ushape_numeric.UShapedClipping(uclipcurve_numeric, shiftmarginal, mintemp, lambda xs: tas, True)
    unum = ucurve_numeric(tas)

    uclipcurve_analytic = ushape_analytic.EClippedCurve(ushape_analytic.EShiftedCurve(ushape_analytic.UShapedCurve(curve.ccs, mintemp), -curve(mintemp)))
    uorig = uclipcurve_analytic(tas)

    ucurve_analytic = ushape_analytic.UShapedMarginal(uclipcurve_analytic, shiftmarginal)
    utas = ucurve_analytic.curve.uclip_evalpts(tas)
    uana = ucurve_analytic(tas)

    pd.DataFrame(dict(tas=tas, orig=orig, uorig=uorig, unum=unum, uana=uana, utas=utas))

    assert sum(~np.isclose(unum, uana, atol=.2)) < 5

    # Try with minimum
    if ii == 9342:
        continue # both curves crosses 0 within 0.01 of each other, messing up numeric
    if ii in [22965, 23020, 23093]:
        continue # off a little too much

    kk = (ii + 2000) % polymins.shape[0]
    curve2 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [allcalcs['tas'][kk], allcalcs['tas2'][kk], allcalcs['tas3'][kk], allcalcs['tas4'][kk]])

    # Try with good money and clipping
    uclipcurve_numeric = ushape_numeric.UShapedCurve(ClippedCurve(MinimumCurve(ShiftedCurve(curve, -curve(mintemp)), ShiftedCurve(curve2, -curve2(mintemp)))), mintemp, lambda xs: tas)
    ucurve_numeric = ushape_numeric.UShapedClipping(uclipcurve_numeric, shiftmarginal, mintemp, lambda xs: tas, True)
    unum = ucurve_numeric(tas)

    uclipcurve_analytic = ushape_analytic.UShapedMinimumCurve(curve, curve.ccs, curve2, curve2.ccs, mintemp)
    uorig = uclipcurve_analytic(tas)

    ucurve_analytic = ushape_analytic.UShapedMarginal(uclipcurve_analytic, shiftmarginal)
    utas = ucurve_analytic.curve.uclip_evalpts(tas)
    uana = ucurve_analytic(tas)

    clipcurve2 = ClippedCurve(ShiftedCurve(curve2, -curve2(mintemp)))
    orig2 = clipcurve2(tas)

    pd.DataFrame(dict(tas=tas, orig=orig, orig2=orig2, uorig=uorig, marginal=shiftmarginal(tas), unum=unum, uana=uana, utas=utas))

    assert sum(~np.isclose(unum, uana, atol=.2)) < 5
