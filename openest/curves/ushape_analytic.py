import numpy as np
from openest.curves.basic import MaximumCurve, MinimumCurve, ClippedCurve, ShiftedCurve, StepCurve, ZeroInterceptPolynomialCurve, UnivariateCurve

##### Evaluation point aware curves (for marginal effects)


class EMaximumCurve(MaximumCurve):
    def uclip_evalpts(self, xs):
        ys1 = self.curve1(xs)
        ys2 = self.curve2(xs)
        if isinstance(self.curve1, ZeroInterceptPolynomialCurve) or (isinstance(self.curve1, ShiftedCurve) and isinstance(self.curve1.curve, ZeroInterceptPolynomialCurve)):
            xs1 = xs
        else:
            xs1 = self.curve1.uclip_evalpts(xs)
        if isinstance(self.curve2, ZeroInterceptPolynomialCurve) or (isinstance(self.curve2, ShiftedCurve) and isinstance(self.curve2.curve, ZeroInterceptPolynomialCurve)):
            xs2 = xs
        else:
            xs2 = self.curve2.uclip_evalpts(xs)

        evalpts = np.nan * xs
        evalpts[ys1 >= ys2] = xs1[ys1 >= ys2]
        evalpts[ys1 < ys2] = xs2[ys1 < ys2]
        
        return evalpts

    
class EMinimumCurve(MinimumCurve):
    def uclip_evalpts(self, xs):
        ys1 = self.curve1(xs)
        ys2 = self.curve2(xs)
        if isinstance(self.curve1, ZeroInterceptPolynomialCurve) or (isinstance(self.curve1, ShiftedCurve) and isinstance(self.curve1.curve, ZeroInterceptPolynomialCurve)):
            xs1 = xs
        else:
            xs1 = self.curve1.uclip_evalpts(xs)
        if isinstance(self.curve2, ZeroInterceptPolynomialCurve) or (isinstance(self.curve2, ShiftedCurve) and isinstance(self.curve2.curve, ZeroInterceptPolynomialCurve)):
            xs2 = xs
        else:
            xs2 = self.curve2.uclip_evalpts(xs)

        evalpts = np.nan * xs
        evalpts[ys1 < ys2] = xs1[ys1 < ys2]
        evalpts[ys1 >= ys2] = xs2[ys1 >= ys2]

        return evalpts

    
class EClippedCurve(ClippedCurve):
    def __init__(self, curve, cliplow=True):
        super(EClippedCurve, self).__init__(curve, cliplow=cliplow)
        assert hasattr(curve, 'uclip_evalpts')
    
    def uclip_evalpts(self, xs):
        ys = self.curve(xs)
        evalpts = self.curve.uclip_evalpts(xs)
        if self.cliplow:
            evalpts[ys <= 0] = np.nan
        else:
            evalpts[ys >= 0] = np.nan

        return evalpts

    
class EStepCurve(StepCurve):
    def __init__(self, xxlimits, yy, evalpts, xtrans=None):
        super(EStepCurve, self).__init__(xxlimits, yy, xtrans=xtrans)
        self.evalpts = evalpts

    def uclip_evalpts(self, xs):
        bins = np.digitize(xs[np.isfinite(xs)], self.xxlimits)
        evalpts = np.nan * xs
        evalpts[np.isfinite(xs)] = np.array(self.evalpts)[bins - 1]
        return evalpts

    
class EShiftedCurve(ShiftedCurve):
    def uclip_evalpts(self, xs):
        if isinstance(self.curve, ZeroInterceptPolynomialCurve):
            return np.copy(xs)
        
        return self.curve.uclip_evalpts(xs)
    

##### Algorithm


def polyderiv(ccs):
    # Construct the derivative
    # Ordered as x, x^2, x^3, x^4; returns as c, x, x^2, x^3
    return np.array(ccs) * np.arange(1, len(ccs) + 1)


def mergeplateaus(alllimits, spans, levels, evalpts):
    xxlimits = np.sort(list(alllimits))
    yy = []
    xx = [] # points corresponding to yy
    for ii in range(len(xxlimits) - 1):
        level = -np.inf
        evalpt = np.nan
        for jj in range(len(spans)):
            if spans[jj][0] <= xxlimits[ii] and spans[jj][1] >= xxlimits[ii+1]:
                level = max(level, levels[jj])
                if level == levels[jj]:
                    evalpt = evalpts[jj]
        yy.append(level)
        xx.append(evalpt)

    return xxlimits, yy, xx


def derivative_clipping(ccs, mintemp):
    # Ordered as x, x^2, x^3, x^4
    derivcoeffs = polyderiv(ccs)
    roots = list(filter(np.isreal, np.roots(derivcoeffs[::-1]))) # only consider real roots
    
    deriv2coeffs = derivcoeffs[1:] * np.arange(1, len(derivcoeffs)) # Second derivative
    direction = np.polyval(deriv2coeffs[::-1], roots)
    rootvals = np.polyval(([0] + ccs)[::-1], roots)

    ## Look at each downturn
    alllimits = set([-np.inf, np.inf])
    levels = [] # min level for each span
    spans = [] # list of tuples with spans
    evalpts = [] # used for marginal calcs
    for ii, root in enumerate(roots):
        if direction[ii] < 0: # ignore if turning back up
            levels.append(rootvals[ii])
            evalpts.append(root)
            if root < mintemp:
                spans.append((-np.inf, root))
                alllimits.update([-np.inf, root])
            else:
                spans.append((root, np.inf))
                alllimits.update([root, np.inf])

    # Look at mintemp
    datmin = np.polyval(derivcoeffs[::-1], mintemp)
    atmin = np.polyval(([0] + ccs)[::-1], mintemp)
    if datmin > 0:
        evalpts.append(mintemp)
        levels.append(atmin)
        spans.append((-np.inf, mintemp))
        alllimits.update([-np.inf, mintemp])
    elif datmin < 0:
        evalpts.append(mintemp)
        levels.append(atmin)
        spans.append((mintemp, np.inf))
        alllimits.update([mintemp, np.inf])

    return mergeplateaus(alllimits, spans, levels, evalpts)


class UShapedCurve(EMaximumCurve):
    def __init__(self, ccs, mintemp):
        xxlimits, levels, evalpts = derivative_clipping(ccs, mintemp)
        super(UShapedCurve, self).__init__(ZeroInterceptPolynomialCurve([-np.inf, np.inf], ccs),
                                           EStepCurve(xxlimits, levels, evalpts))


class UShapedMinimumCurve(EMaximumCurve):
    def __init__(self, curve1, ccs1, curve2, ccs2, mintemp):
        baselevel1 = curve1(mintemp)
        derivcoeffs1 = polyderiv(ccs1)
        roots1 = np.real(filter(np.isreal, np.roots(derivcoeffs1[::-1])))

        baselevel2 = curve2(mintemp)
        derivcoeffs2 = polyderiv(ccs2)
        roots2 = np.real(filter(np.isreal, np.roots(derivcoeffs2[::-1])))

        crosses = np.real(filter(np.isreal, np.roots(np.concatenate((np.array(ccs1[::-1]) - np.array(ccs2[::-1]), [-baselevel1 + baselevel2])))))

        alllimits = set(crosses)
        alllimits.update(roots1)
        alllimits.update(roots2)
        alllimits.update([-np.inf, mintemp, np.inf])
        xxlimits = np.sort(list(alllimits))

        clipcurve1 = EClippedCurve(EShiftedCurve(curve1, -baselevel1))
        clipcurve2 = EClippedCurve(EShiftedCurve(curve2, -baselevel2))

        # Force plateaus to be u-shaped
        lolevels = [] # in opposite order
        loevalpts = [] # used for marginal calcs
        minlevel = 0
        evalpt = mintemp
        for ii in range(np.where(xxlimits == mintemp)[0][0])[::-1]:
            # What is shown between xxlimits[ii] and xxlimits[ii+1]?  Note: ended at minlevel = xxlimits[ii+1]
            if ii == 0:
                curve1x = clipcurve1(xxlimits[ii+1] - 1)
                curve2x = clipcurve2(xxlimits[ii+1] - 1)
            else:
                curve1x = clipcurve1(xxlimits[ii])
                curve2x = clipcurve2(xxlimits[ii])
            lolevels.append(minlevel)
            loevalpts.append(evalpt)
            if curve1x < curve2x:
                if curve1x > minlevel:
                    minlevel = curve1x
                    evalpt = xxlimits[ii]
            elif curve1x > curve2x:
                if curve2x > minlevel:
                    minlevel = curve2x
                    evalpt = xxlimits[ii]
            elif curve1x > minlevel: # and equal
                minlevel = curve1x
                evalpt = xxlimits[ii]

        hilevels = []
        hievalpts = []
        minlevel = 0
        evalpt = mintemp
        for ii in range(np.where(xxlimits == mintemp)[0][0], len(xxlimits) - 1):
            # What is shown between xxlimits[ii] and xxlimits[ii+1]?  Note: ended at minlevel = xxlimits[ii]
            if ii+1 == len(xxlimits) - 1:
                curve1x = clipcurve1(xxlimits[ii] + 1)
                curve2x = clipcurve2(xxlimits[ii] + 1)
            else:
                curve1x = clipcurve1(xxlimits[ii+1])
                curve2x = clipcurve2(xxlimits[ii+1])
            hilevels.append(minlevel)
            hievalpts.append(evalpt)
            if curve1x < curve2x:
                if curve1x > minlevel:
                    minlevel = curve1x
                    evalpt = xxlimits[ii+1]
            elif curve1x > curve2x:
                if curve2x > minlevel:
                    minlevel = curve2x
                    evalpt = xxlimits[ii+1]
            elif curve1x > minlevel: # and equal
                minlevel = curve1x    
                evalpt = xxlimits[ii+1]

        super(UShapedMinimumCurve, self).__init__(EMinimumCurve(clipcurve1, clipcurve2),
                                                  EStepCurve(xxlimits, lolevels[::-1] + hilevels, loevalpts[::-1] + hievalpts))


class UShapedMarginal(UnivariateCurve):
    def __init__(self, curve, marginal):
        super(UShapedMarginal, self).__init__([-np.inf, np.inf])
        self.curve = curve
        self.marginal = marginal

    def __call__(self, xs):
        xs2 = self.curve.uclip_evalpts(xs)
        ys = np.zeros(len(xs))
        ys[np.isfinite(xs2)] = self.marginal(xs2[np.isfinite(xs2)])
        return ys

