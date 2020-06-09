import numpy as np
from .univariate_model import UnivariateModel
from scipy.interpolate import UnivariateSpline
from statsmodels.distributions.empirical_distribution import StepFunction

class UnivariateCurve(UnivariateModel):
    def __init__(self, xx):
        super(UnivariateCurve, self).__init__(xx_is_categorical=False, xx=xx, scaled=True)

    def __call__(self, x):
        raise NotImplementedError("call not implemented")

    def get_xx(self):
        return self.xx

    def eval_pval(self, x, p, threshold=1e-3):
        return self(x)

    def eval_pvals(self, x, p, threshold=1e-3):
        return self(x)

class CurveCurve(UnivariateCurve):
    def __init__(self, xx, curve):
        super(CurveCurve, self).__init__(xx)
        self.curve = curve

    def __call__(self, x):
        return self.curve(x)

    @staticmethod
    def make_linear_spline_curve(xx, yy, limits):
        xx = np.concatenate(([limits[0]], xx, [limits[1]]))
        yy = np.concatenate(([yy[0]], yy, [yy[-1]]))

        return UnivariateSpline(xx, yy, s=0, k=1)

class FlatCurve(CurveCurve):
    def __init__(self, yy):
        super(FlatCurve, self).__init__([-np.inf, np.inf], lambda x: yy)

class LinearCurve(CurveCurve):
    def __init__(self, yy):
        super(LinearCurve, self).__init__([-np.inf, np.inf], lambda x: yy * x)

class StepCurve(CurveCurve):
    def __init__(self, xxlimits, yy, xtrans=None):
        step_function = StepFunction(xxlimits[1:-1], yy[1:], ival=yy[0])
        if xtrans is None:
            super(StepCurve, self).__init__((np.array(xxlimits[0:-1]) + np.array(xxlimits[1:])) / 2, lambda x: step_function(x))
        else:
            super(StepCurve, self).__init__((np.array(xxlimits[0:-1]) + np.array(xxlimits[1:])) / 2, lambda x: step_function(xtrans(x)))

        self.xxlimits = xxlimits
        self.yy = yy

class ZeroInterceptPolynomialCurve(UnivariateCurve):
    def __init__(self, xx, ccs):
        super(ZeroInterceptPolynomialCurve, self).__init__(xx)
        self.ccs = ccs
        self.pvcoeffs = list(ccs[::-1]) + [0] # Add on constant and start with highest order

    def __call__(self, x):
        return np.polyval(self.pvcoeffs, x)

def pos(x):
    return x * (x > 0)

class CubicSplineCurve(UnivariateCurve):
    def __init__(self, knots, coeffs):
        super(CubicSplineCurve, self).__init__(knots)
        self.knots = knots
        self.coeffs = coeffs

    def get_terms(self, x):
        """Get the set of knots-1 terms representing temperature x."""
        terms = [x]
        for kk in range(len(self.knots) - 2):
            termx_k = pos(x - self.knots[kk])**3 - pos(x - self.knots[-2])**3 * (self.knots[-1] - self.knots[kk]) / (self.knots[-1] - self.knots[-2]) + pos(x - self.knots[-1])**3 * (self.knots[-2] - self.knots[kk]) / (self.knots[-1] - self.knots[-2])
            terms.append(termx_k)

        return terms

    def __call__(self, x):
        """Get the set of knots-1 terms representing temperature x and multiply by the coefficients."""
        x = np.array(x) # make sure we have an array_like
        total = x * self.coeffs[0]
        for kk in range(len(self.knots) - 2):
            termx_k = pos(x - self.knots[kk])**3 - pos(x - self.knots[-2])**3 * (self.knots[-1] - self.knots[kk]) / (self.knots[-1] - self.knots[-2]) + pos(x - self.knots[-1])**3 * (self.knots[-2] - self.knots[kk]) / (self.knots[-1] - self.knots[-2])
            total += termx_k * self.coeffs[kk + 1]

        return total

class CoefficientsCurve(UnivariateCurve):
    """A curve represented by the sum of multiple predictors, each multiplied by a coefficient."""
    def __init__(self, coeffs, curve, xtrans=None):
        super(CoefficientsCurve, self).__init__([-np.inf, np.inf])
        self.coeffs = coeffs
        self.curve = curve
        self.xtrans = xtrans

    def __call__(self, x):
        if np.isscalar(x):
            return self.curve(x)
        elif self.xtrans is not None:
            return self.xtrans(x).dot(self.coeffs)
        else:
            x = np.array(x)
            return x.dot(self.coeffs)
        
class ShiftedCurve(UnivariateCurve):
    def __init__(self, curve, offset):
        super(ShiftedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.offset = offset

    def __call__(self, xs):
        return self.curve(xs) + self.offset

class ProductCurve(UnivariateCurve):
    def __init__(self, curve1, curve2):
        super(ProductCurve, self).__init__(curve1.xx)
        self.curve1 = curve1
        self.curve2 = curve2

    def __call__(self, xs):
        return self.curve1(xs) * self.curve2(xs)
    
class ClippedCurve(UnivariateCurve):
    def __init__(self, curve, cliplow=True):
        super(ClippedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.cliplow = cliplow

    def __call__(self, xs):
        ys = self.curve(xs)
        if self.cliplow:
            return ys * (ys > 0)
        else:
            return ys * (ys < 0)            

class OtherClippedCurve(ClippedCurve):
    def __init__(self, clipping_curve, value_curve, clipy=0):
        super(OtherClippedCurve, self).__init__(value_curve)
        self.clipping_curve = clipping_curve
        self.clipy = clipy

    def __call__(self, xs):
        ys = self.curve(xs)
        clipping = self.clipping_curve(xs)
        ys = [y if y is not None else 0 for y in ys]
        clipping = [y if not np.isnan(y) else 0 for y in clipping]
        return ys * (clipping > self.clipy)

class MinimumCurve(UnivariateCurve):
    def __init__(self, curve1, curve2):
        super(MinimumCurve, self).__init__(curve1.xx)
        self.curve1 = curve1
        self.curve2 = curve2

    def __call__(self, xs):
        return np.minimum(self.curve1(xs), self.curve2(xs))

class MaximumCurve(UnivariateCurve):
    def __init__(self, curve1, curve2):
        super(MaximumCurve, self).__init__(curve1.xx)
        self.curve1 = curve1
        self.curve2 = curve2

    def __call__(self, xs):
        return np.maximum(self.curve1(xs), self.curve2(xs))

class SelectiveInputCurve(UnivariateCurve):
    """Assumes input is a matrix, and only pass selected input columns to child curve."""
    
    def __init__(self, curve, indices):
        super(SelectiveInputCurve, self).__init__(curve.xx)
        self.curve = curve
        self.indices = np.array(indices)

    def __call__(self, xs):
        return self.curve(xs[:, self.indices])

class PiecewiseCurve(UnivariateCurve):
    def __init__(self, curves, knots, xtrans=lambda x: x):
        super(PiecewiseCurve, self).__init__(knots)
        self.curves = curves
        self.knots = knots
        self.xtrans = xtrans # for example, to select first column

    def __call__(self, xs):
        if np.isscalar(xs):
            for ii in range(len(self.knots) - 1):
                if xs >= self.knots[ii] and xs < self.knots[ii+1]:
                    return self.curves[ii](xs)
            return np.nan

        ys = np.ones(len(xs)) * np.nan
        
        for ii in range(len(self.knots) - 1):
            txs = self.xtrans(xs)
            within = (txs >= self.knots[ii]) & (txs < self.knots[ii+1])
            wixs = xs[within]
            if len(wixs) > 0:
                ys[within] = self.curves[ii](wixs)

        return ys
