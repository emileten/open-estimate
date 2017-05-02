import numpy as np
from univariate_model import UnivariateModel
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
    def __init__(self, xxlimits, yy):
        step_function = StepFunction(xxlimits[1:-1], yy[1:], ival=yy[0])
        super(StepCurve, self).__init__((np.array(xxlimits[0:-1]) + np.array(xxlimits[1:])) / 2, step_function)

        self.xxlimits = xxlimits
        self.yy = yy

class PolynomialCurve(UnivariateCurve):
    def __init__(self, xx, ccs):
        super(PolynomialCurve, self).__init__(xx)
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
        total = x * self.coeffs[0]
        for kk in range(self.knots - 2):
            termx_k = pos(x - self.knots[kk])**3 - pos(x - self.knots[-2])**3 * (self.knots[-1] - self.knots[kk]) / (self.knots[-1] - self.knots[-2]) + pos(x - self.knots[-1])**3 * (self.knots[-2] - self.knots[kk]) / (self.knots[-1] - self.knots[-2])
            total += termx_k * self.coeffs[kk + 1]

        return total

class AdaptableCurve(UnivariateCurve):
    def __init__(self, xx):
        super(AdaptableCurve, self).__init__(xx)

    # Subclasses can define their own interfaces
    def update(self):
        pass

