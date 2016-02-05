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
        super(FlatCurve, self).__init__([-40, 0, 80], lambda x: yy)

class StepCurve(CurveCurve):
    def __init__(self, xxlimits, yy):
        step_function = StepFunction(xxlimits[1:-1], yy[1:], ival=yy[0])
        super(StepCurve, self).__init__((np.array(xxlimits[0:-1]) + np.array(xxlimits[1:])) / 2, step_function)

        self.xxlimits = xxlimits
        self.yy = yy

class AdaptableCurve(UnivariateCurve):
    def __init__(self, xx):
        super(AdaptableCurve, self).__init__(xx)

    def setup(self, yyyyddd, temps, **kw):
        pass

    def update():
        pass

