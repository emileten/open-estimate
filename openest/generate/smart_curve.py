import numpy as np
from univariate_model import UnivariateModel
from scipy.interpolate import UnivariateSpline
from statsmodels.distributions.empirical_distribution import StepFunction

## Smart Curves fall back on Curve logic, but take xarray DataSets and know which variables they want

class SmartCurve(object):
    def __call__(self, ds):
        raise NotImplementedError("call not implemented")

class CurveCurve(SmartCurve):
    def __init__(self, curve, variable):
        self.curve = curve
        self.variable = variable

    def __call__(self, ds):
        return self.curve(ds[self.variable])

class ConstantCurve(SmartCurve):
    def __init__(self, cosntant, dimension):
        self.constant = constant
        self.dimension = dimension

    def __call__(self, ds):
        return np.repeat(self.constant, len(ds[self.dimension]))

class LinearCurve(CurveCurve):
    def __init__(self, slope, variable):
        super(LinearCurve, self).__init__(lambda x: slope * x, variable)

class StepCurve(CurveCurve):
    def __init__(self, xxlimits, levels, variable):
        step_function = StepFunction(xxlimits[1:-1], levels[1:], ival=levels[0])
        super(StepCurve, self).__init__(step_function, variable)

        self.xxlimits = xxlimits
        self.levels = levels

class CoefficientsCurve(SmartCurve):
    def __init__(self, coeffs, variables):
        self.coeffs = coeffs
        self.variables = variables

        assert isinstance(variables, list) and len(variable) == len(coeffs)

    def __call__(self, ds):
        result = np.zeros(ds[self.variables[0]].shape)
        for ii in range(len(self.variables)):
            result += self.coeffs[ii] * ds[self.variables[ii]].values

        return result
