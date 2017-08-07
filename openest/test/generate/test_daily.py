import numpy as np
import xarray as xr
from openest.models.bin_model import BinModel
from openest.models.spline_model import SplineModel
from openest.generate.daily import YearlyDayBins, Constant

test_model = BinModel([-40, 0, 80], SplineModel.create_gaussian({0: (0, 1), 1: (1, 1)}, order=range(2)))

def test_YearlyDayBins():
    application = YearlyDayBins(test_model, 'days').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 300 + [10] * 65)},
                                                      coords={'time': 1800 + np.arange(365)})):
        np.testing.assert_equal(result, 65.)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 65 + [10] * 300)},
                                                      coords={'time': 2000 + np.arange(365)})):
        np.testing.assert_equal(result, 300.)

def test_Constant():
    application = Constant(33, 'widgets').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': 1800 + np.arange(365)})):
        np.testing.assert_equal(result, 33)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': 2000 + np.arange(365)})):
        np.testing.assert_equal(result, 33)
