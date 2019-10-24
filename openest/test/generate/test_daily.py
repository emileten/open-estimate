import numpy as np
import pandas as pd
import xarray as xr

from openest.generate.base import Constant
from openest.generate.daily import YearlyDayBins
from openest.models.bin_model import BinModel
from openest.models.spline_model import SplineModel

test_model = BinModel([-40, 0, 80], SplineModel.create_gaussian({0: (0, 1), 1: (1, 1)}, order=range(2)))


def test_yearlydaybins():
    application = YearlyDayBins(test_model, 'days').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 300 + [10] * 65)},
                                                      coords={'time': pd.date_range('1800-01-01', periods=365)})):
        np.testing.assert_equal(result, 65.)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 65 + [10] * 300)},
                                                      coords={'time': pd.date_range('2000-01-01', periods=365)})):
        np.testing.assert_equal(result, 300.)


def test_constant():
    application = Constant(33, 'widgets').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': pd.date_range('1800-01-01', periods=365)})):
        np.testing.assert_equal(result, 33)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': pd.date_range('2000-01-01', periods=365)})):
        np.testing.assert_equal(result, 33)
