import numpy as np
import xarray as xr
from openest.models.bin_model import BinModel
from openest.models.spline_model import SplineModel
from openest.generate.daily import YearlyDayBins
from openest.generate.base import Constant

test_model = BinModel([-40, 0, 80], SplineModel.create_gaussian({0: (0, 1), 1: (1, 1)}, order=range(2)))

def test_YearlyDayBins():
    application = YearlyDayBins(test_model, 'days').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 300 + [10] * 65)},
                                                      coords={'time': map(lambda nn: datetime.date(1800, 1, 1) + datetime.timedelta(nn), np.arange(365))})):
        np.testing.assert_equal(result, 65.)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], [-10] * 65 + [10] * 300)},
                                                      coords={'time': map(lambda nn: datetime.date(2000, 1, 1) + datetime.timedelta(nn), np.arange(365))})):
        np.testing.assert_equal(result, 300.)

def test_Constant():
    application = Constant(33, 'widgets').test()
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': map(lambda nn: datetime.date(1800, 1, 1) + datetime.timedelta(nn), np.arange(365))})):
        np.testing.assert_equal(result, 33)
    for (year, result) in application.push(xr.Dataset({'x': (['time'], np.random.rand(365))},
                                                      coords={'time': map(lambda nn: datetime.date(2000, 1, 1) + datetime.timedelta(nn), np.arange(365))})):
        np.testing.assert_equal(result, 33)
