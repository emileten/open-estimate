import numpy as np
from models.bin_model import BinModel
from models.spline_model import SplineModel
from generate.daily import YearlyDayBins, Constant

test_model = BinModel([-40, 0, 80], SplineModel.create_gaussian({0: (0, 1), 1: (1, 1)}, order=range(2)))

def test_YearlyDayBins():
    application = YearlyDayBins(test_model, 'days').test()
    for (year, result) in application.push(1000 + np.arange(365), [-10] * 300 + [10] * 65):
        np.testing.assert_equal(result, 65. / 365.)
    for (year, result) in application.push(2000 + np.arange(365), [-10] * 65 + [10] * 300):
        np.testing.assert_equal(result, 300. / 365.)

def test_Constant():
    application = Constant(33, 'widgets').test()
    for (year, result) in application.push(1000 + np.arange(365), np.random.rand(365)):
        np.testing.assert_equal(result, 33)
    for (year, result) in application.push(2000 + np.arange(365), np.random.rand(365)):
        np.testing.assert_equal(result, 33)
