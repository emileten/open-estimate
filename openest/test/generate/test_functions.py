import numpy as np
from generate.daily import YearlyDayBins, Constant
from generate.functions import Scale, Instabase
from test_daily import test_model

def test_Scale():
    application = Scale(Constant(33, 'widgets'), {'test': 1. / 11}, 'widgets', 'subwigs').test()
    for yearresult in application.push(1000 + np.arange(365), np.random.rand(365)):
        np.testing.assert_equal(yearresult[1], 3)

    def check_units():
        application = Scale(Constant(33, 'widgets'), {'test': 1.8}, 'deg. C', 'deg F.').test()
        for yearresult in application.push(1000 + np.arange(365), np.random.rand(365)):
            return

    np.testing.assert_raises(Exception, check_units)

def test_Instabase():
    application = Instabase(YearlyDayBins(test_model, 'days'), 2).test()
    for yearresult in application.push(1000 + np.arange(365), [-10] * 300 + [10] * 65):
        np.testing.assert_equal(True, False) # Should get nothing here
    for yearresult in application.push(2000 + np.arange(365), [-10] * 65 + [10] * 300):
        if year[0] == 1:
            np.testing.assert_equal(yearresult[1], 300. / 65.)
        if year[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
