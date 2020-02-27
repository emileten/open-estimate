import numpy as np
import pandas as pd
import xarray as xr

from openest.generate.base import Constant
from openest.generate.daily import YearlyDayBins
from openest.generate.functions import Scale, Instabase, SpanInstabase
from .test_daily import test_curve


def make_year_ds(year, values):
    year = str(year)
    time_coord = pd.date_range(year + '-01-01', periods=365)
    return xr.Dataset({'x': (['time'], values)},
                      coords={'time': time_coord})

def test_scale():
    application = Scale(Constant(33, 'widgets'), {'test': 1. / 11}, 'widgets', 'subwigs').test()
    for yearresult in application.push(make_year_ds(2000, np.random.rand(365))):
        np.testing.assert_equal(yearresult[1], 3)

    def check_units():
        app = Scale(Constant(33, 'widgets'), {'test': 1.8}, 'deg. C', 'deg F.').test()
        for _ in app.push(make_year_ds(2000, np.random.rand(365))):
            return

    np.testing.assert_raises(Exception, check_units)


def test_instabase():
    calc = Instabase(YearlyDayBins(test_curve, 'days', lambda ds: ds['x']), 2)
    app1 = calc.test()
    app2 = calc.test()
    for _ in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        raise AssertionError  # Should get nothing here
    for _ in app2.push(make_year_ds(1800, [10] * 365)):
        raise AssertionError  # Should get nothing here
    for yearresult in app1.push(make_year_ds(2000, [-10] * 65 + [10] * 300)):
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 65. / 300.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
    for yearresult in app2.push(make_year_ds(2000, [10] * 365)):
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 1.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)


def test_spaninstabase():
    calc = SpanInstabase(YearlyDayBins(test_curve, 'days', lambda ds: ds['x']), 2, 2)
    app1 = calc.test()
    app2 = calc.test()
    for _ in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        raise AssertionError  # Should get nothing here
    for _ in app2.push(make_year_ds(1800, [10] * 365)):
        raise AssertionError  # Should get nothing here
    for yearresult in app1.push(make_year_ds(2000, [-10] * 65 + [10] * 300)):
        print(app1.denomterms)
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 65. / 300.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
    for yearresult in app2.push(make_year_ds(2000, [10] * 365)):
        print(app2.denomterms)
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 1.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
