import numpy as np
import pandas as pd
import xarray as xr
import pytest

from openest.generate.base import Constant
from openest.generate.daily import YearlyDayBins
from openest.generate.functions import Scale, Instabase, SpanInstabase, Clip
from .test_daily import test_curve


class MockApplication:
    """Mocks openest.generate.calculation.CustomFunctionalCalculation-like for ease

    This primarily allows us to prime the Application generators with simple Sequences.
    """
    def __init__(self, years, values, unitses):
        self.unitses = unitses
        # Mimic generator nature of the data, test if data leaks.
        self.data = zip(years, values)

    def apply(self, *args, **kwargs):
        return self

    def column_info(self):
        return [{'name': 'mockname', 'title': 'mocktitle', 'description': 'mockdescription'}]

    def push(self, *args, **kwargs):
        for x in self.data:
            yield x


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


class TestClip:
    """Basic tests for openest.generate.functions.Clip
    """
    def test_units_append(self):
        """Tests that Clip instances correctly append units from the original subcalc
        """
        unit = ['fakeunit']
        subcalc_mock = MockApplication(years=[0], values=[1.0], unitses=unit)
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        assert clipped_calc.unitses == unit * 2

    @pytest.mark.parametrize(
        "in_years,in_values,expected",
        [
            ([0], [-0.5], [0, 0.0, -0.5]),
            ([1], [0.0], [1, 0.0, 0.0]),
            ([2], [0.5], [2, 0.5, 0.5]),
        ],
        ids=['outside clip', 'clip edge', 'inside clip'],
    )
    def test_apply(self, in_years, in_values, expected):
        """Test Clip.apply() clips values in, out, and on edge of interval
        """
        subcalc_mock = MockApplication(
            years=in_years,
            values=in_values,
            unitses=['fakeunit'],
        )
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim_gen = clipped_calc.apply('foobar_region').push('not_a_ds')
        assert next(victim_gen) == expected

    def test_apply_memory(self):
        """Test that Clip.apply() doesn't hold memory between yields
        """
        subcalc_mock = MockApplication(
            years=[0, 1, 2],
            values=[0.0, 0.5, 1.0],
            unitses=['fakeunit'],
        )
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim_gen = clipped_calc.apply('foobar_region').push('not_a_ds')
        iter0 = next(victim_gen)
        iter1 = next(victim_gen)
        iter2 = next(victim_gen)
        assert (iter0 == [0, 0.0, 0.0]) and (iter1 == [1, 0.5, 0.5]) and (iter2 == [2, 1.0, 1.0])

    def test_column_info(self):
        """Ensure Clip.column_info() appends dict with correct keys
        """
        subcalc_mock = MockApplication(years=[0], values=[1.0], unitses=['fakeunit'])
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim = clipped_calc.column_info()
        # Check that first dict is from Clip instance
        assert victim[0]['name'] == 'clip'
        assert list(victim[0].keys()) == ['name', 'title', 'description']

    def test_describe(self):
        """Ensure Clip.describe() returns dict with correct keys
        """
        subcalc_mock = MockApplication(years=[0], values=[1.0], unitses=['fakeunit'])
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim = clipped_calc.describe()
        assert list(victim.keys()) == ['input_timerate', 'output_timerate', 'arguments', 'description']
