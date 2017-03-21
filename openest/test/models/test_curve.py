import numpy as np
from openest.models.curve import CubicSplineCurve

def test_CubicSplineCurve():
    knots = [-12, -7, 0, 10, 18, 23, 28, 33]
    curve = CubicSplineCurve(knots, np.zeros(len(knots) - 1))
    print curve.get_terms(20)
    np.testing.assert_equal(curve.get_terms(20), [20, 32768, 19683, 8000, 1000, 8, 0])
