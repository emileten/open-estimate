import pytest
import numpy as np

from openest.models.curve import ZeroInterceptPolynomialCurve, CoefficientsCurve
from openest.curves.linextrap import *

## 1-D orthotope
def test_1d_orthotope():
    basecurve1 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [1, 1])

    bounds1 = {0: (0, 1)}
    clipcurve = LinearExtrapolationCurve(basecurve1, bounds1, [.1], 1, lambda x: x)
    yy0 = basecurve1([0, .5, 1])

    points1 = np.expand_dims(np.array([-.2, -.1, .5, 1.2, 1.3]), axis=-1)
    slope0 = (basecurve1(0) - basecurve1(.1)) / .1
    slope1 = (basecurve1(1) - basecurve1(.9)) / .1
    np.testing.assert_almost_equal(slope0, -1.1)
    np.testing.assert_almost_equal(slope1, 2.9)

    yy1 = clipcurve(points1)[:, 0]
    desired = [yy0[0] + .2 * slope0, yy0[0] + .1 * slope0, yy0[1], yy0[2] + .2 * slope1, yy0[2] + .3 * slope1]
    np.testing.assert_allclose(yy1, desired)

    clipcurve = LinearExtrapolationCurve(basecurve1, bounds1, [.1], .1, lambda x: x)
    yy1 = clipcurve(points1)[:, 0]
    desired = [yy0[0] + .2 * slope0 / 10, yy0[0] + .1 * slope0 / 10, yy0[1], yy0[2] + .2 * slope1 / 10, yy0[2] + .3 * slope1 / 10]
    np.testing.assert_allclose(yy1, desired)

## 1-D polytope
def test_1d_polytope():
    basecurve1 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [1, 1])

    bounds1 = [(0.,), (1.,)]
    clipcurve = LinearExtrapolationCurve(basecurve1, bounds1, [.1], 1, lambda x: x)
    yy0 = basecurve1([0, .5, 1])

    points1 = np.expand_dims(np.array([-.2, -.1, .5, 1.2, 1.3]), axis=-1)
    slope0 = (basecurve1(0) - basecurve1(.1)) / .1
    slope1 = (basecurve1(1) - basecurve1(.9)) / .1
    np.testing.assert_almost_equal(slope0, -1.1)
    np.testing.assert_almost_equal(slope1, 2.9)

    yy1 = clipcurve(points1)[:, 0]
    desired = [yy0[0] + .2 * slope0, yy0[0] + .1 * slope0, yy0[1], yy0[2] + .2 * slope1, yy0[2] + .3 * slope1]
    np.testing.assert_allclose(yy1, desired)

    clipcurve = LinearExtrapolationCurve(basecurve1, bounds1, [.1], .1, lambda x: x)
    yy1 = clipcurve(points1)[:, 0]
    desired = [yy0[0] + .2 * slope0 / 10, yy0[0] + .1 * slope0 / 10, yy0[1], yy0[2] + .2 * slope1 / 10, yy0[2] + .3 * slope1 / 10]
    np.testing.assert_allclose(yy1, desired)
    
## 2-D orthotope
def test_2d_orthotope():
    basecurve1 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [1, 1])
    basecurve2 = CoefficientsCurve([1, 1], basecurve1)

    bounds2 = {0: (0, 1), 1:(0, 1)}
    clipcurve = LinearExtrapolationCurve(basecurve2, bounds2, [.1, .1], 1, lambda x: x)
    yy0 = basecurve2(np.array([[0, .5], [0, 0], [.5, .25], [.5, 1], [1, 1]]))
                  
    points2 = np.array([[-.1, .5], [-.1, -.1], [.5, .25], [.5, 1.21], [1.1, 1.21]])
    slope0x = (basecurve2([0, .5]) - basecurve2([.1, .5])) / .1
    slope0y = (basecurve2([0, 0]) - basecurve2([0, .1])) / .1
    slope1y = (basecurve2([.5, 1]) - basecurve2([.5, .9])) / .1
    slope1x = (basecurve2([1, 1]) - basecurve2([.9, 1])) / .1
    np.testing.assert_allclose([slope0x, slope0y, slope1y, slope1x], [-1, -1, 1, 1])

    yy1 = clipcurve(points2)
    desired = [yy0[0] + .1 * slope0x, yy0[1] + .1 * slope0x + .1 * slope0y, yy0[2], yy0[3] + .21 * slope1y, yy0[4] + .1 * slope1x + .21 * slope1y]
    print(yy1)
    np.testing.assert_allclose(yy1, desired)

## 2-D polytope
def test_2d_polytope():
    basecurve1 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [1, 1])
    basecurve2 = CoefficientsCurve([1, 1], basecurve1)

    bounds2 = [(1, 0), (0, -1), (-1, 0)]
    clipcurve = LinearExtrapolationCurve(basecurve2, bounds2, [.1, .1], 1, lambda x: x)
    yy0 = basecurve2(np.array([[0, 0], [.5, -.5], [.2, -.1]]))
                  
    points2 = np.array([[0, .5], [1, -1], [.2, -.1]])
    slope0y = (basecurve2([0, 0]) - basecurve2([0, -.1])) / .1
    slope1xy = (basecurve2([.5, -.5]) - basecurve2([.5 - .1 * np.sqrt(2)/2, -.5 + .1 * np.sqrt(2)/2])) / .1
    np.testing.assert_allclose([slope0y, slope1xy], [1., 0.])

    yy1 = clipcurve(points2)
    desired = [yy0[0] + .5 * slope0y, yy0[1] + .5 * slope1xy * np.sqrt(2)/2, yy0[2]]
    np.testing.assert_allclose(yy1, desired)
