import numpy as np
from openest.models.curve import UnivariateCurve
import bounding

class LinearExtrapolationCurve(UnivariateCurve):
    def __init__(self, curve, bounds, margins, scaling, getindeps):
        super(LinearExtrapolationCurve, self).__init__(curve.xx)
        assert isinstance(bounds, list) or isinstance(bounds, dict)
        
        self.curve = curve
        self.bounds = bounds
        self.margins = margins
        self.scaling = scaling
        self.getindeps = getindeps

    def __call__(self, xs):
        values = self.curve(xs)
        indeps = self.getindeps(xs)

        known_slopes = {}
        if isinstance(self.bounds, dict):
            for ii, edgekey, invector in self.beyond_orthotope(indeps):
                if edgekey not in known_slopes:
                    slopes = []
                    for kk in self.bounds:
                        if invector[kk] == 0:
                            slopes.append(0)
                        else:
                            y0 = self.curve(indeps[ii, :] + invector)
                            y1 = self.curve(indeps[ii, :] + invector + self.margins[kk] * (np.sign(invector[kk]) * (np.arange(len(invector)) == kk)))
                            slopes.append(self.scaling * (y1 - y0) / self.margins[kk])
                    known_slopes[edgekey] = np.array(slopes)

                depen = self.curve(indeps[ii, :] + invector) + np.sum(known_slopes[edgekey] * -np.abs(invector))
                values[ii] = depen
                
        else:
            for ii, edgekey, invector in self.beyond_polytope(indeps):
                if edgekey not in known_slopes:
                    y0 = self.curve(indeps[ii, :] + invector)
                    y1 = self.curve(indeps[ii, :] + invector + self.margins * invector / np.linalg.norm(invector))
                    slope = self.scaling * (y1 - y0) / np.linalg.norm(self.margins * invector / np.linalg.norm(invector))
                    known_slopes[edgekey] = slope

                depen = self.curve(indeps[ii, :] + invector) + np.sum(known_slopes[edgekey] * -np.abs(invector))
                values[ii] = depen

        return(values)

    def beyond_orthotope(self, indeps):
        # Case 1: k-orthotope, provided by dict of {index: (low, high)}
        assert isinstance(self.bounds, dict)
        
        outside = np.zeros(indeps.shape[0], np.bool_)
        invector = np.zeros(indeps.shape)

        for kk in self.bounds:
            belows = self.bounds[kk][0] - indeps[:, kk]
            idx = np.nonzero(belows > 0)[0]
            outside[idx] = True
            invector[idx, kk] = belows[idx]
                
            aboves = indeps[:, kk] - self.bounds[kk][1]
            idx = np.nonzero(aboves > 0)[0]
            outside[idx] = True
            invector[idx, kk] = -aboves[idx]

        for ii in np.nonzero(outside)[0]:
            edgekey = np.sum(np.sign(invector[ii,:]) * (3 ** np.arange(invector.shape[1])))
            yield ii, edgekey, invector[ii,:]

    def beyond_polytope(self, indeps):
        # Case 2: Convex polytope
        assert isinstance(self.bounds, list)

        dists, edgekeys, bounds = bounding.within_convex_polytope(indeps, self.bounds)
        for ii in np.nonzero(dists > 0)[0]:
            yield ii, edgekeys[ii], -dists[ii] * bounds[edgekeys[ii]]['outvec']

from openest.models.curve import ZeroInterceptPolynomialCurve, CoefficientsCurve
basecurve1 = ZeroInterceptPolynomialCurve([-np.inf, np.inf], [1, 1])

## 1-D orthotope

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

## 2-D orthotope
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
np.testing.assert_allclose(yy1, desired)

## NEXT: Need the smart_curve version
