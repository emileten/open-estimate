"""Curve definition that implements linear extrapolation clipping.

These Curve classes take a curve and represent it with a linear
extrapolation beyond certain bounds. Slopes for the linear
extrapolation are determined by the behavior of the curve within some
margin of the bounds.

Two options are available for representing bounds, and they imply
different logic for clipping. If the bounds are a (convex) polytope,
the extrapolation for a given point is determined by the orthogonal
vector from whatever facet it lies closest to. If that facet is
slanted, slopes will be applied to multiple dimensions. If the bounds
are given as a set of limits for each dimension, the extrapolation for
a given point is determined separately for each of its
dimensions. Only those dimensions that it lies beyond will be
adjusted.
"""

import numpy as np
from openest.models.curve import UnivariateCurve
from . import bounding


class LinearExtrapolationCurve(UnivariateCurve):
    """Linear extrapolation clipping curve which takes a list of multidimensional points.

    Bounds can be described as a polytope (see options in
    bounding.py), or a dictionary of bounds (see above for difference
    in logic). In the latter case, the dictionary should have a key
    for each dimension that should be clipped. The values of the
    dictionary should be 2-tuples with the high and low bounds for
    that dimension.

    Margins should be provided for each dimension. These are allowed
    to be different by dimension, and should depend on the scale of
    the variable.

    The inputs should be a list of N x L, where L is the total number
    of variables. This may be smaller than the number of independent
    dimensions, if some of the variables are transformations of
    others. For example, a 4th order polynomial input would be applied
    to the curve as an N x 4, but only the first dimension in the
    independent variable. The `getindeps` function provides this
    translation, and in the polynomial case would return an N x 1
    matrix.

    Parameters
    ----------
    curve : UnivariateCurve
        Curve to be clipped
    bounds : dict or list
        Either a dictionary of {dim: (lower, upper)} bounds or a polytope
    margins : array_like
        A array_like with a margin for each dimension
    scaling : float
        A factor that the slope is scaled by
    getindeps : function(array_like) -> array_like
        Translates from the full variable matrix to just the independent variables.
    """
    
    def __init__(self, curve, bounds, margins, scaling, getindeps):
        super(LinearExtrapolationCurve, self).__init__(curve.xx)
        assert isinstance(bounds, list) or isinstance(bounds, dict)
        
        self.curve = curve
        self.bounds = bounds
        self.margins = margins
        self.scaling = scaling
        self.getindeps = getindeps

    def __call__(self, xs):
        """Returns the projected variables after clipping.

        The values should be appropriate for passing to both the
        internal curve and the `getindeps` function.

        Parameters
        ----------
        xs : array_like
            A matrix of N x L with all input data.

        Returns
        -------
        array_like
            A sequence of values after clipping.
        """
        values = self.curve(xs)
        indeps = self.getindeps(xs)

        return replace_oob(values, indeps, self.curve, self.bounds, self.margins, self.scaling)

    
def replace_oob(values, indeps, curve, bounds, margins, scaling):
    """Replace out-of-bound point values.

    This is split out from LinearExtrapolationCurve to make
    smart-generalization easier.

    Parameters
    ----------
    values : array_like
        A vector of before clipping curve values
    indeps : array_like
        A matrix of N x K with independent variables
    curve : UnivariateCurve
        Curve to be clipped
    bounds : dict or list
        Either a dictionary of {dim: (lower, upper)} bounds or a polytope
    margins : array_like
        A array_like with a margin for each dimension
    scaling : float
        A factor that the slope is scaled by

    Returns
    -------
    array_like
        A vector of values after clipping.
    """

    known_slopes = {}
    if isinstance(bounds, dict):
        for ii, edgekey, invector in beyond_orthotope(bounds, indeps):
            if edgekey not in known_slopes:
                slopes = []
                for kk in bounds:
                    if invector[kk] == 0:
                        slopes.append(0)
                    else:
                        y0 = curve(indeps[ii, :] + invector)
                        y1 = curve(indeps[ii, :] + invector + margins[kk] * (np.sign(invector[kk]) * (np.arange(len(invector)) == kk)))
                        slopes.append(scaling * (y1 - y0) / margins[kk])
                known_slopes[edgekey] = np.array(slopes)

            depen = curve(indeps[ii, :] + invector) + np.sum(known_slopes[edgekey] * -np.abs(invector))
            values[ii] = depen

    else:
        for ii, edgekey, invector in beyond_polytope(bounds, indeps):
            if edgekey not in known_slopes:
                y0 = curve(indeps[ii, :] + invector)
                y1 = curve(indeps[ii, :] + invector + margins * invector / np.linalg.norm(invector))
                slope = scaling * (y1 - y0) / np.linalg.norm(margins * invector / np.linalg.norm(invector))
                known_slopes[edgekey] = slope

            depen = curve(indeps[ii, :] + invector) + np.sum(known_slopes[edgekey] * -np.abs(invector))
            values[ii] = depen

    return values


def beyond_orthotope(bounds, indeps):
    """Yields each point that needs to clipped for orthotope bounds.

    Checks whether each point is beyond any of the bounds and
    yields the following for those points that need to be clipped:
      (ii, edgekey, invector)

    where ii is the index of the point, edgekey is a numeric key
      unique to every combination of bounds that can be crossed,
      and invector is a vector from the point to the closest
      orthogonal point on the boundary.

    Parameters
    ----------
    indeps : array_like
        A matrix of N x K with independent variables

    Yields
    ------
    Tuple of (int, int, array_like)

    """
    # Case 1: k-orthotope, provided by dict of {index: (low, high)}
    assert isinstance(bounds, dict)

    outside = np.zeros(indeps.shape[0], np.bool_)
    invector = np.zeros(indeps.shape)

    for kk in bounds:
        belows = bounds[kk][0] - indeps[:, kk]
        idx = np.nonzero(belows > 0)[0]
        outside[idx] = True
        invector[idx, kk] = belows[idx]

        aboves = indeps[:, kk] - bounds[kk][1]
        idx = np.nonzero(aboves > 0)[0]
        outside[idx] = True
        invector[idx, kk] = -aboves[idx]

    for ii in np.nonzero(outside)[0]:
        edgekey = np.sum(np.sign(invector[ii,:]) * (3 ** np.arange(invector.shape[1])))
        yield ii, edgekey, invector[ii,:]

        
def beyond_polytope(bounds, indeps):
    """Yields each point that needs to clipped for a convex polytope.

    As with beyond_orthotope, except in the calculation of the yielded outputs:
      (ii, edgekey, invector)

    where ii is the index of the point to be clipped, edgekey is
      the index of the facet exceeded, and invector is an
      orthogonal vector to the exceeded facet.

    Parameters
    ----------
    indeps : array_like
        A matrix of N x K with independent variables

    Yields
    ------
    Tuple of (int, int, array_like)

    """
    # Case 2: Convex polytope
    assert isinstance(bounds, list)

    dists, edgekeys, bounds = bounding.within_convex_polytope(indeps, bounds)
    for ii in np.nonzero(dists < np.inf)[0]:
        yield ii, edgekeys[ii], -np.array(dists[ii]) * np.array(bounds[int(edgekeys[ii])]['outvec'])
