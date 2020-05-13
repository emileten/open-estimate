# class LinearExtrapolationCurve(UnivariateCurve):
#     def __init__(self, curve, limits, margin, scaling, getindep):
#         super(UShapedCurve, self).__init__(curve.xx)
#         self.curve = curve
#         self.limits = limits
#         self.margin = margin
#         self.scaling = scaling
#         self.getindep = getindep

#     def __call__(self, xs):
#         values = self.curve(xs)
#         indeps = self.getindep(xs)

#         known_affines = {}
#         for ii, indep, edge, invector in self.beyond(indeps):
#             if edge not in known_affines:
#                 y0 = self.curve(indep + invector)
#                 y1 = self.curve(indep + invector + self.margin * invector / numpy.linalg.norm(invector))
#                 slope = self.scaling * (y1 - y0) / self.margin
#                 known_affines[edge] = (y0, slope)

#             depen = known_affines[edge][0] + sum(known_affines[edge][0] * -invector)
#             values[ii] = depen

import numpy as np

def ray_tracing_numpy(points, polytope):
    ## polytope: list of K-tuples
    ## points: np.array of N x K

    inside = np.zeros(points.shape[0], np.bool_)
    if len(polytope[0]) == 1:
        for xxs in polytope:
            idx = np.nonzero(points[:, 0] < xxs[0])[0]
            inside[idx] = ~inside[idx]
        return inside
            
    for facet, outunit in facets(polytope): # facet is list of points
        if outunit[0] == 0:
            continue # Ray can't pass through
        maxs = np.maximum.reduce(facet)
        mins = np.minimum.reduce(facet)
        idx = np.nonzero((points[:, 1:] > mins[1:]).all(axis=1) & (points[:, 1:] <= maxs[1:]).all(axis=1) & (points[:, 0] < maxs[0]))[0]
        if len(idx) == 0:
            continue
        # Define plane as dot(outunit, pt - facet[0]) = 0
        # Ray from point is point + a i-hat
        # Solve for a = dot(outunit, point - facet[0]) / outunit[0]
        raydist = -np.dot(points[idx,:] - facet[0], outunit) / outunit[0]
        pointsleft = raydist > 0 #((outunit[0] > 0) & (raydist < 0)) | ((outunit[0] < 0) & (raydist > 0)) # only these rays pass through
        subinside = ray_tracing_numpy(points[idx[pointsleft], 1:], [tuple(pt[1:]) for pt in facet])
        passedthrough = idx[pointsleft][subinside]
        inside[passedthrough] = ~inside[passedthrough]

    return inside

def facets(polytope):
    dim = len(polytope[0]) if not isinstance(polytope[0], list) else 'N'
    
    for ii in range(len(polytope)):
        if dim == 'N':
            facet = polytope[ii]
        elif ii + dim <= len(polytope):
            facet = polytope[ii:(ii+dim)]
        else:
            facet = polytope[ii:] + polytope[:(ii + dim - len(polytope))]

        if dim == 1:
            if polytope[ii][0] < polytope[(ii+1) % len(polytope)][0]:
                outunit = (-1,)
            else:
                outunit = (1,)
        elif dim == 2:
            along = np.array(facet[1]) - np.array(facet[0])
            alongunit = along / np.linalg.norm(along)
            outunit = (-alongunit[1], alongunit[0])
        elif dim == 'N':
            one = np.array(facet[1]) - np.array(facet[0])
            two = np.array(facet[2]) - np.array(facet[0])
            outvec = np.cross(one, two)
            outunit = outvec / np.linalg.norm(outvec)
                
        yield facet, outunit

## 1-D test
print("1-D")
polytope = [(.25,), (.75,)]
iter = facets(polytope)
np.testing.assert_equal(next(iter), ([(0.25,)], (-1,)))
np.testing.assert_equal(next(iter), ([(0.75,)], (1,)))

points = np.expand_dims(np.array([.1, .2, .4, .8]), axis=-1)

inside = ray_tracing_numpy(points, polytope)
np.testing.assert_equal(inside, [False, False, True, False])

## 2-D test
print("2-D")

polytope = [(1, 0), (0, -1), (-1, 0), (0, 1)]
iter = facets(polytope)
np.testing.assert_equal(next(iter), ([(1, 0), (0, -1)], (0.7071067811865475, -0.7071067811865475)))
np.testing.assert_equal(next(iter), ([(0, -1), (-1, 0)], (-0.7071067811865475, -0.7071067811865475)))
np.testing.assert_equal(next(iter), ([(-1, 0), (0, 1)], (-0.7071067811865475, 0.7071067811865475)))
np.testing.assert_equal(next(iter), ([(0, 1), (1, 0)], (0.7071067811865475, 0.7071067811865475)))

points = np.array([[.75, -.5], [.75, .05], [-.75, .5], [-.75, .05]])

inside = ray_tracing_numpy(points, polytope)
np.testing.assert_equal(inside, [False, True, False, True])

## 3-D test
print("3-D")
polytope = [[(0, 0, 1), (1, .5, 0), (-1, .5, 0)], # front
            [(1, .5, 0), (0, -1, 0), (-1, .5, 0)], # bottom
            [(-1, .5, 0), (0, -1, 0), (0, 0, 1)], # left
            [(1, .5, 0), (0, 0, 1), (0, -1, 0)]] # right
iter = facets(polytope)
facet, outunit = next(iter)
np.testing.assert_almost_equal(facet, [(0, 0, 1), (1, 0.5, 0), (-1, 0.5, 0)])
np.testing.assert_almost_equal(outunit, np.array([0.        , 0.89442719, 0.4472136 ]))
facet, outunit = next(iter)
np.testing.assert_almost_equal(facet, [(1, 0.5, 0), (0, -1, 0), (-1, 0.5, 0)])
np.testing.assert_almost_equal(outunit, np.array([-0.,  0., -1.]))
facet, outunit = next(iter)
np.testing.assert_almost_equal(facet, [(-1, 0.5, 0), (0, -1, 0), (0, 0, 1)])
np.testing.assert_almost_equal(outunit, np.array([-0.72760688, -0.48507125,  0.48507125]))
facet, outunit = next(iter)
np.testing.assert_almost_equal(facet, [(1, 0.5, 0), (0, 0, 1), (0, -1, 0)])
np.testing.assert_almost_equal(outunit, np.array([ 0.72760688, -0.48507125,  0.48507125]))

points = np.array([(0, 1, .05), (0, .5, .05), (0, 0, .05), (0, -.5, .05), (0, 0, .75), (0, -.5, .75)])
inside = ray_tracing_numpy(points, polytope)

np.testing.assert_equal(inside, [False, False, True, True, True, False])
