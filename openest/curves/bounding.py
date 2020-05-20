"""Functions to the detection of polytope boundaries.

The main functions here, `ray_tracing_inside` and
`within_convex_polytope` are used to determine if a list of points is
contained in a polytope (the K-dimensional generalization of a
polygon).

For fewer than 4 dimensions, arbitrary polygons are supported, using a
ray-tracing method. Otherwise, it is only supported to detect if a
point is within a set of bounds, represented by K-dimensional planes.

Multiple representations of polytopes are supported. For 1-D and 2-D,
a list of points can be used, with the convention that points proceed
clockwise. For up-to 3-D, the polytope can represented as a list of
facets, each of which is a K-1 polytope (e.g., a polygon for a
polyhedron). For any dimension, convex polytopes can also be
represented by planes characterized by a point and an outward-facing
vector.
"""

import numpy as np

def ray_tracing_inside(points, polytope):
    """Determine if the set of points is within the polytope.

    The method passes a ray parallel to the x-axis. If it passes
    through an odd number of facets, the point is inside the polytope.

    Parameters
    ----------
    points: np.array of N x K
        N points with K dimensions
    polytope: list of K-tuples or K-1-polytope facets
        See main description for representations.

    Returns
    -------
    np.array of N booleans
        Each entry is True iff the corresponding point is within the
        polytope.
    """
    assert isinstance(polytope, list)

    # Set up the result array
    inside = np.zeros(points.shape[0], np.bool_)

    # Faster version for simple bounds
    if len(polytope[0]) == 1:
        for xxs in polytope:
            idx = np.nonzero(points[:, 0] < xxs[0])[0]
            inside[idx] = ~inside[idx]
        return inside

    # Iterate through facets, checking if passes through each
    for facet, outunit in facets(polytope): # facet is list of points
        if outunit[0] == 0:
            continue # Ray can't pass through
        # Check if it could intersect at all
        maxs = np.maximum.reduce(facet)
        mins = np.minimum.reduce(facet)
        idx = np.nonzero((points[:, 1:] > mins[1:]).all(axis=1) & (points[:, 1:] <= maxs[1:]).all(axis=1) & (points[:, 0] < maxs[0]))[0]
        if len(idx) == 0:
            continue
        # Define plane as dot(outunit, pt - facet[0]) = 0
        # Ray from point is point + a i-hat
        # Solve for a = dot(outunit, point - facet[0]) / outunit[0]
        raydist = -np.dot(points[idx,:] - facet[0], outunit) / outunit[0]
        pointsleft = raydist > 0 # only these rays pass through
        subinside = ray_tracing_inside(points[idx[pointsleft], 1:], [tuple(pt[1:]) for pt in facet])
        passedthrough = idx[pointsleft][subinside]
        inside[passedthrough] = ~inside[passedthrough]

    return inside

def facets(polytope):
    """Yields each of the K-1 facets and its corresponding outward unit vector.

    For determining the outward vector, 2-D polygon points are assumed
    to be organized clockwise and 3-D faces as such that the first
    three points have that (P2 - P1) x (P3 - P1) faces outward.

    Parameters
    ----------
    polytope: list of K-tuples or K-1-polytope facets
        See main description for representations.

    Yields
    ------
    2-tuple of list of points, and outward vector

    """
    assert isinstance(polytope, list)
    
    # Check how the polytope is represented
    if isinstance(polytope[0], list):
        assert len(polytope[0][0]) <= 3
        dim = '<=3'
    else:
        dim = len(polytope[0])
    
    for ii in range(len(polytope)):
        # Determine the facet points
        if dim == '<=3':
            facet = polytope[ii]
        elif ii + dim <= len(polytope):
            facet = polytope[ii:(ii+dim)]
        else:
            facet = polytope[ii:] + polytope[:(ii + dim - len(polytope))]

        # Determine the outward vector
        if dim == 1:
            if polytope[ii][0] < polytope[(ii+1) % len(polytope)][0]:
                outunit = (-1,)
            else:
                outunit = (1,)
        elif dim == 2:
            along = np.array(facet[1]) - np.array(facet[0])
            alongunit = along / np.linalg.norm(along)
            outunit = (-alongunit[1], alongunit[0])
        elif dim == '<=3':
            if len(facet[0]) < 3:
                _drop, outunit = next(facets(facet[0]))
            else:
                one = np.array(facet[1]) - np.array(facet[0])
                two = np.array(facet[2]) - np.array(facet[0])
                outvec = np.cross(one, two)
                outunit = outvec / np.linalg.norm(outvec)
            
        yield facet, outunit

def within_convex_polytope(points, bounds):
    """Determine if the set of points is within convex bounds.

    Each bound is characterized by a point and an outward vector,
    defining a plane. Those points that are on the "out" side of the
    plane are consider outside the bounds. Returns which bound is
    exceeded and by how much.

    Parameters
    ----------
    points: np.array of N x K
        N points with K dimensions
    polytope: list of K-tuples, K-1-polytope facets, or dict of point, outvec
        See main description for representations.

    Returns
    -------
    2-tuple of np.array of N dists and np.array of N indexes
        If the distance corresponding to a point is < inf, then this is
        the distance to the closest exceeded bound, and the index will
        be non-NaN identifying the exceeded bound.
    """
    assert isinstance(bounds, list)

    # Convert to a canonical {point, outvec} form.
    if not isinstance(bounds[0], dict):
        bounds = [dict(point=facet[0], outvec=outunit) for facet, outunit in facets(bounds)]

    # Set up the result arrays
    beyond_dists = np.ones(points.shape[0]) * np.inf
    beyond_bounds = np.zeros(points.shape[0]) * np.nan

    # Check if outside of each bound
    for ii, bound in enumerate(bounds):
        # Calcuate the distance along the outvec
        outunit = bound['outvec'] / np.linalg.norm(bound['outvec'])
        outdists = np.dot(points - bound['point'], outunit)
        # Update the results if outside and less outside than other estimates
        idx = np.nonzero((outdists > 0) & (outdists < beyond_dists))[0]
        beyond_dists[idx] = outdists[idx]
        beyond_bounds[idx] = ii

    return beyond_dists, beyond_bounds, bounds
## [bounds[ii] if ii != 0 else None for ii in beyond_bounds]
