import pytest
import numpy as np

from openest.curves.bounding import *

## 1-D test
def test_1d():
    polytope = [(.25,), (.75,)]
    iter = facets(polytope)
    np.testing.assert_equal(next(iter), ([(0.25,)], (-1,)))
    np.testing.assert_equal(next(iter), ([(0.75,)], (1,)))

    points = np.expand_dims(np.array([.1, .2, .4, .8]), axis=-1)

    inside = ray_tracing_inside(points, polytope)
    np.testing.assert_equal(inside, [False, False, True, False])

    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, polytope)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, False, True, False])

## 2-D test
def test_2d():
    polytope = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    iter = facets(polytope)
    np.testing.assert_equal(next(iter), ([(1, 0), (0, -1)], (0.7071067811865475, -0.7071067811865475)))
    np.testing.assert_equal(next(iter), ([(0, -1), (-1, 0)], (-0.7071067811865475, -0.7071067811865475)))
    np.testing.assert_equal(next(iter), ([(-1, 0), (0, 1)], (-0.7071067811865475, 0.7071067811865475)))
    np.testing.assert_equal(next(iter), ([(0, 1), (1, 0)], (0.7071067811865475, 0.7071067811865475)))

    points = np.array([[.75, -.5], [.75, .05], [-.75, .5], [-.75, .05]])

    inside = ray_tracing_inside(points, polytope)
    np.testing.assert_equal(inside, [False, True, False, True])

    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, polytope)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, True, False, True])

## 3-D test
def test_3d():
    polytope = [[(0, 0, 1), (1, .5, 0), (-1, .5, 0)], # front
                [(1, .5, 0), (0, -1, 0), (-1, .5, 0)], # bottom
                [(-1, .5, 0), (0, -1, 0), (0, 0, 1)], # left
                [(1, .5, 0), (0, 0, 1), (0, -1, 0)]] # right
    iter = facets(polytope)
    facet, outunit1 = next(iter)
    np.testing.assert_almost_equal(facet, [(0, 0, 1), (1, 0.5, 0), (-1, 0.5, 0)])
    np.testing.assert_almost_equal(outunit1, np.array([0.        , 0.89442719, 0.4472136 ]))
    facet, outunit2 = next(iter)
    np.testing.assert_almost_equal(facet, [(1, 0.5, 0), (0, -1, 0), (-1, 0.5, 0)])
    np.testing.assert_almost_equal(outunit2, np.array([-0.,  0., -1.]))
    facet, outunit3 = next(iter)
    np.testing.assert_almost_equal(facet, [(-1, 0.5, 0), (0, -1, 0), (0, 0, 1)])
    np.testing.assert_almost_equal(outunit3, np.array([-0.72760688, -0.48507125,  0.48507125]))
    facet, outunit4 = next(iter)
    np.testing.assert_almost_equal(facet, [(1, 0.5, 0), (0, 0, 1), (0, -1, 0)])
    np.testing.assert_almost_equal(outunit4, np.array([ 0.72760688, -0.48507125,  0.48507125]))
    
    points = np.array([(0, 1, .05), (0, .5, .05), (0, 0, .05), (0, -.5, .05), (0, 0, .75), (0, -.5, .75)])
    inside = ray_tracing_inside(points, polytope)

    np.testing.assert_equal(inside, [False, False, True, True, True, False])

    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, polytope)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, False, True, True, True, False])

## 4-D test
def test_4d():
    # Get out-vectors from 3D form
    polytope = [[(0, 0, 1), (1, .5, 0), (-1, .5, 0)], # front
                [(1, .5, 0), (0, -1, 0), (-1, .5, 0)], # bottom
                [(-1, .5, 0), (0, -1, 0), (0, 0, 1)], # left
                [(1, .5, 0), (0, 0, 1), (0, -1, 0)]] # right
    iter = facets(polytope)
    facet, outunit1 = next(iter)
    facet, outunit2 = next(iter)
    facet, outunit3 = next(iter)
    facet, outunit4 = next(iter)

    bounds = [dict(point=(0, 0, 1, 0), outvec=np.concatenate((outunit1, (0,)))), # front
              dict(point=(1, .5, 0, 0), outvec=np.concatenate((outunit2, (0,)))), # bottom
              dict(point=(-1, .5, 0, 0), outvec=np.concatenate((outunit3, (0,)))), # left
              dict(point=(1, .5, 0, 0), outvec=np.concatenate((outunit4, (0,)))), # right
              dict(point=(0, 0, 0, -1), outvec=(0, 0, 0, -1)), # near
              dict(point=(0, 0, 0, 1), outvec=(0, 0, 0, 1))] # far

    points = np.array([(0, 1, .05, -2), (0, .5, .05, -2), (0, 0, .05, -2), (0, -.5, .05, -2), (0, 0, .75, -2), (0, -.5, .75, -2)])
    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, bounds)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, False, False, False, False, False])
    
    points = np.array([(0, 1, .05, 0), (0, .5, .05, 0), (0, 0, .05, 0), (0, -.5, .05, 0), (0, 0, .75, 0), (0, -.5, .75, 0)])
    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, bounds)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, False, True, True, True, False])

    points = np.array([(0, 1, .05, 2), (0, .5, .05, 2), (0, 0, .05, 2), (0, -.5, .05, 2), (0, 0, .75, 2), (0, -.5, .75, 2)])
    beyond_dists, beyond_bounds, bounds = within_convex_polytope(points, bounds)
    np.testing.assert_equal(np.isnan(beyond_bounds), [False, False, False, False, False, False])
