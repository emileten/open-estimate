from openest.lincombo.pooling import *
from numpy.testing import *

def test_disagg():
    mv = lincombo_pooled([4, 7, 2], [1, 3, 1], [[1, 0], [0, 1], [1, 0]])
    assert_equal(mv.means, [3, 7])
    assert_equal(mv.big_sigma, [[.5, 0], [0, 9.]])

def test_single():
    mv = lincombo_pooled([0, 1, 0], [1, 1, 2], [[1], [1], [1]])
    assert_almost_equal(mv.means, [4. / 9.])
    assert_almost_equal(mv.big_sigma, [[4. / 9.]])

def test_split():
    mv = lincombo_pooled([0, 1, 0], [1, 1, 1], [[1, 0], [.5, .5], [0, 1]])
    assert_almost_equal(mv.means, [1 / 3.] * 2)

def test_other():
    mv = lincombo_pooled([0, 1], [1, 1], [[1, 0], [.5, .5]])
    assert_equal(mv.means, [0, 2])
    assert_equal(mv.big_sigma, [[1, -1], [-1, 5]])

def test_partial():
    mv = lincombo_pooled([0, 1, 0], [1, 1, 10], [[1, 0], [.5, .5], [0, 1]])
    assert_almost_equal(mv.means, [0.01904762, 1.9047619])

if __name__ == '__main__':
    test_disagg()
    test_single()
    test_split()
    test_other()
    test_partial()
