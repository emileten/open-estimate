import sys
sys.path.append("../../lib/models")

import numpy as np
from outer_multi_model import OuterMultiModel
from spline_model import *

# Linear model is like
#  y ~ N(a + b x + c z, sigma^2)
# coeffs = [b, c], const = a, var = sigma^2
# xxs is a list of lists, for x and y of their values
# xx_is_categoricals is a list of bools
def make_linreg_model_outer(coeffs, const, var, xxs, xx_is_categoricals):
    source = SplineModel(xx_is_categorical=True, scaled=True)
    xxs = outer_all(xxs)
    for xx in xxs:
        mean = const + sum([coeffs[ii] * xx[ii] for ii in range(len(coeffs))])
        source.add_conditional(":".join(map(str, xx)), SplineModelConditional.make_gaussian(SplineModel.neginf, SplineModel.posinf, mean, var))
    
    return OuterMultiModel(xxs, xx_is_categoricals, source)

def outer_all(xxs):
    cs = xxs[0]
    for xx in xxs[1:]:
        cs = outer_combos(cs, xx)

    return cs

def outer_combos(xx, bs):
    cs = []
    for x in xx:
        for b in bs:
            if isinstance(x, tuple):
                cs.append(x + (b,))
            elif isinstance(b, tuple):
                cs.append((x,) + b)
            else:
                cs.append((x, b))

    return cs

if __name__ == '__main__':
    model = make_linreg_model_outer([.5, -2], .5, 1, [np.linspace(0, 1, 10), np.linspace(0, 1, 10)], [False, False])
    model.write_file("reg1.csv", ",")
