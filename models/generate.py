import math
import numpy as np
from numpy import linalg
from spline_model import SplineModel, SplineModelConditional

def uniform_doseless(start, end, height=None):
    scaled = False
    if height is None:
        height = 1 / (end - start)
        scaled = True

    conditional = SplineModelConditional()
    conditional.add_segment(SplineModel.neginf, start, [SplineModel.neginf])
    conditional.add_segment(start, end, [math.log(height)])
    conditional.add_segment(end, SplineModel.posinf, [SplineModel.neginf])

    return SplineModel(True, [''], [conditional], scaled)

# Generate constant uniform
def uniform_constant(xx, yy, min, max):
    # header row
    table = [['ddp1']]
    table[0].extend(yy)

    nn = float(sum(np.logical_and(yy >= min, yy < max)))
    # data rows
    for ii in range(len(xx)):
        row = [xx[ii]]
        row.extend(np.logical_and(yy >= min, yy < max) / nn)
        table.append(row)

    return table

def polynomial(lowbound, highbound, betas, covas, num=40):
    betas = np.array(betas)
    covas = np.mat(covas)

    if covas.shape[0] != covas.shape[1] and len(betas) != covas.shape[0]:
        return "Error: Please provide a complete covariance matrix."
    if np.any(linalg.eig(covas)[0] < 0):
        return "Error: Covariance matrix is not positive definite."

    xx = np.linspace(lowbound, highbound, num=num)
    xxs = {}
    for x in xx:
        xvec = np.mat([[1, x, x**2, x**3][0:len(betas)]])
        serr = np.sqrt(xvec * covas * np.transpose(xvec))
        xxs[x] = (betas.dot(np.squeeze(np.asarray(xvec))), serr[0,0])

    return SplineModel.create_gaussian(xxs, xx, False)
