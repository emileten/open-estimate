import math
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

