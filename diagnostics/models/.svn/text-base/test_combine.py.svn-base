import sys, StringIO
sys.path.append("../lib/models")

from model import Model
from ddp_model import DDPModel
from spline_model import SplineModel

mod1 = DDPModel.create_lin([0, 1], dict(control=[.5, .5], treatment=[0, 1]))
mod2 = DDPModel.create_lin([0, 1], dict(control=[.5, .5], treatment=[1, 0]))

def printmod(mod):
    output = StringIO.StringIO()
    mod.write(output, ',')
    print output.getvalue()

printmod(Model.combine([mod1, mod2], [.5, .5]))
printmod(Model.combine([mod1, mod2], [.25, 2]))

mod1 = SplineModel.create_gaussian(dict(control=(0, 1), treatment=(1, 1e-3)))
mod2 = SplineModel.create_gaussian(dict(control=(0, 1), treatment=(0, 1e-3)))

def printmod2(mod):
    output = StringIO.StringIO()
    mod.write_gaussian_plus(output, ',')
    print output.getvalue()

printmod2(Model.combine([mod1, mod2], [.5, .5]))
printmod2(Model.combine([mod1, mod2], [.25, 2]))
