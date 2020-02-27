import numpy as np
from aggregator.lib.bayes.spline_model import SplineModel, SplineModelConditional
from aggregator.lib.bayes.features_interpreter import FeaturesInterpreter

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [0,0.58245,0.28391,1.1949], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [8,0.59244,0.30072,1.1672], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [8.25,0.60302,0.31794,1.1437], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [8.5,0.61398,0.33583,1.1225], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [8.75,0.62532,0.35437,1.1035], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])

cond = FeaturesInterpreter.features_to_gaussian(['dpc1', 'mean', .025, .975], [9,0.63715,0.37326,1.0876], (SplineModel.neginf, SplineModel.posinf))
print(cond.find_mode()[0])
