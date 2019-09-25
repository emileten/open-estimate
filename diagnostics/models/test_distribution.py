import sys, StringIO
sys.path.append("../lib/models")

from ddp_model import DDPModel
from spline_model import SplineModel
from distribution_model import DistributionModel
from features_interpreter import FeaturesInterpreter

def test_tc(modelfile, distfile):
    with open(modelfile, "r") as modelfp:
        model = DDPModel()
        model.init_from(modelfp, ',')
        
    with open(distfile, "r") as distfp:
        dist = DistributionModel()
        dist.init_from(distfp, ',')

    output = StringIO.StringIO()

    dist.apply_as_distribution(model).write(output, ',')
    print output.getvalue()

    output.close()

def test_gtc(modelfile, distfile):
    with open(modelfile, "r") as modelfp:
        model = SplineModel()
        FeaturesInterpreter.init_from_feature_file(model, modelfp, ',', (SplineModel.neginf, SplineModel.posinf))
        
    with open(distfile, "r") as distfp:
        dist = DistributionModel()
        dist.init_from(distfp, ',')

    output = StringIO.StringIO()

    dist.apply_as_distribution(model).write(output, ',')
    print output.getvalue()

    output.close()
    
if __name__ == "__main__":
    test_tc('example-tc.csv', 'dist-tc.csv')
    test_tc('example-tc.csv', 'dist-tc2.csv')
    test_gtc('example-gtc.csv', 'dist-tc2.csv')
