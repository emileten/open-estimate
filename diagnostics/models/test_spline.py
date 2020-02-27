import sys, io
sys.path.append("../lib/models")

from spline_model import SplineModel
from features_interpreter import FeaturesInterpreter

def test_exponential():
    fp = io.StringIO()
    fp.write("dpc1,mean\n")
    fp.write("0,1\n")
    fp.seek(0)

    output = io.StringIO()

    FeaturesInterpreter.init_from_feature_file(SplineModel(), fp, ',', (0, 100)).write(output, ',')
    print(output.getvalue())

    fp.close()
    output.close()

    # 0,0,100,-0.0486320833207,-1.0

def test_gaussian():
    fp = io.StringIO()
    fp.write("dpc1,mean,var\n")
    fp.write("0,5,1\n")
    fp.seek(0)

    output = io.StringIO()

    FeaturesInterpreter.init_from_feature_file(SplineModel(), fp, ',', (0, 10)).write(output, ',')
    print(output.getvalue())

    fp.close()
    output.close()

    # 0,0,10,-25.5713644426,10.0,-1.0

def test_gaussian2():
    fp = io.StringIO()
    fp.write("dpc1,mean,.025,.975\n")
    fp.write("0,0,-2,2\n")
    fp.seek(0)

    output = io.StringIO()

    FeaturesInterpreter.init_from_feature_file(SplineModel(), fp, ',', ('-inf', 'inf')).write_gaussian(output, ',')
    print(output.getvalue())

    fp.close()
    output.close()

    # 0,0,10,-25.5713644426,10.0,-1.0

def test_non_gaussian():
    fp = io.StringIO()
    fp.write("dpc1,mean,.025,.975\n")
    fp.write("0,0,-1,2\n")
    fp.seek(0)

    output = io.StringIO()

    FeaturesInterpreter.init_from_feature_file(SplineModel(), fp, ',', ('-inf', 'inf')).write(output, ',')
    print(output.getvalue())

    fp.close()
    output.close()

    # 0,0,10,-25.5713644426,10.0,-1.0

if __name__ == "__main__":
    test_exponential()
    test_gaussian()
    test_gaussian2()
    test_non_gaussian()
