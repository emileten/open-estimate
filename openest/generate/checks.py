import re
import numpy as np
import xarray as xr
from openest.generate.smart_curve import SmartCurve
from openest.models.curve import UnivariateCurve, StepCurve

PATTERN_NONWORD = re.compile(r"\W")

def loosematch(one, two):
    one = PATTERN_NONWORD.sub("", one).lower()
    two = PATTERN_NONWORD.sub("", two).lower()

    return one == two

def assert_conformant_weather_input(curve, values):
    if isinstance(curve, SmartCurve):
        if isinstance(values, xr.Dataset):
            return True
        else:
            assert False, "SmartCurves require Datasets."
    if isinstance(curve, UnivariateCurve):
        try:
            np.array(values)
        except Exception as ex:  # CATBELL
            import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
            assert False, "Trying to pass a non-array_like into a curve. Did you mean to use weather_change to extract a variable?"
        return True
    assert False, "Unknown curve type: %s" % curve.__class__

if __name__ == '__main__':
    print(loosematch("1,000 ", "1000"))
    print(loosematch("1 frog", "1 toad"))
    
