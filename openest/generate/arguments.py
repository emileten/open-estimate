"""Collection of argument types, based on the ArgumentType system.

This file instantiates a variety of example ArgumentType objects, in
common use across the Calculation objects.
"""

import numpy as np
from openest.models.curve import UnivariateCurve, LinearCurve
from openest.models.univariate_model import UnivariateModel
from openest.generate.calculation import Calculation
from openest.generate.base import Constant
from openest.generate.curvegen import CurveGenerator, ConstantCurveGenerator
from .arguments_base import *

## Meta-calculation

calculation = ArgumentType("calculation", "A previous calculation step.",
                           [Calculation], lambda context: Constant(np.pi, 'widgets/radius'))
calculationss = ArgumentType("calculations", "A list of previous calculations steps.",
                             [list], lambda context: [Constant(np.pi, 'widgets/radius')])
model = ArgumentType("model", "A univariate curve or OpenEstimate model.",
                              [UnivariateCurve, UnivariateModel], lambda context: LinearCurve(.5))
qval = ArgumentType('qval', "Quantile of uncertainty, or seed for multivariate uncertainty.",
                    [float, int, type(None)], lambda context: .5)
curvegen = ArgumentType("curvegen", "A subclass of CurveGenerator.", [CurveGenerator],
                        lambda context: ConstantCurveGenerator(['resources'], 'widgets', LinearCurve(.5)))
curve_or_curvegen = ArgumentType("curve_or_curvegen", "A subclass of UnivariateCurve or CurveGenerator.",
                                 [UnivariateCurve, CurveGenerator], lambda context: LinearCurve(.5))
input_change = ArgumentType('input_change', "A function which transform the input data before application.",
                            [type(lambda x: x), UnivariateCurve], lambda context: lambda x: np.log(x))
input_reduce = ArgumentType('input_reduce', "A function of multiple arguments which combines the inputs.",
                            [type(lambda x: x)], lambda context: lambda x, y: x * y)

## Units

input_unit = ArgumentType("input_unit", "Units for the input.",
                          [str], lambda context: 'resources')
output_unitss = ArgumentType("ouput_unitss", "Units for each of the results.",
                      [list], lambda context: ['widgets'])

## Values

ordered_list = ArgumentType('ordered_list', "A list of numeric values, in order.",
                            [list], lambda context: [-np.inf, 0, np.inf])
region_dictionary = ArgumentType('region_dictionary', "A dictionary with a value for each region.",
                                 [dict], lambda context: {'here': np.pi})
year = ArgumentType('year', "A specified year.", [int], lambda context: 1981)
time = ArgumentType('time', "A specified time.", [int], lambda context: 1981)
monthvalues = ArgumentType('monthvalues', "The values for each month, in a list starting with January.",
                           [list], lambda context: [np.pi] * 12)
regions = ArgumentType('regions', "The list of region names.", [list], lambda context: ['here'])
parameter_getter = ArgumentType('parameter_getter', "A function that retrieves the underlying parameters from an object.",
                                [type(lambda x: x)], lambda x: x.coeffs)
coefficient = ArgumentType('coeff', "A given coefficient.", [float], lambda context: 3.)

## Configuration

debugging = ArgumentType('debugging', "A boolean to turn on a debugging system or diagnosting output.",
                         [bool], lambda context: True)
skip_on_missing = ArgumentType('skip_on_missing', "A boolean to not report results if we never see the baseline year.",
                               [bool], lambda context: True)
unshift = ArgumentType('unshift', "A boolean to turn off (if false) the recording of outputs.",
                       [bool], lambda context: True)

## Description

column_names = ArgumentType('column_names', "A list of the names of the result columns.",
                            [list], lambda context: ['capacity'])
column_titles = ArgumentType('column_titles', "A list of the titles of the result columns.",
                            [list], lambda context: ['Capacity of widget production.'])
column_descriptions = ArgumentType('column_descriptions', "A list of the descriptions of the result columns.",
                                   [list], lambda context: ['A calculation that produces the capacity of widget producing.'])
curve_description = ArgumentType('curve_description', "A text description of the curve.",
                                 str, lambda context: "A basic curve.")
description = ArgumentType('description', "An english description of what an operation does.",
                           str, lambda context: "Geworfenheit")
label = ArgumentType('label', "A short name for this result", str, lambda context: 'token')
