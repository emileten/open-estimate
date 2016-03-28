import numpy as np
import latextools, calculation

"""Scale the results by the value in scale_dict, or the mean value (if it is set).
make_generator: we encapsulate this function, passing in data and opporting on outputs
func: default operation is to multiple (scale), but can do other things (e.g., - for re-basing)
"""
class Scale(calculation.Calculation):
    def __init__(self, subcalc, scale_dict, from_units, to_units, func=lambda x, y: x*y, latexpair=(r"\bar{I}", "Region-specific scaling")):
        super(Scale, self).__init__([to_units] + subcalc.unitses)
        assert(subcalc.unitses[0] == from_units)

        self.subcalc = subcalc
        self.scale_dict = scale_dict
        self.func = func
        self.latexpair = latexpair
        self.from_units = from_units

    def latex(self, *args, **kwargs):
        for (key, value, units) in self.subcalc.latex(*args, **kwargs):
            if key == "Equation":
                for eqnstr in latextools.call(self.func, self.unitses[0], "Scaling function", value, self.latexpair[0]):
                    yield eqnstr
            else:
                yield (key, value, units)

        fulllatexpair = (self.latexpair[0], self.latexpair[1], self.from_units + ' -> ' + self.unitses[0])
        yield fulllatexpair

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            if region in self.scale_dict:
                return self.func(result, self.scale_dict[region])
            else:
                return self.func(result, self.scale_dict['mean'])

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Scaled ' + infos[0]['title']
        equation = latextools.english_function(self.func, infos[0]['name'], self.latexpair[1])
        description = "Computed from the %s variable, as %s." % (infos[0]['name'], equation)
        return [dict(name='scaled', title=title, description=description)] + infos

"""
Transform all results by a function.
"""
class Transform(calculation.Calculation):
    def __init__(self, subcalc, from_units, to_units, func, description, long_description):
        super(Transform, self).__init__([to_units] + subcalc.unitses)
        assert(subcalc.unitses[0] == from_units)

        self.subcalc = subcalc
        self.func = func
        self.description = description
        self.long_description = long_description
        self.from_units = from_units

    def latex(self, *args, **kwargs):
        for (key, value, units) in self.subcalc.latex(*args, **kwargs):
            if key == "Equation":
                for eqnstr in latextools.call(self.func, self.unitses[0], self.description, value):
                    yield eqnstr
            else:
                yield (key, value, units)

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return self.func(result)

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = self.description
        description = self.long_description
        return [dict(name='transformed', title=title, description=description)] + infos

## make-apply logic for generating make_generators

def make(handler, make_generator, *handler_args, **handler_kw):
    """Construct a generator from a function, operating on the results of another generator.
    handler(generator, *handler_args, **handler_kw) takes an enumerator and returns an enumerator
    """

    if 'latexhandler' in handler_kw:
        latexhandler = handler_kw['latexhandler']
        del handler_kw['latexhandler']
    else:
        latexhandler = handler

    # The make_generator function to return
    def generate(fips, yyyyddd, temps, *args, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            # Pass on signal for end
            print "completing make"
            make_generator(fips, yyyyddd, temps, *args, **kw).next()
            return

        # Pass on data
        generator = make_generator(fips, yyyyddd, temps, *args, **kw)

        if fips == effect_bundle.LATEX_STRING:
            for equation, latex in generator:
                if equation == "Equation":
                    for (equation2, latex2) in latexhandler(latex, *handler_args, **handler_kw):
                        yield (equation2, latex2)
                else:
                    yield (equation, latex)
            return

        # Apply function to results of data
        for yearresult in handler(generator, *handler_args, **handler_kw):
            yield yearresult

    return generate

class Instabase(calculation.CustomFunctionalCalculation):
    """Re-base the results of make_generator(...) to the values in baseyear
    baseyear is the year to use as the 'denominator'; None for the first year
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    Tacks on the value to the front of the results
    """

    def __init__(self, subcalc, baseyear, func=lambda x, y: x / y, units='portion', skip_on_missing=True):
        super(Instabase, self).__init__(subcalc, subcalc.unitses[0], units, True, baseyear, func, skip_on_missing)
        self.baseyear = baseyear
        self.denom = None # The value in the baseyear
        self.pastresults = [] # results before baseyear

    def latexhandler(self, equation, baseyear, func, skip_on_missing):
        for eqnstr in latextools.call(func, self.unitses[0], "Re-basing function", equation, r"\left[%s\right]_{t = %d}" % (equation, baseyear)):
            yield eqnstr

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, yyyyddd, weather, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(yyyyddd, weather):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == baseyear or (baseyear is None and self.denom is None):
                self.denom = result

                # Print out all past results, relative to this year
                for self.pastresult in self.pastresults:
                    yield [self.pastresult[0], func(self.pastresult[1], self.denom)] + list(self.pastresult[1:])

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
            else:
                # calculate this and tack it on
                yield [year, func(result, self.denom)] + list(yearresult[1:])

    def donehandler(self, baseyear, func, skip_on_missing):
        if self.denom is None and skip_on_missing:
            # Never got to this year: just write out results
            for pastresult in self.pastresults:
                yield pastresult

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Rebased ' + infos[0]['title']
        description = "The result calculated relative to the year %d, by re-basing variable %s." % (self.baseyear, infos[0]['name'])
        return [dict(name='rebased', title=title, description=description)] + infos

class SpanInstabase(Instabase):
    """Re-base the results of a calculation to the average of values between two years.
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    """
    def __init__(self, subcalc, year1, year2, func=lambda x, y: x / y, units='portion', skip_on_missing=True):
        super(SpanInstabase, self).__init__(subcalc, (year1 + year2) / 2, func, units, skip_on_missing)
        self.year1 = year1
        self.year2 = year2
        self.denomterms = []

    def latexhandler(self, equation, baseyear, func, skip_on_missing):
        for eqnstr in latextools.call(func, self.unitses[0], "Re-basing function", equation, r"Average\left[%s\right]_{%d \le t le %d}" % (equation, self.year1, self.year2)):
            yield eqnstr

    def pushhandler(self, yyyyddd, weather, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(yyyyddd, weather):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == self.year2:
                self.denom = np.mean(self.denomterms)

                # Print out all past results, reb-ased
                for self.pastresult in self.pastresults:
                    yield [self.pastresult[0], func(self.pastresult[1], self.denom)] + list(self.pastresult[1:])

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
                self.denomterms.append(result)
            else:
                # calculate this and tack it on
                yield [year, func(result, self.denom)] + list(yearresult[1:])

def make_runaverage(make_generator, priors, weights, unshift=False):
    """Generate results as an N-year running average;
    priors: list of size N, with the values to use before we get data
    weights: list of size N, with the weights of the years (earliest first)
    """

    # Use the runaverage function to do all the operations
    return make(runaverage, make_generator, priors, weights, unshift)

def runaverage(generator, priors, weights, unshift=False):
    """Generate results as an N-year running average;
    priors: list of size N, with the values to use before we get data (first value not used)
    weights: list of size N, with the weights of the years (earliest first)
    unshift: if true, tack on result at front of result list
    """

    values = list(priors) # Make a copy of the priors list
    totalweight = sum(weights) # Use as weight denominator

    for yearresult in generator:
        # The set of values to average ends with the new value
        values = values[1:] + [yearresult[1]]
        # Calculate weighted average
        smoothed = sum([values[ii] * weights[ii] for ii in range(len(priors))]) / totalweight

        # Produce the new result
        if unshift:
            yield [yearresult[0], smoothed] + list(yearresult[1:])
        else:
            yield (yearresult[0], smoothed)

def make_weighted_average(make_generators, weights):
    """This produces a weighted average of results from *multiple generators*.
    make_generators: list of make_generator functions; all must produce identical years
    weights: list of weight dictionaries ({FIPS: weight})
    len(make_generators) == len(weights)
    """

    def generate(fips, yyyyddd, weather, **kw):
        # Is this county represented in any of the weight dictionaries?
        inany = False
        for weight in weights:
            if fips in weight and weight[fips] > 0:
                inany = True
                break

        if not inany:
            return # Produce no results

        # Construct a list of generators from make_generators
        generators = [make_generators[ii](fips, yyyyddd, weather, **kw) for ii in range(len(make_generators))]

        # Iterate through all generators simultaneously
        for values in generators[0]:
            # Construct a list of the value from each generator
            values = [values] + [generator.next() for generator in generators[1:]]
            # Ensure that year is identical across all
            for ii in range(1, len(generators)):
                assert(values[0][0] == values[ii][0])

            # Construct (year, result) where result is a weighted average using weights
            yield (values[0][0], np.sum([values[ii][1] * weights[ii].get(fips, 0) for ii in range(len(generators))]) /
                   np.sum([weights[ii].get(fips, 0) for ii in range(len(generators))]))

    return generate

def make_product(vars, make_generators):
    """This produces a product of results from *multiple generators*.
    vars: a list of single variables to pass into each generator
    make_generators: list of make_generator functions; all must produce identical years
    len(make_generators) == len(vars)
    """

    def generate(fips, yyyyddd, weather, **kw):
        if fips == effect_bundle.FIPS_COMPLETE:
            # Pass on signal for end
            for make_generator in make_generators:
                make_generator(fips, yyyyddd, weather, **kw).next()
            return

        # Construct a list of generators from make_generators
        generators = [make_generators[ii](fips, yyyyddd, weather[vars[ii]], **kw) for ii in range(len(make_generators))]

        # Iterate through all generators simultaneously
        for values in generators[0]:
            values = [values] + [generator.next() for generator in generators[1:]]
            # Ensure that year is identical across all
            for ii in range(1, len(generators)):
                assert(values[0][0] == values[ii][0])

            # Construct (year, result) where result is a product
            yield (values[0][0], np.product([values[ii][1] for ii in range(len(generators))]))

    return generate
