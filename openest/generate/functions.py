import numpy as np
import latextools, calculation, diagnostic

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

    def init_apply(self):
        self.denomterms = [] # don't copy this across instances!
        self.pastresults = []

    def pushhandler(self, yyyyddd, weather, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(yyyyddd, weather):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == self.year2:
                self.denomterms.append(result)
                self.denom = np.mean(self.denomterms)

                # Print out all past results, re-based
                for pastresult in self.pastresults:
                    yield [pastresult[0], func(pastresult[1], self.denom)] + list(pastresult[1:])

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
                if year >= self.year1:
                    self.denomterms.append(result)
            else:
                diagnostic.record(self.region, year, 'baseline', self.denom)
                # calculate this and tack it on
                yield [year, func(result, self.denom)] + list(yearresult[1:])

class InstaZScore(calculation.CustomFunctionalCalculation):
    """
    Collects up to `baseyear` of values and then uses them to represent all values as a z-score.
    """

    def __init__(self, subcalc, lastyear, units='z-score'):
        super(InstaZScore, self).__init__(subcalc, subcalc.unitses[0], units, True, lastyear)
        self.lastyear = lastyear
        self.mean = None # The mean to subtract off
        self.sdev = None # The sdev to divide by
        self.pastresults = [] # results before lastyear

    def latexhandler(self, equation, lastyear):
        raise NotImplementedError()

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, yyyyddd, weather, lastyear):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(yyyyddd, weather):
            year = yearresult[0]
            result = yearresult[1]

            # Have we collected all the data?
            if year == lastyear or (lastyear is None and self.mean is None):
                self.mean = np.mean(map(lambda mx: mx[1], self.pastresults))
                self.sdev = np.std(map(lambda mx: mx[1], self.pastresults))

                # Print out all past results, now that we have them
                for pastresult in self.pastresults:
                    yield [pastresult[0], (pastresult[1] - self.mean) / self.sdev] + list(pastresult[1:])

            if self.mean is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
            else:
                # calculate this and tack it on
                yield [year, (result - self.mean) / self.sdev] + list(yearresult[1:])

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Z-Score of ' + infos[0]['title']
        description = "Z-scores of %s calculated relative to the years before %d." % (infos[0]['name'], self.lastyear)
        return [dict(name='zscore', title=title, description=description)] + infos

"""
Sum two results
"""
class Sum(calculation.Calculation):
    def __init__(self, subcalcs):
        fullunitses = subcalcs[0].unitses[:]
        for ii in range(1, len(subcalcs)):
            assert subcalcs[0].unitses[0] == subcalcs[ii].unitses[0], "%s <> %s" % (subcalcs[0].unitses[0], subcalcs[ii].unitses[0])
            fullunitses.extend(subcalcs[ii].unitses)
        super(Sum, self).__init__([subcalcs[0].unitses[0]] + fullunitses)

        self.subcalcs = subcalcs

    def latex(self, *args, **kwargs):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        def generate(year, results):
            return np.sum(map(lambda x: x[1] if x is not None else np.nan, results))

        # Prepare the generator from our encapsulated operations
        subapps = [subcalc.apply(region, *args, **kwargs) for subcalc in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=True)

    def column_info(self):
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Sum of previous results'
        description = 'Sum of ' + ', '.join([infos[0]['title'] for infos in infoses])

        fullinfos = []
        for infos in infoses:
            fullinfos.extend(infos)
        return [dict(name='sum', title=title, description=description)] + fullinfos
