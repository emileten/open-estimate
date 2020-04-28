import numpy as np
from . import juliatools, latextools, calculation, diagnostic, arguments, formatting
from .formatting import FormatElement

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

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        scaledesc = FormatElement(self.latexpair[1])
        if lang == 'latex':
            elements.update(latextools.call(self.func,
                                            "Scaling function", value,
                                            self.latexpair[0]))
            elements[self.latexpair[0]] = scaledesc
            return elements
        elif lang == 'julia':
            variable = formatting.get_variable()
            elements.update(latextools.call(self.func,
                                            "Scaling function", value, variable))
            elements[variable] = scaledesc
            return elements

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

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.region_dictionary, arguments.input_unit,
                               arguments.output_unit, arguments.input_reduce.optional(),
                               arguments.latexpair],
                    description="Scale each result by a region-specific value.")

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

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        if lang == 'latex':
            elements.update(latextools.call(self.func,
                                            self.long_description, elements['main']))
        elif lang == 'julia':
            elements.update(juliatools.call(self.func,
                                            self.long_description, elements['main']))
        return elements

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

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.input_unit,
                               arguments.output_unit, arguments.input_change,
                               arguments.description, arguments.description.rename('long_description')],
                    description="Apply an arbitrary function to the results.")

    
class Instabase(calculation.CustomFunctionalCalculation):
    """Re-base the results of make_generator(...) to the values in baseyear
    baseyear is the year to use as the 'denominator'; None for the first year
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    Tacks on the value to the front of the results
    """

    def __init__(self, subcalc, baseyear, func=lambda x, y: x / y, units='portion', skip_on_missing=True, unshift=True):
        super(Instabase, self).__init__(subcalc, subcalc.unitses[0], units, unshift, baseyear, func, skip_on_missing)
        self.unshift = unshift
        self.baseyear = baseyear
        self.denom = None # The value in the baseyear
        self.pastresults = [] # results before baseyear

    def format_handler(self, equation, lang, baseyear, func, skip_on_missing):
        eqvar = formatting.get_variable(equation)
        if lang == 'latex':
            result = latextools.call(func, "Re-basing function", eqvar,
                                     r"\left[%s\right]_{t = %d}" % (eqvar, baseyear))
        elif lang == 'julia':
            result = juliatools.call(func, "Re-basing function", eqvar,
                                     "%s[findfirst(year .== %d)" % (eqvar, baseyear))
        result['main'].dependencies.append(eqvar)
        result[eqvar] = equation

        formatting.add_label('rebased', result)
        return result

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, ds, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
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

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year,
                               arguments.input_reduce.optional(), arguments.output_unit.optional(),
                               arguments.skip_on_missing.optional()],
                    description="Translate all results relative to the baseline year.")

class SpanInstabase(Instabase):
    """Re-base the results of a calculation to the average of values between two years.
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    """
    def __init__(self, subcalc, year1, year2, func=lambda x, y: x / y, units='portion', skip_on_missing=True, unshift=True):
        super(SpanInstabase, self).__init__(subcalc, (year1 + year2) / 2, func, units, skip_on_missing, unshift)
        self.year1 = year1
        self.year2 = year2
        self.denomterms = []

    def format_handler(self, equation, lang, baseyear, func, skip_on_missing):
        eqvar = formatting.get_variable(equation)
        if lang == 'latex':
            result = latextools.call(func, "Re-basing function", eqvar,
                                     r"Average\left[%s\right]_{%d \le t le %d}" % (formatting.get_repstr(eqvar), self.year1, self.year2))
        elif lang == 'julia':
            result = juliatools.call(func, "Re-basing function", eqvar,
                                     "mean(%s[(year .>= %d) & (year .<= %d)])" % (formatting.get_repstr(eqvar), self.year1, self.year2))
        if isinstance(eqvar, str):
            result['main'].dependencies.append(eqvar)
            result[eqvar] = equation
        else:
            # FormatElement
            result['main'].dependencies.extend(eqvar.dependencies)

        formatting.add_label('rebased', result)
        return result

    def init_apply(self):
        self.denomterms = [] # don't copy this across instances!
        self.pastresults = []

    def pushhandler(self, ds, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == self.year2:
                self.denomterms.append(result)
                if not self.deltamethod:
                    self.denom = np.mean(self.denomterms)
                else:
                    self.denom = np.mean(self.denomterms, 0)

                # Print out all past results, re-based
                for pastresult in self.pastresults:
                    diagnostic.record(self.region, pastresult[0], 'baseline', self.denom)
                    if self.unshift:
                        yield [pastresult[0], func(pastresult[1], self.denom)] + list(pastresult[1:])
                    else:
                        yield [pastresult[0], func(pastresult[1], self.denom)]

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
                if year >= self.year1:
                    self.denomterms.append(result)
            else:
                diagnostic.record(self.region, year, 'baseline', self.denom)
                # calculate this and tack it on
                if self.unshift:
                    yield [year, func(result, self.denom)] + list(yearresult[1:])
                else:
                    yield [year, func(result, self.denom)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year.describe("The starting year"),
                               arguments.year.describe("The ending year"),
                               arguments.input_reduce.optional(), arguments.output_unit.optional(),
                               arguments.skip_on_missing.optional()],
                    description="Translate all results relative to a span of baseline years.")

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

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, ds, lastyear):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
            year = yearresult[0]
            result = yearresult[1]

            # Have we collected all the data?
            if year == lastyear or (lastyear is None and self.mean is None):
                self.mean = np.mean([mx[1] for mx in self.pastresults])
                self.sdev = np.std([mx[1] for mx in self.pastresults])

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

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year,
                               arguments.output_unit.optional()],
                    description="Translate all results to z-scores against results up to a given year.")


class Sum(calculation.Calculation):
    """Sum two or more subcalculations

    Parameters
    ----------
    subcalcs : Sequence of ``openest.generate.calculation.Calculation``
    unshift : bool, optional
    """
    def __init__(self, subcalcs, unshift=True):
        fullunitses = subcalcs[0].unitses[:]
        for ii in range(1, len(subcalcs)):
            assert subcalcs[0].unitses[0] == subcalcs[ii].unitses[0], "%s <> %s" % (subcalcs[0].unitses[0], subcalcs[ii].unitses[0])
            fullunitses.extend(subcalcs[ii].unitses)
        if unshift:
            super(Sum, self).__init__([subcalcs[0].unitses[0]] + fullunitses)
        else:
            super(Sum, self).__init__([subcalcs[0].unitses[0]])

        self.unshift = unshift
        self.subcalcs = subcalcs

    def format(self, lang, *args, **kwargs):
        mains = []
        elements = {}
        alldeps = set()
        for subcalc in self.subcalcs:
            elements.update(subcalc.format(lang, *args, **kwargs))
            mains.append(elements['main'])
            alldeps.update(elements['main'].dependencies)
            
        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement(' + '.join([main.repstr for main in mains]), list(alldeps))

        formatting.add_label('sum', elements)
        return elements
        
    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """
        def generate(year, results):
            if not self.deltamethod:
                return np.sum([x[1] if x is not None else np.nan for x in results])
            else:
                return np.sum([x[1] if x is not None else np.nan for x in results], 0)

        # Prepare the generator from our encapsulated operations
        subapps = [subcalc.apply(region, *args, **kwargs) for subcalc in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=self.unshift)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Sum of previous results'
        description = 'Sum of ' + ', '.join([infos[0]['title'] for infos in infoses])

        fullinfos = []
        for infos in infoses:
            fullinfos.extend(infos)
        return [dict(name='sum', title=title, description=description)] + fullinfos

    def enable_deltamethod(self):
        self.deltamethod = True
        for subcalc in self.subcalcs:
            subcalc.enable_deltamethod()

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates
        are supported.
        """
        return Sum([subcalc.partial_derivative(covariate, covarunit) for subcalc in self.subcalcs])

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculationss, arguments.unshift.optional()],
                    description="Sum the results of multiple previous calculations.")


class Prod(calculation.Calculation):
    """Product of two or more subcalculations

    Parameters
    ----------
    subcalcs : Sequence of ``openest.generate.calculation.Calculation``
    unshift : bool, optional
    """

    def __init__(self, subcalcs, unshift=True):
        fullunitses = subcalcs[0].unitses[:]
        for ii in range(1, len(subcalcs)):
            assert subcalcs[0].unitses[0] == subcalcs[ii].unitses[0], "%s <> %s" % (
            subcalcs[0].unitses[0], subcalcs[ii].unitses[0])
            fullunitses.extend(subcalcs[ii].unitses)
        if unshift:
            super().__init__([subcalcs[0].unitses[0]] + fullunitses)
        else:
            super().__init__([subcalcs[0].unitses[0]])

        self.unshift = unshift
        self.subcalcs = subcalcs

    def format(self, lang, *args, **kwargs):
        mains = []
        elements = {}
        alldeps = set()
        for subcalc in self.subcalcs:
            elements.update(subcalc.format(lang, *args, **kwargs))
            mains.append(elements['main'])
            alldeps.update(elements['main'].dependencies)

        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement(' * '.join([main.repstr for main in mains]), list(alldeps))

        formatting.add_label('prod', elements)
        return elements

    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """

        def generate(year, results):
            if not self.deltamethod:
                return np.prod([x[1] if x is not None else np.nan for x in results])
            else:
                return np.prod([x[1] if x is not None else np.nan for x in results], 0)

        # Prepare the generator from our encapsulated operations
        subapps = [subcalc.apply(region, *args, **kwargs) for subcalc in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=self.unshift)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Product of previous results'
        description = 'Product of ' + ', '.join([infos[0]['title'] for infos in infoses])

        fullinfos = []
        for infos in infoses:
            fullinfos.extend(infos)
        return [dict(name='prod', title=title, description=description)] + fullinfos

    def enable_deltamethod(self):
        self.deltamethod = True
        for subcalc in self.subcalcs:
            subcalc.enable_deltamethod()

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates
        are supported.
        """
        return Sum([subcalc.partial_derivative(covariate, covarunit) for subcalc in self.subcalcs])

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculationss, arguments.unshift.optional()],
                    description="Product of results from multiple previous calculations.")


"""
ConstantScale
"""
class ConstantScale(calculation.Calculation):
    def __init__(self, subcalc, coeff):
        super(ConstantScale, self).__init__([subcalc.unitses[0]] + subcalc.unitses)
        self.subcalc = subcalc
        self.coeff = coeff

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
            
        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement('%f * (%s)' % (self.coeff, elements['main'].repstr), elements['main'].dependencies)

        return elements
        
    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return self.coeff * result

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Previous result multiplied by %f' % self.coeff
        description = 'Previous result multiplied by %f' % self.coeff

        return [dict(name='constscale', title=title, description=description)] + infos

    def partial_derivative(self, covariate, covarunit):
        return Sum([subcalc.partial_derivative(covariate, covarunit) for subcalc in self.subcalcs],
                   unshift=self.unshift)

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.coefficient],
                    description="Multiply the result by a constant factor.")

class Positive(calculation.Calculation):
    """
    Return 0 if subcalc is less than 0
    """
    def __init__(self, subcalc):
        super(Positive, self).__init__([subcalc.unitses[0]] + subcalc.unitses)
        self.subcalc = subcalc

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return result if result > 0 else 0

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Positive-only form of ' + infos[0]['title']
        description = 'The value of ' + infos[0]['title'] + ', if positive and otherwise 0.'

        return [dict(name='positive', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation],
                    description="Return the maximum of a previous result or 0.")

class Exponentiate(calculation.Calculation):
    def __init__(self, subcalc, errorvar):
        assert subcalc.unitses[0][:3] == 'log'
        super(Exponentiate, self).__init__([subcalc.unitses[0][3:].strip()] + subcalc.unitses)
        self.subcalc = subcalc
        self.errorvar = errorvar

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        if lang == 'latex':
            elements.update({'main': FormatElement(r"\exp{%s + %f/2}" % (elements['main'].repstr, self.errorvar))})
        elif lang == 'julia':
            elements.update({'main': FormatElement(r"exp(%s + %f/2)" % (elements['main'].repstr, self.errorvar))})
        return elements
            
    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return np.exp(result + 0.5*self.errorvar)

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'exp(' + infos[0]['title'] + ')'
        description = 'Exponentiation of ' + infos[0]['title']

        return [dict(name='exp', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.variance.rename('errorvar')],
                    description="Return the the exponentiation of a previous result.")

class AuxillaryResult(calculation.Calculation):
    """
    Produce an additional output, but then pass the main result on.
    """
    def __init__(self, subcalc_main, subcalc_aux, auxname):
        super(AuxillaryResult, self).__init__([subcalc_main.unitses[0], subcalc_aux.unitses[0]] + subcalc_main.unitses[1:])
        self.subcalc_main = subcalc_main
        self.subcalc_aux = subcalc_aux
        self.auxname = auxname

    def format(self, lang, *args, **kwargs):
        beforeauxlen = len(formatting.format_labels)
        auxres = self.subcalc_aux.format(lang, *args, **kwargs)
        formatting.format_labels = formatting.format_labels[:beforeauxlen] # drop any new ones added
        formatting.add_label(self.auxname, auxres)
        return self.subcalc_main.format(lang, *args, **kwargs)

    def apply(self, region, *args, **kwargs):
        subapp_main = self.subcalc_main.apply(region, *args, **kwargs)
        subapp_aux = self.subcalc_aux.apply(region, *args, **kwargs)
        return AuxillaryResultApplication(region, subapp_main, subapp_aux)

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates are supported.
        """
        return AuxillaryResult(self.subcalc_main.partial_derivative(covariate, covarunit),
                               self.subcalc_aux.partial_derivative(covariate, covarunit), self.auxname)
        
    def column_info(self):
        infos_main = self.subcalc_main.column_info()
        infos_aux = self.subcalc_aux.column_info()
        infos_aux[0]['name'] = self.auxname

        return [infos_main[0], infos_aux[0]] + infos_main[1:]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.calculation.describe("An auxillary calculation, placed behind the main calculation."), arguments.label],
                    description="Add an additional result to the columns.")

class AuxillaryResultApplication(calculation.Application):
    """
    Perform both main and auxillary calculation, and order as main[0], aux, main[1:]
    """
    def __init__(self, region, subapp_main, subapp_aux):
        super(AuxillaryResultApplication, self).__init__(region)
        self.subapp_main = subapp_main
        self.subapp_aux = subapp_aux

    def push(self, ds):
        for yearresult in self.subapp_main.push(ds):
            for yearresult_aux in self.subapp_aux.push(ds):
                next # Just take the last one
            yield list(yearresult[0:2]) + [yearresult_aux[1]] + list(yearresult[2:])

    def done(self):
        self.subapp_aux.done()
        return self.subapp_main.done()

class KeepOnly(calculation.Calculation):
    """
    Keep only a subset of the calculation results, with given names.
    """
    def __init__(self, subcalc, names):
        self.subcalc = subcalc
        self.iskept = [info['name'] in names for info in subcalc.column_info()]
        super(KeepOnly, self).__init__(self.keeplist(subcalc.unitses))

    def keeplist(self, lst):
        assert len(lst) == len(self.iskept), "Given %d <> %d when choosing which to keep." % (len(lst), len(self.iskept))
        return [lst[ii] for ii in range(len(lst)) if self.iskept[ii]]
        
    def format(self, lang, *args, **kwargs):
        return self.subcalc.format(lang, *args, **kwargs)
            
    def apply(self, region, *args, **kwargs):
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return KeepOnlyApplication(region, subapp, self.keeplist)

    def column_info(self):
        return self.keeplist(self.subcalc.column_info())

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.column_names],
                    description="Only keep a subset of the preliminary calculations.")

class KeepOnlyApplication(calculation.Application):
    def __init__(self, region, subapp, keeplist):
        super(KeepOnlyApplication, self).__init__(region)
        self.subapp = subapp
        self.keeplist = keeplist

    def push(self, ds):
        for yearresult in self.subapp.push(ds):
            yield [yearresult[0]] + self.keeplist(yearresult[1:])

    def done(self):
        return self.subapp.done()

class Clip(calculation.Calculation):
    """Clip the values in a subcalculation.

    Given a (min, max) interval of values in a subcalculation, values outside
    of this interval are clipped to the interval edges. For example, with the
    intervals ``[0.0, 1.0]``, values less than 0 become 0 and values greater
    than 1 become 1.

    Parameters
    ----------
    subcalc : openest.generate.calculation.Application
    subcalc_min : float
    subcalc_max : float
    """
    def __init__(self, subcalc, subcalc_min, subcalc_max):
        super().__init__([subcalc.unitses[0]] + subcalc.unitses)
        self.subcalc = subcalc
        self.min = float(subcalc_min)
        self.max = float(subcalc_max)

    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations (`self.subcalc`)

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """
        def generate(year, result):
            # This is where the actual clipping happens
            return max(self.min, min(result, self.max))

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infos = self.subcalc.column_info()
        title_str = str(infos[0]['title'])
        title = f'Clipped form of {title_str}'
        description = f'The value of {title_str}, clipped to the interval [{self.min}, {self.max}].'
        return [dict(name='clip', title=title, description=description)] + infos

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.numeric.rename('min'), arguments.numeric.rename('max')],
                    description="Return the clipped values of a previous result.")
