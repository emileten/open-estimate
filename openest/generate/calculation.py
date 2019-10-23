"""
Abstract and concrete classes to delegate data and iterate calculations.
"""

import copy
import numpy as np
import xarray as xr


class Calculation(object):
    """ABC for calculations used in an Application

    Parameters
    ----------
    unitses : sequence of str
        Post-calculation units.

    Attributes
    ----------
    deltamethod : bool
        Does this calculation use The Deltamethod.
    """

    def __init__(self, unitses):
        self.unitses = unitses
        self.deltamethod = False

    def format(self, lang, *args, **kwargs):
        """Returns a dictionary of FormatElements.
        Only keys in the tree of dependencies will be output.
        """
        raise NotImplementedError()

    def test(self):
        return self.apply('test')

    def apply(self, region, *args, **kwargs):
        raise NotImplementedError()

    def cleanup(self):
        pass

    def column_info(self):
        """Returns an array of {'name', 'title', 'description'}.
        """
        raise NotImplementedError()

    def enable_deltamethod(self):
        """Enable deltamethod calculation?

        When applied, yield will contain arrays of coefficient multiplicands
        as a vector the length of the CSVV coefficients.
        """
        self.deltamethod = True
    
    @staticmethod
    def describe():
        """Describe the calculation

        Returns
        -------
        out : dict
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
        raise NotImplementedError()


class FunctionalCalculation(Calculation):
    """
    Calculation that calls a handler when it's applied

    Parameters
    ----------
    subcalc : openest.generate.calculation.Calculation-like
        Sub-calculation object. `subcalc.uniteses[0]` must equal
        `from_units`.
    from_units : str
        Pre-calculation units.
    to_units : str
        Post-calculation units.
    unshift : bool
    handler_args :
        Additional arguments passed on to `self.handler()` when applying
        (via `self.apply()`) the calculation.
    handler_kw :
        Additional keyword arguments passed on to `self.handler()` when
        applying (via `self.apply()`) the calculation.
    """

    def __init__(self, subcalc, from_units, to_units, unshift, *handler_args, **handler_kw):
        if unshift:
            super(FunctionalCalculation, self).__init__([to_units] + subcalc.unitses)
        else:
            super(FunctionalCalculation, self).__init__([to_units] + subcalc.unitses[1:])

        assert(subcalc.unitses[0] == from_units)
        self.subcalc = subcalc
        self.handler_args = handler_args
        self.handler_kw = handler_kw

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        handler_elements = self.format_handler(elements['main'], lang, *self.handler_args, **self.handler_kw)
        elements = copy.copy(elements)  # don't update possibly saved elements
        elements.update(handler_elements)

        return elements

    def format_handler(self, substr, lang, *handler_args, **handler_kw):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        callpass = lambda year, result: self.handler(year, result, *self.handler_args, **self.handler_kw)
        return ApplicationPassCall(region, subapp, callpass)

    def handler(self, year, result, *handler_args, **handler_kw):
        raise NotImplementedError()

    def cleanup(self):
        """Pass cleanup signal to subcalculations
        """
        # Pass on signal for end
        print "completing make"
        self.subcalc.cleanup()

    def enable_deltamethod(self):
        self.deltamethod = True
        self.subcalc.enable_deltamethod()


class Application(object):
    """
    ABC for objects connecting region data to Calculation-likes

    Parameters
    ----------
    region : str
        Region to apply to.
    """
    def __init__(self, region):
        self.region = region

    def push(self, ds):
        """
        Yields values for years from a dataset (yyyy, value, ...)

        Parameters
        ----------
        ds : xarray.Dataset
            Input data values to apply calculations to for subset in target
            region.

        Yields
        ------
        year : int or float
        value : int or float
        ... :
            Optional additional values.
        """
        raise NotImplementedError()

    def done(self):
        return []


class CustomFunctionalCalculation(FunctionalCalculation, Application):
    """Calculation that creates a copy of itself for an application

    Parameters
    ----------
    subcalc : openest.generate.calculation.Calculation-like
        Sub-calculation object. `subcalc.uniteses[0]` must equal
        `from_units`.
    from_units : str
        Pre-calculation units.
    to_units : str
        Post-calculation units.
    unshift : bool
    handler_args :
        Additional arguments passed on to `self.handler()` when applying
        (via `self.apply()`) the calculation.
    handler_kw :
        Additional keyword arguments passed on to `self.handler()` when
        applying (via `self.apply()`) the calculation.
    """
    def __init__(self, subcalc, from_units, to_units, unshift, *handler_args, **handler_kw):
        super(CustomFunctionalCalculation, self).__init__(subcalc, from_units, to_units, unshift, *handler_args,
                                                          **handler_kw)
        self.subapp = None
        self.region = None

    def apply(self, region, *args, **kwargs):
        """
        Get generator to apply all subcalculations

        Parameters
        ----------
        region : str
            Target region to apply calculations. Passed to
            ``self.subcalc.apply()``.
        args :
            Passed to ``self.subcalc.apply()``.
        kwargs :
            Passed to ``self.subcalc.apply()``.

        Returns
        -------
        app : CustomFunctionalCalculation
            Generator (copy of self) with subcalculations primed with `region`,
            `args`, and `kwargs`.

        See Also
        --------
        CustomFunctionalCalculation.push : yield yearly results from ``self.pushhandler``.
        """
        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        allargs = list(self.handler_args) + list(args)
        allkwargs = self.handler_kw.copy()
        allkwargs.update(kwargs)

        app = copy.copy(self)
        app.subapp = subapp
        app.region = region
        app.init_apply()

        return app

    def init_apply(self):
        pass

    def push(self, ds):
        """Push to yield yearly results from ``self.pushhandler``.

        Push `ds`, ``self.handler_args``, ``self.handler_kw`` to
        ``self.pushhandler``.

        You'd likely want to run this after setting up the generators with
        ``self.apply``.

        Parameters
        ----------
        ds : xarray.Dataset
            Variables which are later subset to a region and year.

        Yields
        -------
        Yearly result from ``self.pushhandler``.

        See Also
        --------
        CustomFunctionalCalculation.apply : Prime ``self.pushhandler`` generator with region and arguments.
        """
        for yearresult in self.pushhandler(ds, *self.handler_args, **self.handler_kw):
            yield yearresult

    def pushhandler(self, ds, *allargs, **allkwargs):
        """
        Generator parsing `ds` to output yearly results.

        Parameters
        ----------
        ds : xarray.Dataset
            Variables to be subset to a region and parsed by year.
        allargs
        allkwargs

        Yields
        -------
        yearresult
        """
        raise NotImplementedError()

    def done(self):
        """
        Prime and return ``self.donehandler`` generator.

        ``self.donehandler`` is primed with ``self.handler_args`` and
        ``self.handler_kw``.

        Returns
        -------
        generator
        """
        return self.donehandler(*self.handler_args, **self.handler_kw)

    def donehandler(self, *allargs, **allkwargs):
        return []


class ApplicationEach(Application):
    """
    Pass every set of values to the calculation for a value.

    Parameters
    ----------
    region : str
        Target region for this application.
    func : callable
        Callable taking a region, time, dataarray and optional positional
        and keyword arguments whenever ``self.push`` is called.
    finishfunc : generator
        Iterated for yearly results without argument whenever
        ``self.done`` is called.
    args :
        Passed to `func` whenever ``self.push`` is called.
    kwargs :
        Passed to `func` whenever ``self.push`` is called.
    """
    def __init__(self, region, func, finishfunc=lambda: [], *args, **kwargs):
        super(ApplicationEach, self).__init__(region)
        self.func = func
        self.finishfunc = finishfunc
        self.args = args
        self.kwargs = kwargs

    def push(self, ds):
        """
        Generates values from ``self.func`` using `ds`.

        Passes ``self.region``, ``self.args``, ``self.kwargs``.

        Parameters
        ----------
        ds : xarray.Dataset
            Input data values to apply calculations to for subset in target
            region. Must have "time" attribute.

        Yields
        ------
        values :
            Returned or yielded from calling ``self.func`` over ``ds.time``.
        """
        for ii in range(len(ds.time)):
            for values in self.func(self.region, ds.time[ii], ds.isel(time=ii), *self.args, **self.kwargs):
                yield values

    def done(self):
        """
        Prime and return ``self.donehandler`` generator.

        ``self.donehandler`` is primed with ``self.handler_args`` and
        ``self.handler_kw``.

        Returns
        -------
        generator
        """
        for yearresult in self.finishfunc():
            yield yearresult


class ApplicationPassCall(Application):
    """Apply a non-enumerator to all elements of a function.

    If unshift, tack on the result to the front of a sequence of results.
    Calls func with each year and value; returns the newly computed value.

    Parameters
    ----------
    region : str
        Target region we apply our calculations to.
    subapp : Application-like or sequence of Application-like
        We use the element's `push` method whenever calling ``self.push``,
        returning a (year, value) or an iterator giving (year, value).
    handler :  callable
        Returns (year, value) or value when passed a year, value,
        `handler_args`, and `handler_kw`.
    handler_args :
        Passed to `handler` whenever ``self.push`` is called.
    handler_kw :
        Passed to `handler` whenever ``self.push`` is called. If contains
        'unshift' key, the corresponding value is assigned to
        ``self.unshift``. Otherwise ``self.unshift`` becomes False.
    """
    def __init__(self, region, subapp, handler, *handler_args, **handler_kw):
        super(ApplicationPassCall, self).__init__(region)
        self.subapp = subapp
        self.handler = handler

        if 'unshift' in handler_kw:
            self.unshift = handler_kw['unshift']
            del handler_kw['unshift']
        else:
            self.unshift = False

        self.handler_args = handler_args
        self.handler_kw = handler_kw

    def push(self, ds):
        """
        Parameters
        ----------
        ds : xarray.Dataset
            Input data values to apply calculations to. Only passed as arg
            to `push` method of elements in ``self.subapp``.

        Yields
        ------
        Iterable each with (year, value, ...).
        """
        if isinstance(self.subapp, list):
            iterators = [subapp.push(ds) for subapp in self.subapp]
            while True:
                yearresults = []
                # Call next on every iterator
                for iterator in iterators:
                    try:
                        yearresult = iterator.next()
                    except StopIteration:
                        yearresult = None
                    yearresults.append(yearresult)

                if yearresults[0] is None:
                    # Ignore the result
                    return
                else:
                    year = yearresults[0][0] if yearresults[0] is not None else None
                    anyresults = False
                    for yearresult in yearresults:
                        if yearresult is not None:
                            assert yearresult[0] == year, "%s <> %s" % (str(yearresult[0]), str(year))
                            anyresults = True
                    if not anyresults:
                        return  # No results

                newresult = self.handler(year, yearresults, *self.handler_args, **self.handler_kw)
                if isinstance(newresult, tuple):
                    yield tuple
                else:
                    if self.unshift:
                        fullresult = [year, newresult]
                        for ii in range(len(yearresults)):
                            fullresult.extend(yearresults[ii][1:] if yearresults[ii] is not None else [np.nan])
                        yield fullresult
                    else:
                        yield (year, newresult)
        else:
            for yearresult in self.subapp.push(ds):
                year = yearresult[0]
                # Call handler to get a new value
                newresult = self.handler(year, yearresult[1], *self.handler_args, **self.handler_kw)
                if isinstance(newresult, tuple):
                    yield tuple
                else:
                    # Construct a new year, value result
                    if self.unshift:
                        yield [year, newresult] + list(yearresult[1:])
                    else:
                        yield (year, newresult)


class ApplicationByChunks(Application):
    """

    Parameters
    ----------
    region : str
        Target region to apply calculation to.
    """
    def __init__(self, region):
        super(ApplicationByChunks, self).__init__(region)
        self.saved_ds = None

    def push(self, ds):

        if self.saved_ds is None:
            return self.push_saved(ds)

        self.saved_ds = xr.concat((self.saved_ds, ds), dim='time')
        return self.push_saved(self.saved_ds)

    def push_saved(self, ds):
        """
        Returns an interator of (yyyy, value, ...).
        Removes used daily values from saved.
        """
        raise NotImplementedError()


class ApplicationByYear(ApplicationByChunks):
    def __init__(self, region, func, *args, **kwargs):
        super(ApplicationByYear, self).__init__(region)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def push_saved(self, ds):
        """
        Returns an interator of (yyyy, value, ...).
        Removes used daily values from saved.
        """
        if len(ds.time) in [365, 366]:
            year = ds.attrs.get('year')
            if year is None:
                year = ds['time.year'].values[0]
                
            for values in self.func(self.region, year, ds, *self.args, **self.kwargs):
                yield values
            return

        print "Seeing an unexpected %d values." % len(ds.time)
        for year, yeards in ds.groupby('time.year'):
            if len(yeards.time) < 365:
                self.saved_ds = yeards
                return

            for values in self.func(self.region, year, yeards, *self.args, **self.kwargs):
                yield values

        self.saved_ds = None


class ApplicationByIrregular(Application):
    def __init__(self, region, func, *args, **kwargs):
        super(ApplicationByIrregular, self).__init__(region)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def push(self, ds):
        year = ds.attrs.get('year')
        if year is None:
            year = ds['time.year'].values[0]

        for values in self.func(self.region, year, ds, *self.args, **self.kwargs):
            yield values
