import copy
import weathertools
import numpy as np

class Calculation(object):
    def __init__(self, unitses):
        self.unitses = unitses

    def latex(self, *args, **kwargs):
        raise NotImplementedError()

    def test(self):
        return self.apply('test')

    def apply(self, region, *args, **kwargs):
        raise NotImplementedError()

    def cleanup(self):
        pass

    def column_info(self):
        """
        Returns an array of dictionaries, with 'name', 'title', and 'description'.
        """
        raise NotImplementedError()

class FunctionalCalculation(Calculation):
    """Calculation that calls a handler when it's applied."""
    def __init__(self, subcalc, from_units, to_units, unshift, *handler_args, **handler_kw):
        if unshift:
            super(FunctionalCalculation, self).__init__([to_units] + subcalc.unitses)
        else:
            super(FunctionalCalculation, self).__init__([to_units] + subcalc.unitses[1:])

        assert(subcalc.unitses[0] == from_units)
        self.subcalc = subcalc
        self.handler_args = handler_args
        self.handler_kw = handler_kw

    def latex(self, *args, **kwargs):
        for (key, value, units) in self.subcalc.latex(*args, **kwargs):
            if key == "Equation":
                for tuple2 in self.latexhandler(value, *self.handler_args, **self.handler_kw):
                    yield tuple2
            else:
                yield (key, value, units)

    def latexhandler(self, latex, *handler_args, **handler_kw):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        callpass = lambda year, result: self.handler(year, result, *self.handler_args, **self.handler_kw)
        return ApplicationPassCall(region, subapp, callpass)

    def handler(self, year, result, *handler_args, **handler_kw):
        raise NotImplementedError()

    def cleanup(self):
        # Pass on signal for end
        print "completing make"
        self.subcalc.cleanup()

class Application(object):
    def __init__(self, region):
        self.region = region

    def push(self, ds):
        """
        Returns an interator of (yyyy, value, ...).
        """
        raise NotImplementedError()

    def done(self):
        return []

class CustomFunctionalCalculation(FunctionalCalculation, Application):
    """Calculation that creates a copy of itself for an application."""
    def __init__(self, subcalc, from_units, to_units, unshift, *handler_args, **handler_kw):
        super(CustomFunctionalCalculation, self).__init__(subcalc, from_units, to_units, unshift, *handler_args, **handler_kw)
        self.subapp = None
        self.region = None

    def apply(self, region, *args, **kwargs):
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
        for yearresult in self.pushhandler(ds, *self.handler_args, **self.handler_kw):
            yield yearresult

    def pushhandler(self, ds, *allargs, **allkwargs):
        raise NotImplementedError()

    def done(self):
        return self.donehandler(*self.handler_args, **self.handler_kw)

    def donehandler(self, *allargs, **allkwargs):
        return []

class ApplicationEach(Application):
    """
    Pass every set of values to the calculation for a value.
    """
    def __init__(self, region, func, finishfunc=lambda: [], *args, **kwargs):
        super(ApplicationEach, self).__init__(region)
        self.func = func
        self.finishfunc = finishfunc
        self.args = args
        self.kwargs = kwargs

    def push(self, ds):
        for ii in range(len(ds.time)):
            for values in self.func(self.region, ds.time[ii], ds.isel(time=ii), *self.args, **self.kwargs):
                yield values

    def done(self):
        for yearresult in self.finishfunc():
            yield yearresult

class ApplicationPassCall(Application):
    """Apply a non-enumerator to all elements of a function.
    if unshift, tack on the result to the front of a sequence of results.
    Calls func with each year and value; returns the newly computed value
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
        Returns an interator of (yyyy, value, ...).
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
                        return # No results

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
            
        for year, yeards in ds.groupby('time.year'):
            if len(yeards.time) < 365:
                self.saved_ds = yeards
                return

            for values in self.func(self.region, year, yeards, *self.args, **self.kwargs):
                yield values

        self.saved_ds = None
