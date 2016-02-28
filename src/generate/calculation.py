import copy
import weather

class Calculation(object):
    def __init__(self, unitses):
        self.unitses = unitses

    def latex(self, *args, **kwargs):
        raise NotImplementedError()

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
    def __init__(self, subcalc, from_units, to_units, *handler_args, **handler_kw):
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

    def push(self, yyyyddd, weather):
        """
        Returns an interator of (yyyy, value, ...).
        """
        raise NotImplementedError()

    def done(self):
        pass

class CustomFunctionalCalculation(FunctionalCalculation, Application):
    def __init__(self, subcalc, *handler_args, **handler_kw):
        super(CustomFunctionalCalculation, self).__init__(subcalc, *handler_args, **handler_kw)
        self.subapp = None

    def apply(self, region, *args, **kwargs):
        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        allargs = list(self.handler_args) + list(args)
        allkwargs = self.handler_kw.copy()
        allkwargs.update(kwargs)

        app = self.__class__(self.subcalc, *allargs, **allkwargs)
        app.subapp = subapp

        return app

    def push(self, yyyyddd, weather):
        return self.pushhandler(yyyyddd, weather, *self.handler_args, **self.handler_kw)

    def pushhandler(self, yyyyddd, weather, *allargs, **allkwargs):
        raise NotImplementedError()

    def done(self):
        return self.donehandler(*self.handler_args, **self.handler_kw)

    def donehandler(self, *allargs, **allkwargs):
        pass

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

    def push(self, yyyyddd, weather):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for (year, result) in self.subapp.push(yyyyddd, weather):
            # Call handler to get a new value
            newresult = self.handler(year, result, *self.handler_args, **self.handler_kw)
            if isinstance(newresult, tuple):
                yield tuple
            else:
                # Construct a new year, value result
                if self.unshift:
                    yield [year, newresult] + yearresult[1:]
                else:
                    yield (year, newresult)

class ApplicationByChunks(Application):
    def __init__(self, region):
        super(ApplicationByChunks, self).__init__(region)
        self.saved_yyyyddd = []
        self.saved_values = []

    def push(self, yyyyddd, values):
        self.saved_yyyyddd.extend(yyyyddd)
        self.saved_values.extend(values)
        return self.push_saved(self.saved_yyyyddd, self.saved_values)

    def push_saved(self, yyyyddd, values):
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

    def push_saved(self, yyyyddd, allvalues):
        """
        Returns an interator of (yyyy, value, ...).
        Removes used daily values from saved.
        """
        for year, values in weather.yearly_daily_ncdf(yyyyddd, allvalues):
            if len(values) < 365:
                self.saved_yyyyddd = year * 1000 + np.arange(len(values)) + 1
                self.saved_values = values
                return

            for values in self.func(self.region, year, values, *self.args, **self.kwargs):
                yield values

        self.saved_yyyyddd = []
        self.saved_values = []
