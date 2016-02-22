import copy
import weather

class Calculation(object):
    def latex(self, *args, **kwargs):
        raise NotImplementedError()

    def apply(self, region, *args, **kwargs):
        raise NotImplementedError()

    def cleanup(self):
        pass

class FunctionalCalculation(Calculation):
    def __init__(self, subcalc, *handler_args, **handler_kw):
        self.subcalc = subcalc
        self.handler_args = handler_args
        self.handler_kw = handler_kw

    def latex(self, *args, **kwargs):
        for (equation, latex) in self.subcalc.latex(*args, **kwargs):
            if equation == "Equation":
                for (equation2, latex2) in self.latexhandler(latex, *self.handler_args, **self.handler_kw):
                    yield (equation2, latex2)
            else:
                yield (equation, latex)

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
        self.pushhandler(yyyyddd, weather, *allargs, **allkwargs)

    def pushhandler(self, yyyyddd, weather, *allargs, **allkwargs):
        raise NotImplementedError()

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
            newresult = handler(year, result, *self.handler_args, **self.handler_kw)
            if isinstance(newresult, tuple):
                yield tuple
            else:
                # Construct a new year, value result
                if unshift:
                    yield [year, newresult] + yearresult[1:]
                else:
                    yield (year, newresult)

class ApplicationByChunks(Application):
    def __init__(self, region):
        super(ApplicationByChunks, self).__init__(region)
        self.saved_yyyyddd = []
        self.saved_weather = []

    def push(self, yyyyddd, weather):
        self.saved_yyyyddd.extend(yyyyddd)
        self.saved_weather.extend(weather)
        return self.push_saved(self.saved_yyyyddd, self.saved_weather)

    def push_saved(self, yyyyddd, weather):
        """
        Returns an interator of (yyyy, value, ...).
        Removes used daily weather from saved.
        """

        raise NotImplementedError()

class ApplicationByYear(ApplicationByChunks):
    def __init__(self, region, func, *args, **kwargs):
        super(ApplicationByYear, self).__init__(region)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def push_saved(self, yyyyddd, weather):
        """
        Returns an interator of (yyyy, value, ...).
        """

        if len(yyyyddd) < 365:
            return

        for year, weather in weather.yearly_daily_ncdf(yyyyddd, weather):
            if len(weather) < 365:
                self.saved_yyyyddd = year * 1000 + np.arange(len(weather)) + 1
                self.saved_weather = weather
                return

            for values in func(region, year, weather, *self.args, **self.kwargs):
                yield values
