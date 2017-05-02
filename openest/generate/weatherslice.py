import numpy as np

MIN_REGIONS = 20000

class WeatherSlice(object):
    """
        times should be a numpy array with a single dimension.  Let it have length T.
        weather should be a numpy array of size T x REGIONS [x K]
          the last dimension is optional, if more than one value is returned for each time.
    """
    def __init__(self, times, weathers):
        self.times = np.array(times)
        T = len(times)

        if len(weathers.shape) == 1:
            weathers = np.expand_dims(weathers, axis=0)

        assert weathers.shape[0] == T, "The first dimension does not match the times: %d <> %d" % (weathers.shape[0], T)
        assert weathers.shape[1] > MIN_REGIONS, "The second dimension does not have enough regions: %d < %d" % (weathers.shape[1], MIN_REGIONS)

        self.weathers = weathers

class DailyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        super(DailyWeatherSlice, self).__init__(times, weathers)

        assert np.all(self.times > 1000000)

    def get_years(self):
        return map(int, self.times // 1000)

class MonthlyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        super(MonthlyWeatherSlice, self).__init__(times, weathers)

class YearlyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        super(YearlyWeatherSlice, self).__init__(times, weathers)

        assert np.all(self.times > 1800)
        assert np.all(self.times < 2200)

    def get_years(self):
        return self.times

class TriMonthlyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        super(TriMonthlyWeatherSlice, self).__init__(times, weathers)

class ForecastMonthlyWeatherSlice(TriMonthlyWeatherSlice):
    def __init__(self, month, ahead, weathers):
        super(ForecastMonthlyWeatherSlice, self).__init__([month + ahead], weathers)

        self.month = month
        self.ahead = ahead
