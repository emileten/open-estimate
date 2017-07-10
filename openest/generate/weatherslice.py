import copy
import numpy as np

MIN_REGIONS = 20000

class WeatherSlice(object):
    """
        WeatherSlice have two public attributes: times and weathers
        times should be a numpy array with a single dimension.  Let it have length T.
        weathers should be a numpy array of size T [x REGIONS] [x K]
          the regions dimension should be excluded iff this is for a single region
          the last dimension is optional, if more than one value is returned for each time.
    """
    def __init__(self, times, weathers, manyregion=True):
        self.times = np.array(times)
        T = len(times)

        if manyregion:
            if len(weathers.shape) == 1:
                weathers = np.expand_dims(weathers, axis=0)

            assert weathers.shape[1] > MIN_REGIONS, "The second dimension does not have enough regions: %d < %d" % (weathers.shape[1], MIN_REGIONS)
        else:
            assert len(weathers.shape) < 2, "Without the region dimension, WeatherSlice may only have 1 or 2 dimensions."
            
        assert weathers.shape[0] == T, "The first dimension does not match the times: %d <> %d" % (weathers.shape[0], T)

        self.weathers = weathers
        self.manyregion = manyregion

    def select_region(self, ii):
        child = copy.copy(self)
        if len(self.weathers.shape) == 3:
            child.weathers = self.weathers[:, ii, :]
        else:
            child.weathers = self.weathers[:, ii]
        child.manyregion = False

        return child

class DailyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        if len(weathers.shape) == 2 and weathers.shape[1] == len(times): # Wrong order
            weathers = np.transpose(weathers)

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
    
    @staticmethod
    def convert(weatherslice):
        """Converts other slices to yearly values, taking the mean by year"""
        
        if isinstance(weatherslice, YearlyWeatherSlice):
            return weatherslice
        
        origyears = np.array(weatherslice.get_years())
        years, indexes = np.unique(origyears, return_index=True)
        if len(years) > 1:
            years = origyears[np.sort(indexes)] # make sure in order

        weather_byyear = []
        for year in years:
            summed = np.mean(weatherslice.weathers[origyears == year], axis=0)
            summed = np.expand_dims(summed, axis=0)
            weather_byyear.append(summed)

        return YearlyWeatherSlice(years, np.concatenate(weather_byyear, axis=0))

class TriMonthlyWeatherSlice(WeatherSlice):
    def __init__(self, times, weathers):
        super(TriMonthlyWeatherSlice, self).__init__(times, weathers)

class ForecastMonthlyWeatherSlice(TriMonthlyWeatherSlice):
    def __init__(self, month, ahead, weathers):
        super(ForecastMonthlyWeatherSlice, self).__init__([month + ahead], weathers)

        self.month = month
        self.ahead = ahead
