from model import Model
import numpy as np

class UnivariateModel(Model):

    def __init__(self, xx_is_categorical=False, xx=None, scaled=True):
        super(UnivariateModel, self).__init__(scaled)
        
        self.xx_is_categorical = xx_is_categorical
        if xx is None:
            self.xx = []
            self.xx_text = []
        elif xx_is_categorical:
            self.xx = range(len(xx))
            self.xx_text = map(str, xx)
        else:
            self.xx = xx
            self.xx_text = map(str, xx)

    def get_xx(self):
        if self.xx_is_categorical:
            return self.xx_text
        else:
            return self.xx

    # Return a new model with only the given values of x
    def filter_x(self, xx):
        return None # override this!

    # Only for non-categorical models
    def interpolate_x(self, xx):
        return None # override this!

    # Only for categorical models
    def recategorize_x(self, oldxx, newxx):
        return None # override this!

    @staticmethod
    def intersect_get_x(xx_is_categorical, one_xx, two_xx):
        if xx_is_categorical:
            if len(one_xx) == len(two_xx) and all([one_xx[ii] == two_xx[ii] for ii in range(len(one_xx))]):
                xx = one_xx
            else:
                xx = set(one_xx).intersection(set(two_xx))
        else:
            xx_min = max(min(one_xx), min(two_xx))
            xx_max = min(max(one_xx), max(two_xx))

            xx_one = np.array(one_xx)
            xx_two = np.array(two_xx)
            xx_one = xx_one[np.logical_and(xx_one >= xx_min, xx_one <= xx_max)]
            xx_two = xx_two[np.logical_and(xx_two >= xx_min, xx_two <= xx_max)]

            if np.array_equal(xx_one, xx_two):
                xx = xx_one
            else:
                xx = sorted(list(set(list(xx_one) + list(xx_two))))

        return xx

    @staticmethod
    def intersect_get_model(model, xx):
        if model.xx_is_categorical:
            if len(model.xx_text) == len(xx) and all([model.xx_text[ii] == xx[ii] for ii in range(len(xx))]):
                pass
            else:
                model = model.filter_x(xx)
        else:
            if np.array_equal(model.xx, xx):
                pass
            else:
                model = model.interpolate_x(xx)

        return model

    @staticmethod
    def intersect_x(one, two):
        if one.xx_is_categorical:
            xx = UnivariateModel.intersect_get_x(True, one.xx_text, two.xx_text)
            one = UnivariateModel.intersect_get_model(one, xx)
            two = UnivariateModel.intersect_get_model(two, xx)
        else:
            xx = UnivariateModel.intersect_get_x(False, one.xx, two.xx)
            one = UnivariateModel.intersect_get_model(one, xx)
            two = UnivariateModel.intersect_get_model(two, xx)

        return (one, two, xx)
