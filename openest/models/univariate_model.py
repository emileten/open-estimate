from .model import Model
import numpy as np

class UnivariateModel(Model):

    def __init__(self, xx_is_categorical=False, xx=None, scaled=True):
        super(UnivariateModel, self).__init__(scaled)
        
        self.xx_is_categorical = xx_is_categorical
        if xx is None:
            self.xx = []
            self.xx_text = []
        elif xx_is_categorical:
            self.xx = list(range(len(xx)))
            self.xx_text = list(map(str, xx))
        else:
            self.xx = xx
            self.xx_text = list(map(str, xx))

    def get_xx(self):
        """Listing conditional values

        Provide a list of all sampled conditional values.
        """
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
            # Assume that doseless responses are constant over all values
            if len(one_xx) == 1 and len(two_xx) > 1:
                return two_xx
            elif len(two_xx) == 1 and len(one_xx) > 1:
                return one_xx
            elif len(one_xx) == 1 and len(two_xx) == 1:
                return ['']

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
            if np.array_equal(model.xx, xx) or len(model.xx) == 1:
                pass
            else:
                model = model.interpolate_x(xx)

        return model

    @staticmethod
    def intersect_x(one, two):
        if one.xx_is_categorical:
            xx = UnivariateModel.intersect_get_x(True, one.xx_text, two.xx_text)
        else:
            xx = UnivariateModel.intersect_get_x(False, one.xx, two.xx)
        one = UnivariateModel.intersect_get_model(one, xx)
        two = UnivariateModel.intersect_get_model(two, xx)

        return one, two, xx

    @staticmethod
    def intersect_x_all(models):
        # Check if any are categorical
        xx_is_categorical = False
        for model in models:
            if model.xx_is_categorical and len(model.get_xx()) > 1:
                xx_is_categorical = True
                break

        # Get a combined x from all
        if xx_is_categorical:
            xx = UnivariateModel.intersect_get_x(True, models[0].xx_text, models[1].xx_text)
            for model in models[2:]:
                xx = UnivariateModel.intersect_get_x(True, xx, model.xx_text)
        else:
            # In this case, assume that doseless responses are constant over all values
            xx = UnivariateModel.intersect_get_x(False, models[0].xx, models[1].xx)
            for model in models[2:]:
                xx = UnivariateModel.intersect_get_x(False, xx, model.xx)

        newmodels = []
        for model in models:
            newmodels.append(UnivariateModel.intersect_get_model(model, xx))

        return newmodels, xx
