from scipy.interpolate import interp1d
import numpy as np

from .model import Model
from .univariate_model import UnivariateModel

class DeltaModel(UnivariateModel):
    def __init__(self, xx_is_categorical=False, xx=None, locations=None, scale=1):
        super(DeltaModel, self).__init__(False, xx, scale == 1)

        self.scale = scale
        self.locations = locations

    def kind(self):
        return 'delta_model'

    def copy(self):
        return DeltaModel(self.xx_is_categorical, list(self.get_xx()), self.locations[:], scale=self.scale)

    def scale_y(self, a):
        self.location = [loc * a for loc in self.locations]
        return self

    def scale_p(self, a):
        """Raising a delta function to a power makes no difference."""
        return self

    def filter_x(self, xx):
        return DeltaModel(self.xx_is_categorical, xx, [self.locations[xx.index(x)] for x in xx], scale=self.scale)

    def interpolate_x(self, newxx, kind='quadratic'):
        fx = interp1d(self.xx, self.locations, kind)
        newlocs = fx(newxx)
        
        return DeltaModel(self.xx_is_categorical, newxx, newlocs, scale=self.scale)

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            self.write(fp, delimiter)

    def write(self, file, delimiter):
        file.write("del1" + delimiter + str(self.scale) + "\n")
        for ii in range(len(self.xx)):
            file.write(str(self.xx[ii]) + delimiter + str(self.locations[ii]) + "\n")

    def to_points_at(self, x, ys):
        return [self.scale if y == self.locations[self.xx.index(x)] else 0 for y in ys]

    def draw_sample(self, x=None):
        return self.locations[self.xx.index(x)]

    def cdf(self, xx, yy):
        location = self.locations[self.xx.index(xx)]
        if yy < location:
            return 0
        else:
            return self.scale

    def init_from_delta_file(self, file, delimiter, status_callback=None):
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader)
        if header[0] != "del1":
            raise ValueError("Unknown format: %s" % (fields[0]))

        self.scale = float(header[1])
        self.locations = []

        xx_text = []
        xx = []
        self.xx_is_categorical = False
        for row in reader:
            xx_text.append(row[0])
            try:
                xx.append(float(row[0]))
            except ValueError:
                xx.append(len(xx))
                self.xx_is_categorical = True

            self.locations.append(float(row[1]))

        self.xx = xx
        self.xx_text = xx_text

    @staticmethod
    def merge(models):
        for model in models:
            if not model.scaled:
                raise ValueError("Only scaled distributions can be merged.")

        if isinstance(models[1], DeltaModel):
            # Check to see if all are equal-- otherwise error!
            for model in models[1:]:
                if len(model.locations) != len(models[0].locations):
                    raise ValueError("Inconsident x-values in delta merge")
                for jj in range(len(model.locations)):
                    if model.locations[jj] != models[0].locations[jj]:
                        raise ValueError("Non-matching location in delta merge")

        # Either identical deltas, or only one delta
        return models[0]
        
    @staticmethod
    def combine(one, two):
        if one.xx_is_categorical != two.xx_is_categorical:
            raise ValueError("Cannot combine models that do not agree on categoricity")
        if not one.scaled or not two.scaled:
            raise ValueError("Cannot combine unscaled models")

        if np.all(np.array(one.locations) == 0):
            return two

        if not isinstance(two, DeltaModel):
            raise ValueError("Combining non-zero delta with non-delta is not yet implemented.")
        
        (one, two, xx) = UnivariateModel.intersect_x(one, two)

        return DeltaModel(one.xx_is_categorical, xx, [one.locations[ii] + two.locations[ii] for ii in range(len(xx))], 1)

    @staticmethod
    def zero_delta(model):
        # The equivalent of scaling model to 0
        return DeltaModel(model.xx_is_categorical, model.xx, np.zeros(len(model.xx)), 1)

Model.mergers["delta_model"] = DeltaModel.merge
Model.mergers["delta_model+ddp_model"] = DeltaModel.merge
Model.mergers["delta_model+spline_model"] = DeltaModel.merge

Model.combiners['delta_model+delta_model'] = DeltaModel.combine
Model.combiners['delta_model+ddp_model'] = DeltaModel.combine
Model.combiners['delta_model+spline_model'] = DeltaModel.combine
