from .multivariate_model import MultivariateModel
from .model import Model

import re

class OuterMultiModel(MultivariateModel):
    def __init__(self, xxs, xx_is_categoricals, union, scaled=True):
        super(OuterMultiModel, self).__init__(xx_is_categoricals, scaled)
        self.xxs = xxs
        self.union = union

    def kind(self):
        return 'outer_model'

    def dims(self):
        return list(map(len, self.xxs))

    def write_file(self, filename, delimiter):
        with open(filename, 'w') as fp:
            return self.write(fp, delimiter)

    def write(self, file, delimiter):
        file.write("omm1\n")
        self.union.write(file, delimiter)

    def init_from_union(self, union):
        # go through elements, filling out xxs and guessing at is_categoricals
        self.union = union

        firsts = union.xx_text[0].split(':')
        self.xxs = [[]] * len(firsts)
        self.xx_is_categoricals = [False] * len(firsts)

        for x_text in union.xx_text:
            coords = x_text.split(':')
            for ii in range(len(coords)):
                try:
                    xx = float(coords[ii])
                except Exception as ex:  # CATBELL
                    import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
                    xx = coords[ii]
                    self.xx_is_categoricals[ii] = True

                if xx not in self.xxs[ii]:
                    self.xxs[ii].append(xx)

    def scale_p(self, a):
        self.union.scale_p(a)
        return self

    def float_condition(self, conditions):
        if conditions.count(None) <= 1:
            # Identify the 2^N or 2^N-1 values bounding the point
            raise ValueError("not implemented yet")
        else:
            # This case isn't "bad", it's just not something we've needed to handle yet
            raise ValueError("condition called with multiple unconditioned variables")

    def condition(self, conditions):
        if conditions.count(None) <= 1:
            # Produce a univariate model
            # Regular expression that matches all in conditions
            pattern = '^' + ':'.join(map(OuterMultiModel.re_condition, conditions)) + '$'
            prog = re.compile(pattern)
            
            xx = [x for x in self.union.get_xx() if prog.match(x)]
            univar = self.union.filter_x(xx)
            if conditions.count(None) == 1:
                univar.xx_text = [prog.match(x).group(1) for x in univar.get_xx()]
                if not self.xx_is_categoricals[conditions.index(None)]:
                    univar.xx = list(map(float, univar.xx_text))
                    univar.xx_is_categorical = False

            return univar
        else:
            # This case isn't "bad", it's just not something we've needed to handle yet
            raise ValueError("condition called with multiple unconditioned variables")

    def default_condition(self):
        return self.condition([None] + [xx[0] for xx in self.xxs[1:]])

    re_numeric = re.compile(r"\d*\.?\d*")

    @staticmethod
    def re_condition(condition):
        if condition is None:
            return "([^:]*)"
        
        if isinstance(condition, float):
            return OuterMultiModel.re_condition(str(condition))
            
        if condition != '.' and OuterMultiModel.re_numeric.match(condition):
            if '.' in condition:
                condition += '0*'
            else:
                condition += '\.?0*'
        return condition
        
Model.mergers["outer_model"] = lambda models: Model.merge([model.default_condition() for model in models])
Model.mergers["outer_model+ddp_model"] = lambda models: Model.merge([models[0].default_condition(), models[1]])
Model.mergers["outer_model+spline_model"] = lambda models: Model.merge([models[0].default_condition(), models[1]])
