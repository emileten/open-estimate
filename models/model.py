import copy

class Model(object):
    mergers = {}
    combiners = {}

    def __init__(self, scaled=True):
        self.scaled = scaled

    def kind(self):
        return 'unknown'

    def copy(self):
        return copy.deepcopy(self)

    # Determine the conditional probability density at ys
    def to_points_at(self, x, ys):
        raise "to_points_at not implemented"

    # Determine the value of y for a given p
    def eval_pval(self, x, p):
        raise "eval_pval not implemented"

    def scale_y(self, a):
        raise "scale_y not implemented"

    def get_mean(self, x=None):
        if not self.scaled:
            raise "Cannot take mean of unscaled distribution."
        
        return np.nan

    def get_sdev(self, x=None):
        if not self.scaled:
            raise "Cannot take mean of unscaled distribution."
        
        return np.nan

    def attribute_list(self):
        return []

    def get_attribute(self, title):
        raise title + " not available"

    @staticmethod
    def merge(models):
        if len(models) == 1:
            return models[0]
        
        # Sort into groups of models
        groups = {}
        for ii in range(len(models)):
            mic = models[ii].kind()
            if mic not in groups:
                groups[mic] = []
            groups[mic].append(models[ii])

        merged = []
        for mic in groups:
            merged.append(Model.mergers[mic](groups[mic]))

        result = merged[0]
        for ii in range(1, len(merged)):
            try:
                result = Model.mergers[result.kind() + "+" + merged[ii].kind()]([result, merged[ii]])
            except:
                result = Model.mergers[merged[ii].kind() + "+" + result.kind()]([merged[ii], result])

        return result

    @staticmethod
    def combine(models, factors):
        """Construct a weighted sum over the shared values of x"""
        sofar = None
        for ii in range(len(models)):
            model = models[ii]

            scaled = model.copy().scale_y(factors[ii])
            if sofar is None:
                sofar = scaled
            else:
                try:
                    sofar = Model.combiners[sofar.kind() + "+" + scaled.kind()](sofar, scaled)
                except:
                    sofar = Model.combiners[scaled.kind() + "+" + sofar.kind()](scaled, sofar)

        return sofar
