from multivariate_model import MultivariateModel

class SumMultiModel(MultivariateModel):
    def __init__(self, unis):
        super(SumMultiModel, self).__init__(map(lambda uni: uni.xx_is_categorical, unis), scaled)

