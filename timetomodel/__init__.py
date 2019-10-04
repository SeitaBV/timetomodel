"""
Create or import top-level classes/methods.
"""
from typing import Tuple, Union

from sklearn.base import RegressorMixin as SkLearnModel
from statsmodels.base.wrapper import ResultsWrapper as StatsModel


# Here we list any class we might use. In principle, they should be expected to have a fit(X, y) and a predict(X)
# method.
# However, the statsmodels library provides Model and Results classes. The former only hold X and y data, while the
# latter represents a fitted (parameterised) Model (with a working predict method), plus the result statistics
# (the Model object is also in there).
MODEL_CLASSES = (StatsModel, SkLearnModel)
MODEL_TYPES = Union[MODEL_CLASSES]


# First public import block
from timetomodel.speccing import (
    ModelSpecs,
    ObjectSeriesSpecs,
    DFFileSeriesSpecs,
    DBSeriesSpecs,
)


class ModelState(object):
    """
    This class abstracts all information we need to describe a model and how it was made.
    It is simply a container to hold all model-relevant state information:
    The trained model and the ModelsSpecs it is based on."""

    model: MODEL_TYPES  # a fitted model
    specs: ModelSpecs

    def __init__(self, model, specs):
        if not isinstance(model, MODEL_CLASSES):
            raise Exception(
                "ModelState(): model parameter needs to be of type %s instead of type %s"
                % (str(MODEL_CLASSES), type(model))
            )
        self.model = model
        if not isinstance(specs, ModelSpecs):
            raise Exception(
                "ModelState(): specs parameter needs to be of type <ModelSpecs>"
            )
        self.specs = specs

    def split(self) -> Tuple[MODEL_TYPES, ModelSpecs]:
        """Return model and specs separately"""
        return self.model, self.specs

    def __repr__(self):
        return "ModelState: <%s, %s>" % (self.model, self.specs)


# second public import block
from timetomodel.modelling import (
    create_fitted_model,
    evaluate_models,
    model_param_grid_search,
)
from timetomodel.forecasting import make_forecast_for, make_rolling_forecasts
