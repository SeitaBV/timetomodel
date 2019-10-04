import logging

import pandas as pd
import numpy as np

from statsmodels.base.transform import BoxCox

"""
Transforming data by configurable functionality.
For instance, after loading, or before and after forecasting.
"""

logger = logging.getLogger(__name__)


class Transformation(object):
    """Base class for one-way transformations."""

    def transform_series(self, x: pd.Series) -> pd.Series:
        return x

    def transform_dataframe(self, x: pd.DataFrame) -> pd.DataFrame:
        return x


class ParameterisedTransformation(Transformation):
    """Base class for transformations which get parameterised when initialised,
    so calls to transform can make use of those.
    transform() could even change parameters on the fly, if needed, to affect later calls."""

    def __init__(self, **kwargs):
        """Initialise transformation with named parameters.
        For example:
            >>> transformation = Transformation(lambda1=0.5, lambda2=1)
            >>> transformation.params.lambda1
            0.5
        """
        self.params = type("Params", (), {})
        self._set_params(**kwargs)

    def _set_params(self, **kwargs):
        """Assign named variables as attributes."""
        for k, v in kwargs.items():
            setattr(self.params, k, v)


class ReversibleTransformation(ParameterisedTransformation):
    """Base class for transformations of time series data with a state, so that back transformations are possible
    which depend on the actual data being transformed.
    An example would be the BoxCox Transformation, for which we provide an implementation below.

    Initialise with your custom transformation parameters and define custom functions to transform and back-transform.
    The transform function can re-set your original parameters, in case it wants to adapt to the data.
    The (optional) back-transformation is applied to forecasted data.
    """

    def back_transform_value(self, x):
        """Return back-transformed values, based on parameters."""
        return x


class BoxCoxTransformation(ReversibleTransformation):
    """Box-Cox transformation.

     For positive-only or negative-only data, no parameters are needed.
     For non-negative or non-positive data with zero values, set lambda2 to a positive number (e.g. 1).

                            {   ( (x' + lambda2) ^ lambda1 − 1) / lambda1        if lambda1 != 0
     y(lambda1, lambda2) = {
                            {   log(x' + lambda2)                                if lambda1 == 0

    where:            x' = x * lambda3
    """

    def __init__(self, lambda2: float = 0.1):
        super().__init__(lambda2=lambda2)

    def transform_series(self, x: pd.Series) -> pd.Series:
        params = {}
        orig_index = x.index
        x = x.values
        if (x[~np.isnan(x)] + self.params.lambda2 > 0).all():
            y, params["lambda1"] = BoxCox.transform_boxcox(
                BoxCox(), x + self.params.lambda2
            )
            params["lambda3"] = 1
        elif (x[~np.isnan(x)] - self.params.lambda2 < 0).all():
            y, params["lambda1"] = BoxCox.transform_boxcox(
                BoxCox(), -x + self.params.lambda2
            )
            params["lambda3"] = -1
        else:
            raise ValueError(
                "Box-Cox transformation not suitable for x with both positive and negative values."
            )
        self._set_params(**params)
        return pd.Series(index=orig_index, data=y)

    def back_transform_value(self, x):
        try:
            y = (
                BoxCox.untransform_boxcox(BoxCox(), x, lmbda=self.params.lambda1)
                - self.params.lambda2
            ) / self.params.lambda3
        except Warning as w:
            logger.debug(  # not sure if this needs to be a warning, it happens quite often to me right now
                "Back-transform failed for y(x, lambda1, lambda2, lambda3) with:\n"
                "x = %s\n"
                "lambda1 = %s\n"
                "lambda2 = %s\n"
                "lambda3 = %s\n"
                "warning = %s\n"
                "Returning 0 value instead."
                % (x, self.params.lambda1, self.params.lambda2, self.params.lambda3, w)
            )
            y = (0.0 - self.params.lambda2) / self.params.lambda3
        return y
