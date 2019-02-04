from typing import Tuple
import logging

import numpy as np

from statsmodels.base.transform import BoxCox

"""
Transforming data by configurable functionality.
For instance, after loading, or before and after forecasting.
These functions work on numpy arrays.
"""

logger = logging.getLogger(__name__)


class Transformation(object):
    """Base class for transformations of time series data.
    Initialise with your custom transformation parameters and define custom functions to transform and back-transform.
    These functions can also re-set your original parameters.
    You can optionally also provide a back-transformation, which is applied to forecasted data.

    An example would be the BoxCox Transformation, for which we provide an implementation below.
    """

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

    def transform(self, x: np.array) -> Tuple[np.array, dict]:
        """Return transformed data and set new transformation parameters if applicable."""
        params = {}
        y = x
        self._set_params(**params)
        return y, params

    def back_transform(self, x: np.array) -> np.array:
        """Return back-transformed data."""
        y = x
        return y


# TODO: move this custom implementation somewhere, maybe a func store?
class BoxCoxTransformation(Transformation):
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

    def transform(self, x: np.array) -> Tuple[np.array, dict]:
        params = {}
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
        return y

    def back_transform(self, x: np.array) -> np.array:
        try:
            y = (
                BoxCox.untransform_boxcox(BoxCox(), x, lmbda=self.params.lambda1)
                - self.params.lambda2
            ) / self.params.lambda3
        except Warning as w:
            if (
                w.__str__() == "invalid value encountered in power"
                and (x < 0).all()
                and self.params.lambda1 < 1
            ):

                # Resolve a numpy problem for raising a number close to 0 to a large number, i.e. -0.12^6.25
                y = (np.zeros(*x.shape) - self.params.lambda2) / self.params.lambda3
            else:
                logger.warn(
                    "Back-transform failed for y(x, lambda1, lambda2, lambda3) with:\n"
                    "x = %s\n"
                    "lambda1 = %s\n"
                    "lambda2 = %s\n"
                    "lambda3 = %s\n"
                    "warning = %s\n"
                    "Returning 0 value instead."
                    % (
                        x,
                        self.params.lambda1,
                        self.params.lambda2,
                        self.params.lambda3,
                        w,
                    )
                )
                y = (np.zeros(*x.shape) - self.params.lambda2) / self.params.lambda3
        return y
