import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.api import OLS

from timetomodel import modelling, speccing, transforming

logger = logging.getLogger(__name__)


def create_dummy_model_state(
    data_start: datetime,
    data_range_in_hours: int,
    outcome_feature_transformation: Optional[
        transforming.ReversibleTransformation
    ] = None,
    regressor_feature_transformation: Optional[
        transforming.ReversibleTransformation
    ] = None,
) -> modelling.ModelState:
    """
    Create a dummy model. data increases linearly, regressor is constant (useless).
    Use two different ways to define Series specs to test them.
    """
    dt_range = pd.date_range(
        data_start,
        data_start + timedelta(hours=data_range_in_hours),
        closed="left",
        freq="1H",
    )
    reg_range = pd.date_range(
        data_start,
        data_start + timedelta(hours=data_range_in_hours) + timedelta(days=1),
        closed="left",
        freq="1H",
    )  # 1 additional day of regressor data is available
    outcome_values = [0]
    regressor_values = [5]
    for i in range(1, len(dt_range)):
        outcome_values.append(outcome_values[i - 1] + 1)
    for i in range(1, len(reg_range)):
        regressor_values.append(5)
    outcome_series = pd.Series(index=dt_range, data=outcome_values)
    regressor_series = pd.Series(index=reg_range, data=regressor_values)
    specs = modelling.ModelSpecs(
        outcome_var=speccing.ObjectSeriesSpecs(
            outcome_series,
            name="my_outcome",
            feature_transformation=outcome_feature_transformation,
        ),
        model=OLS,
        lags=[2, 3],  # lags of interest are 2 and 3 hours (we'll ignore the last hour)
        frequency=timedelta(hours=1),
        horizon=timedelta(minutes=240),
        remodel_frequency=timedelta(hours=48),
        regressors=[
            speccing.ObjectSeriesSpecs(
                regressor_series,
                "my_regressor",
                feature_transformation=regressor_feature_transformation,
            )
        ],
        start_of_training=data_start
        + timedelta(hours=3),  # leaving room for NaN in lags
        end_of_testing=data_start + timedelta(hours=int(data_range_in_hours / 3)),
    )
    return modelling.ModelState(
        modelling.create_fitted_model(specs, version="0.1"), specs
    )


class MyDFPostProcessing(transforming.Transformation):
    def transform_dataframe(self, df: pd.DataFrame):
        """Keep the most recent observation, drop duplicates"""
        return (
            df.sort_values(by=["horizon"], ascending=True)
            .drop_duplicates(subset=["datetime"], keep="first")
            .sort_values(by=["datetime"])
        )


class MyAdditionTransformation(transforming.ReversibleTransformation):
    def transform_series(self, x: pd.Series):
        logger.debug("Adding %s to %s ..." % (self.params.addition, x))
        return x + self.params.addition

    def back_transform_value(self, y: np.array):
        logger.debug("Subtracting %s from %s ..." % (self.params.addition, y))
        return y - self.params.addition


class MyMultiplicationTransformation(transforming.ReversibleTransformation):
    def transform_series(self, x: pd.Series):
        return x * self.params.factor

    def back_transform_value(self, y: np.array):
        return y / self.params.factor


class MyNanIntroducingTransformation(transforming.ReversibleTransformation):
    def transform_series(self, x: pd.Series) -> pd.Series:
        x.iloc[5] = np.nan
        return x

    def back_transform_value(self, y: np.array) -> np.array:
        return y
