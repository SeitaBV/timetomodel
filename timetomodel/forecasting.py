from typing import Tuple
from datetime import datetime
import logging

import pandas as pd
import pytz

from timetomodel import MODEL_CLASSES, ModelState, ModelSpecs
from timetomodel.modelling import create_fitted_model
from timetomodel.featuring import construct_features, get_time_steps
from timetomodel.utils.time_utils import timedelta_to_pandas_freq_str


"""
Functionality for making predictions per time slot.
"""


logger = logging.getLogger(__name__)


def make_forecast_for(
    specs: ModelSpecs, features: pd.DataFrame, model: MODEL_CLASSES
) -> float:
    """
    Make a forecast for the given feature vector.
    """

    y_hat = model.predict(features)

    if isinstance(y_hat, pd.Series):
        y_hat = y_hat.iloc[0]

    # Apply back-transformation, as the output data was transformed before
    if specs.outcome_var.feature_transformation is not None:
        y_hat = specs.outcome_var.feature_transformation.back_transform_value(y_hat)

    return y_hat


def update_model(
    time_step: datetime,
    current_model: MODEL_CLASSES,
    specs: ModelSpecs,
    feature_frame: pd.DataFrame,
) -> ModelState:
    new_model: MODEL_CLASSES = current_model
    """ Create model if current one is outdated or not yet created."""
    if (
        current_model is None
        or time_step - specs.creation_time >= specs.remodel_frequency
    ):
        logger.debug("Fitting new model before predicting %s ..." % time_step)
        if current_model is not None:
            # move the model's series specs further in time
            specs.start_of_training = specs.start_of_training + specs.remodel_frequency
            specs.end_of_testing = specs.end_of_testing + specs.remodel_frequency
        relevant_time_steps = get_time_steps(time_range="train", specs=specs)
        new_model = create_fitted_model(
            specs, "", regression_frame=feature_frame.loc[relevant_time_steps]
        )
        specs.creation_time = time_step

    return ModelState(new_model, specs)


def make_rolling_forecasts(
    start: datetime,  # Start of forecast period
    end: datetime,  # End of forecast period
    model_specs: ModelSpecs,
) -> Tuple[pd.Series, ModelState]:
    """
    Repeatedly call make_forecast - for all time steps the desired time window
    (end is excluding).
    The time window of the specs (training + test data) is moved forwards also step by step.
    Will fail if series specs do not allocate enough data.
    May create a model whenever the previous one is outdated.
    Return Pandas.Series as result, as well as the last ModelState.
    """

    # Prepare time range
    for dt in (start, end):
        if dt.tzinfo is None:
            dt.replace(tzinfo=pytz.utc)

    # First, compute one big feature frame, once.
    feature_frame: pd.DataFrame = construct_features(
        (model_specs.start_of_training, end), model_specs
    )

    pd_frequency = timedelta_to_pandas_freq_str(model_specs.frequency)
    values = pd.Series(
        index=pd.date_range(
            start, end, freq=pd_frequency, closed="left", tz=start.tzinfo
        )
    )
    time_step = start
    model = None
    logger.info("Forecasting from %s to %s" % (start, end))
    while time_step < end:
        model, specs = update_model(
            time_step, model, model_specs, feature_frame=feature_frame
        ).split()
        features = feature_frame.loc[time_step:time_step].iloc[:, 1:]
        values[time_step] = make_forecast_for(model_specs, features, model)
        time_step = time_step + model_specs.frequency

    return values, ModelState(model, model_specs)
