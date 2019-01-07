from typing import Tuple
from datetime import datetime, timedelta

from pandas import DataFrame, Series, date_range
import pytz

from ts_forecasting_pipeline import MODEL_CLASSES, ModelState, ModelSpecs
from ts_forecasting_pipeline.modelling import create_fitted_model
from ts_forecasting_pipeline.featuring import construct_features, get_time_steps
from ts_forecasting_pipeline.utils.time_utils import get_closest_quarter


"""
Functionality for making predictions per time slot.
"""


def make_forecast_for(
    specs: ModelSpecs, features: DataFrame, model: MODEL_CLASSES
) -> float:
    """
    Make a forecast for the given feature vector.
    """

    y_hat = model.predict(features)

    # Apply back-transformation if the output data was transformed
    if specs.transformation is not None:
        y_hat = specs.transformation.back_transform(y_hat)

    return y_hat


def update_model(
    time_step: datetime,
    current_model: MODEL_CLASSES,
    specs: ModelSpecs,
    feature_frame: DataFrame,
) -> ModelState:
    new_model: MODEL_CLASSES = current_model
    """ Create model if current one is outdated or not yet created."""
    if (
        current_model is None
        or time_step - specs.creation_time >= specs.remodel_frequency
    ):
        # print("Fitting new model before predicting %s ..." % time_step)
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
) -> Tuple[Series, ModelState]:
    """
    Repeatedly call make_forecast - for all time steps the desired time window.
    The time window of the specs (training + test data) is moved forwards also step by step.
    Will fail if series specs do not allocate enough data.
    May create a model whenever the previous one is outdated.
    Return Pandas.Series as result, as well as the last ModelState.
    """

    # Prepare time range
    for dt in (start, end):
        if dt.tzinfo is None:
            dt.replace(tzinfo=pytz.utc)
    training_start = get_closest_quarter(model_specs.start_of_training)
    start = get_closest_quarter(start)
    end = get_closest_quarter(end)

    # First, compute one big feature frame, once.
    feature_frame: DataFrame = construct_features((training_start, end), model_specs)

    """
    specs.outcome_var = ObjectSeriesSpecs(
        name=specs.outcome_var.name, data=specs.outcome_var.load_series()
    )
    specs.regressors = [
        ObjectSeriesSpecs(name=r.name, data=r.load_series()) for r in specs.regressors
    ]
    """
    values = Series(
        index=date_range(start, end, freq="15T", closed="left", tz=start.tzinfo)
    )
    time_step = start
    model = None
    print("[ts-forecasting-pipeline] Forecasting from %s to %s" % (start, end))
    while time_step < end:
        model, specs = update_model(
            time_step, model, model_specs, feature_frame=feature_frame
        ).split()
        features = feature_frame.loc[time_step:time_step].iloc[:, 1:]
        values[time_step] = make_forecast_for(model_specs, features, model)
        time_step = time_step + timedelta(minutes=15)

    return values, ModelState(model, model_specs)
