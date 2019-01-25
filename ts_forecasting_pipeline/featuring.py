"""
functionality to create the feature data necessary.
"""
from typing import List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

import pandas as pd
import pytz

from ts_forecasting_pipeline.speccing import ModelSpecs
from ts_forecasting_pipeline.utils.time_utils import (
    get_closest_quarter,
    timedelta_to_pandas_freq_str,
)


logger = logging.getLogger(__name__)


def construct_features(
    time_range: Union[Tuple[datetime, datetime], datetime, str], specs: ModelSpecs
) -> pd.DataFrame:
    """
    Create a DataFrame based on ModelSpecs. This means loading data, potentially running a custom transformation,
    adding lags & returning only the relevant date rows.

    time_range can be specified by you (a range of steps between two datetimes or a single step - one datetime-,
    which is useful when we are predicting said step) but usually you’d say “train” or “test” to let the model specs
    determine the time steps.
    """
    df = pd.DataFrame()
    # load raw data series for the outcome data
    outcome_series = specs.outcome_var.load_series(expected_frequency=specs.frequency)
    if outcome_series.index.freqstr != timedelta_to_pandas_freq_str(specs.frequency):
        raise Exception(
            "Loaded data for %s has different frequency (%s) than used in model (%s)."
            % (
                specs.outcome_var.name,
                outcome_series.index.freqstr,
                timedelta_to_pandas_freq_str(specs.frequency),
            )
        )
    df[specs.outcome_var.name] = outcome_series

    datetime_indices = get_time_steps(time_range, specs)

    # perform the custom transformation, if needed, and store the transformation parameters to be able to back-transform
    if specs.transformation is not None:
        df[specs.outcome_var.name] = specs.transformation.transform(
            df[specs.outcome_var.name]
        )

    # load raw data series for the regressors
    for reg_spec in specs.regressors:
        df[reg_spec.name] = reg_spec.load_series(expected_frequency=specs.frequency)
        if outcome_series.index.freqstr != timedelta_to_pandas_freq_str(
            specs.frequency
        ):
            raise Exception(
                "Loaded data for %s has different frequency (%s) than used in model (%s)."
                % (
                    reg_spec.name,
                    outcome_series.index.freqstr,
                    timedelta_to_pandas_freq_str(specs.frequency),
                )
            )

    # perform the custom transformation, if needed
    if specs.regressor_transformation is not None:
        for reg_spec in specs.regressors:
            if reg_spec.name in specs.regressor_transformation:
                if specs.regressor_transformation[reg_spec.name] is not None:
                    df[reg_spec.name] = specs.regressor_transformation[
                        reg_spec.name
                    ].transform(df[reg_spec.name])

    # add lags on the outcome var
    df = add_lags(df, specs.outcome_var.name, specs.lags)
    outcome_lags = [
        lag_name
        for lag_name in [
            specs.outcome_var.name + "_" + lag_to_suffix(lag) for lag in specs.lags
        ]
        if lag_name in df.columns
    ]

    # now select only relevant columns and relevant datetime indices
    relevant_columns = (
        [specs.outcome_var.name] + outcome_lags + [r.name for r in specs.regressors]
    )
    df = df[relevant_columns].reindex(datetime_indices)

    # Check for nan values in lagged and regressor data
    if df.drop(specs.outcome_var.name, axis=1).isnull().values.any():
        raise Exception(
            "I found nan values and I obviously can't make forecasts lacking data. Here's how many I found:\n\n%s\n\n"
            % df.drop(specs.outcome_var.name, axis=1).isnull().sum()
        )

    # if specs.model_type == "OLS":
    #    df = sm.add_constant(df)

    return df


def get_time_steps(
    time_range: Union[str, datetime, Tuple[datetime, datetime]], specs: ModelSpecs
) -> pd.DatetimeIndex:
    """ get relevant datetime indices to build features for."""
    if not (
        isinstance(time_range, datetime)
        or (
            isinstance(time_range, tuple)
            and isinstance(time_range[0], datetime)
            and isinstance(time_range[1], datetime)
        )
        or (isinstance(time_range, str) and time_range in ("train", "test"))
    ):
        raise Exception(
            "Purpose for dataframe construction needs to be either 'train', 'test',"
            "a tuple of two datetime objects or one datetime object."
        )

    pd_frequency = timedelta_to_pandas_freq_str(specs.frequency)
    if isinstance(time_range, datetime):
        return pd.date_range(time_range, time_range, closed="left", freq=pd_frequency)
    elif isinstance(time_range, tuple):
        return pd.date_range(
            time_range[0], time_range[1], closed="left", freq=pd_frequency
        )

    length_of_data = specs.end_of_testing - specs.start_of_training
    datetime_indices = None
    if time_range == "train":
        end_of_training = (
            specs.start_of_training + length_of_data * specs.ratio_training_testing_data
        )
        if pd_frequency == "15T":  # TODO: support other frequencies or find a generic rounding
            end_of_training = get_closest_quarter(end_of_training)
        logger.info("Start of training: %s" % specs.start_of_training)
        logger.info("End of training: %s" % end_of_training)
        datetime_indices = pd.date_range(
            specs.start_of_training, end_of_training, freq=pd_frequency
        )
    elif time_range == "test":
        start_of_testing = (
            specs.start_of_training
            + (length_of_data * specs.ratio_training_testing_data)
            + timedelta(minutes=15)
        )
        if pd_frequency == "15T":
            start_of_testing = get_closest_quarter(start_of_testing)
        logger.info("Start of testing: %s" % start_of_testing)
        logger.info("End of testing: %s" % specs.end_of_testing)
        datetime_indices = pd.date_range(
            start_of_testing, specs.end_of_testing, freq=pd_frequency
        )

    return datetime_indices


def lag_to_suffix(lag: int) -> str:
    """
    Return the suffix for a column, given its lag.
    """
    if lag < 0:
        str_lag = "f" + str(abs(lag))
    else:
        str_lag = "l" + str(abs(lag))
    return str_lag


def add_lags(df: pd.DataFrame, column: str, lags: List[int]) -> Optional[pd.DataFrame]:
    """
    Creates lag columns for a column in the dataframe. Lags are in fifteen minute steps (15T).
    Positive values are lags, while negative values are future values.
    The new columns are named like the lagged column, plus "_l<lag>" (or "_f<lag>" for positive lags (future values)),
    where <lag> is the 15-minute lag value or the translation of it into days, weeks or years.
    In case of positive 'lags' (future values), new columns are named like the lagged column, plus "_f<lag>".

    TODO: We could also review if using statsmodels.tsa.tsatools.add_lag is of interest, but here self-made is probably
          what we need.
    """

    if not lags:
        return df

    # Make sure the DataFrame has rows to accommodate each lag
    max_lag = timedelta(minutes=15 * max(lags))
    min_lag = timedelta(minutes=15 * min(lags))
    df_start = min(df.index[0], df.index[0] + min_lag)
    df_end = max(df.index[-1], df.index[-1] + max_lag)
    df = df.reindex(
        pd.date_range(
            start=df_start.astimezone(pytz.utc),
            end=df_end.astimezone(pytz.utc),
            freq=df.index.freq,
        )
    )

    lag_names = [l for l in lags]

    for lag in lags:
        lag_name = lag_names[lags.index(lag)]
        df[column + "_" + lag_to_suffix(lag_name)] = df[column].shift(lag)

    return df
