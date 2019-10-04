"""
functionality to create the feature data necessary.
"""
from typing import List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

import pandas as pd
import pytz

from timetomodel.speccing import ModelSpecs
from timetomodel.utils.time_utils import (
    timedelta_to_pandas_freq_str,
    round_datetime,
    timedelta_fits_into,
)
from timetomodel.exceptions import NaNData


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
    datetime_indices = get_time_steps(time_range, specs)

    # load raw data series for the outcome data and regressors
    df[specs.outcome_var.name] = specs.outcome_var.load_series(
        expected_frequency=specs.frequency,
        transform_features=True,
        check_time_window=(datetime_indices[0], datetime_indices[-1]),
    )
    for reg_spec in specs.regressors:
        df[reg_spec.name] = reg_spec.load_series(
            expected_frequency=specs.frequency,
            transform_features=True,
            check_time_window=(datetime_indices[0], datetime_indices[-1]),
        )

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
        raise NaNData(
            "I found nan values in the feature frame I just constructed and I obviously can't"
            " make forecasts lacking data. Here's how many I found:\n\n%s\n\n"
            % df.drop(specs.outcome_var.name, axis=1).isnull().sum()
        )

    # if specs.model_type == "OLS":
    #    df = sm.add_constant(df)

    return df


def get_time_steps(
    time_range: Union[str, datetime, Tuple[datetime, datetime]], specs: ModelSpecs
) -> pd.DatetimeIndex:
    """ Get relevant datetime indices to build features for.

        The time_range parameter can be one or two datetime objects, in which case this function builds a DateTimeIndex.
        It can also be one of two strings: "train" or "test". In this situation, this function creates a training or
        testing period from model specs.

        TODO: we can check (and complain) if datetime objects are incompatible to specs.frequency
              e.g. if round_datetime(dt, by_seconds=specs.frequency.total_seconds()) != dt:
                       raise Exception("%s is not compatible with frequency %s." % (dt, specs.frequency))
              We have to discuss if we allow to use any time to start intervals or rather 15:00, 15:15, 15:30 etc ...
    """
    # check valid time_range parameter
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
            "Goal for DateTimeIndex construction needs to be either a string ('train', 'test'),"
            "a tuple of two datetime objects or one datetime object."
        )

    pd_frequency = timedelta_to_pandas_freq_str(specs.frequency)

    # easy cases: one or two datetime objects
    if isinstance(time_range, datetime):
        return pd.date_range(time_range, time_range, closed="left", freq=pd_frequency)
    elif isinstance(time_range, tuple):
        if not timedelta_fits_into(specs.frequency, time_range[1] - time_range[0]):
            raise Exception(
                "Start & end period (%s to %s) does not cleanly fit a multiple of the model frequency (%s)"
                % (time_range[0], time_range[1], specs.frequency)
            )
        return pd.date_range(
            time_range[0], time_range[1], closed="left", freq=pd_frequency
        )

    # special cases: "train" or "test" - we have to calculate from model specs
    length_of_data = specs.end_of_testing - specs.start_of_training
    if time_range == "train":
        end_of_training = (
            specs.start_of_training + length_of_data * specs.ratio_training_testing_data
        )
        end_of_training = round_datetime(
            end_of_training, specs.frequency.total_seconds()
        )
        logger.debug("Start of training: %s" % specs.start_of_training)
        logger.debug("End of training: %s" % end_of_training)
        return pd.date_range(
            specs.start_of_training, end_of_training, freq=pd_frequency
        )
    elif time_range == "test":
        start_of_testing = (
            specs.start_of_training
            + (length_of_data * specs.ratio_training_testing_data)
            + specs.frequency
        )
        start_of_testing = round_datetime(
            start_of_testing, specs.frequency.total_seconds()
        )
        logger.debug("Start of testing: %s" % start_of_testing)
        logger.debug("End of testing: %s" % specs.end_of_testing)
        return pd.date_range(start_of_testing, specs.end_of_testing, freq=pd_frequency)


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
