import inspect
import logging
import os
import warnings
from datetime import datetime, timedelta, tzinfo
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytz
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query

from timetomodel.exceptions import IncompatibleModelSpecs, MissingData, NaNData, UnsupportedModel
from timetomodel.transforming import ReversibleTransformation, Transformation
from timetomodel.utils.debug_utils import render_query
from timetomodel.utils.time_utils import (timedelta_fits_into,
                                          timedelta_to_pandas_freq_str,
                                          tz_aware_utc_now)

"""
Specs for the context of your model and how to treat your model data.
"""

DEFAULT_RATIO_TRAINING_TESTING_DATA = 2 / 3
DEFAULT_REMODELING_FREQUENCY = timedelta(days=1)

np.seterr(all="warn")
warnings.filterwarnings("error", message="invalid value encountered in power")

logger = logging.getLogger(__name__)


class SeriesSpecs(object):
    """Describes a time series (e.g. a pandas Series).
    In essence, a column in the regression frame, filled with numbers.

    Using this base class, the column will be filled with NaN values.

    If you have data to be loaded in automatically, you should be using one of the subclasses, which allow to describe
    or pass in an actual data source to be loaded.

    When dealing with columns, our code should usually refer to this superclass so it does not need to care
    which kind of data source it is dealing with.
    """

    # The name in the resulting feature frame, and possibly in the saved model specs (named by outcome var)
    name: str
    # The name of the data column in the data source. If None, the name will be tried.
    column: Optional[str]
    # timezone of the data - e.g. useful when de-serializing data (pandas serialises to UTC)
    original_tz: tzinfo
    # Custom transformation on feature data to be made before forecasting, back-transformed right after.
    feature_transformation: Optional[ReversibleTransformation]
    # Custom processing on data right after loading, e.g. for cleanup
    post_load_processing: Optional[Transformation]
    # Custom resampling parameters. All parameters apply to pd.resample, only "aggregation" is the name
    # of the aggregation function to be called of the resulting resampler
    resampling_config: Dict[str, Any]
    interpolation_config: Dict[str, Any]

    def __init__(
        self,
        name: str,
        original_tz: Optional[
            tzinfo
        ] = None,  # TODO: why should this be possible to be set?
        feature_transformation: Optional[ReversibleTransformation] = None,
        post_load_processing: Optional[Transformation] = None,
        resampling_config: Dict[str, Any] = None,
        interpolation_config: Dict[str, Any] = None,
    ):
        self.name = name
        self.original_tz = original_tz
        self.feature_transformation = feature_transformation
        self.post_load_processing = post_load_processing
        self.resampling_config = resampling_config
        self.interpolation_config = interpolation_config
        self.__series_type__ = self.__class__.__name__

    def as_dict(self):
        return vars(self)

    def _load_series(self) -> pd.Series:
        """Subclasses overwrite this function to get the raw data.
        This method is responsible to call any post_load_processing at the right place."""
        data = pd.Series()
        if self.post_load_processing is not None:
            return self.post_load_processing.transform_series(data)
        return data

    def load_series(
        self,
        expected_frequency: timedelta,
        time_window: Tuple[datetime, datetime] = None,
        transform_features: bool = False,
        check_time_window: bool = False,
    ) -> pd.Series:
        """Load the series data, check compatibility of series data with model specs
           and perform feature transformation, if needed.

           The actual implementation how to load is deferred to _load_series. Overwrite that for new subclasses.

           This function resamples data if the frequency is not equal to the expected frequency.
           It is possible to customise this resampling (without that, we aggregate means after default resampling).
           To customize resampling, pass in a `resampling_config` argument when you initialize a SeriesSpecs,
           with an aggregation method name (e.g. "mean") and kw params which are to be passed into
           `pandas.Series.resample`. For example:

           `resampling_config={"closed": "left", "aggregation": "sum"}`

           Similarly, pass in an `interpolation_config` to the class with kw params to pass into 
           `pandas.Series.interpolate`. For example, to fill gaps of at most 1 consecutive NaN value through
           interpolation of the time index:

           `interpolation_config={"method": "time", "limit": 1}`

           To be able to upsample, make sure a time_window is set.
           The time window is a tuple stating the index of the first and the index of the last data point.

           You can check if a time window would be feasible, i.e. if enough data is loaded, and get suggestions.
        """
        data = self._load_series().sort_index()

        # check if data has a DateTimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise IncompatibleModelSpecs(
                "Loaded series has no DatetimeIndex, but %s" % type(data.index).__name__
            )

        # make sure we have a time zone (default to UTC), save original time zone
        if data.index.tzinfo is None:
            self.original_tz = pytz.utc
            data.index = data.index.tz_localize(self.original_tz)
        else:
            self.original_tz = data.index.tzinfo

        # interpret naive time_window in timezone of data
        if time_window is not None and time_window[0].tzinfo is None:
            time_window = (self.original_tz.localize(time_window[0]), time_window[1])
        if time_window is not None and time_window[1].tzinfo is None:
            time_window = (time_window[0], self.original_tz.localize(time_window[1]))

        # check if time series frequency is okay, if not then resample, and check again
        if data.index.freqstr != timedelta_to_pandas_freq_str(expected_frequency):
            data = self.resample_data(
                data=data, expected_frequency=expected_frequency, time_window=time_window
            )

        # Raise error if data is empty or contains nan values
        if data.empty:
            raise MissingData(
                "No values found in requested %s data. It's no use to continue I'm afraid."
            )
        if data.isnull().values.any() and self.interpolation_config is None:
            raise NaNData(
                "Nan values found in the requested %s data. It's no use to continue I'm afraid."
            )

        # check if we have enough data for the expected time window
        if check_time_window and time_window is not None:
            error_msg = ""
            if data.index[0] > time_window[0]:
                error_msg += (
                    "Data for %s starts too late (at %s), while we need data from %s "
                    % (
                        self.name,
                        data.index[0],
                        time_window[0].astimezone(data.index[0].tzinfo),
                    )
                )
            if data.index[-1] < time_window[1]:
                error_msg += (
                    "Data for %s ends too early (at %s), while we need data until %s "
                    % (
                        self.name,
                        data.index[-1],
                        time_window[1].astimezone(
                            data.index[-1].tzinfo
                        ),
                    )
                )
            if error_msg:
                raise MissingData(error_msg)

            if data.index.freqstr != timedelta_to_pandas_freq_str(expected_frequency):
                raise IncompatibleModelSpecs(
                    "Loaded data for %s has different frequency (%s) than used in model specs expect (%s)."
                    % (
                        self.name,
                        data.index.freqstr,
                        timedelta_to_pandas_freq_str(expected_frequency),
                    )
                )

        # interpolate after the frequency is set (setting the frequency may have created additional nan values)
        if self.interpolation_config is not None:
            data = self.interpolate_data(data)

        if transform_features and self.feature_transformation is not None:
            data = self.feature_transformation.transform_series(data)

        return data

    def resample_data(
        self,
        data,
        time_window,
        expected_frequency,
    ) -> pd.Series:
        if time_window is None:
            time_window = (data.index[0], data.index[-1])
        index = pd.date_range(
            *time_window,
            freq=expected_frequency,
        )
        if len(data) < len(index):
            # upsampling: try to determine the number of consecutive NaN values to pad
            try:
                # in some cases the data explicitly defines the resolution of events to which the data pertains (timely-beliefs BeliefsDataFrames)
                event_resolution = data.event_resolution if hasattr(data, "event_resolution") and isinstance(data.event_resolution, timedelta) else pd.totimedelta(pd.infer_freq(data.index))
                limit = event_resolution // expected_frequency - 1
            except Exception:
                # no discernible event resolution; index frequency isn't helpful either
                limit = None  # keep padding until the end of index
            if limit is None or limit >= 1:
                return data.reindex(index).fillna(method="pad", limit=limit)
            return data.reindex(index)
        # else downsampling (see below)

        if self.resampling_config is None:
            data = data.resample(
                timedelta_to_pandas_freq_str(expected_frequency)
            ).mean()
        else:
            data_resampler = data.resample(
                timedelta_to_pandas_freq_str(expected_frequency),
                **{
                    k: v
                    for k, v in self.resampling_config.items()
                    if k != "aggregation"
                }
            )
            if "aggregation" not in self.resampling_config:
                data = data_resampler.mean()
            else:
                for agg_name, agg_method in inspect.getmembers(
                    data_resampler, inspect.ismethod
                ):
                    if self.resampling_config["aggregation"] == agg_name:
                        data = agg_method()
                        break
                else:
                    raise IncompatibleModelSpecs(
                        "Cannot find resampling aggregation %s on %s"
                        % (self.resampling_config["aggregation"], data_resampler)
                    )
        return data

    def interpolate_data(self, data) -> pd.Series:
        try:
            data = data.interpolate(**self.interpolation_config)
        except ValueError as e:
            raise IncompatibleModelSpecs(
                "Cannot call interpolate function with arguments %s. %s"
                % (self.interpolation_config, e)
            )
        return data

    def __repr__(self):
        return "%s: <%s>" % (self.__class__.__name__, self.as_dict())


class ObjectSeriesSpecs(SeriesSpecs):
    """
    Spec for a pd.Series object that is being passed in and is stored directly in the specs.
    """

    data: pd.Series

    def __init__(
        self,
        data: pd.Series,
        name: str,
        original_tz: Optional[tzinfo] = None,
        feature_transformation: Optional[ReversibleTransformation] = None,
        post_load_processing: Optional[Transformation] = None,
        resampling_config: Dict[str, Any] = None,
        interpolation_config: Dict[str, Any] = None,
    ):
        super().__init__(
            name,
            original_tz,
            feature_transformation,
            post_load_processing,
            resampling_config,
            interpolation_config,
        )
        if not isinstance(data.index, pd.DatetimeIndex):
            raise IncompatibleModelSpecs(
                "Please provide a DatetimeIndex. Only found %s."
                % type(data.index).__name__
            )
        self.data = data

    def _load_series(self) -> pd.Series:
        if self.post_load_processing is not None:
            return self.post_load_processing.transform_series(self.data)
        return self.data


class DFFileSeriesSpecs(SeriesSpecs):
    """
    Spec for a pandas DataFrame source.
    This class holds the filename, from which we unpickle the data frame, then read the column.
    """

    file_path: str
    time_column: str
    value_column: str

    def __init__(
        self,
        file_path: str,
        time_column: str,
        value_column: str,
        name: str,
        original_tz: Optional[tzinfo] = None,
        feature_transformation: ReversibleTransformation = None,
        post_load_processing: Optional[Transformation] = None,
        resampling_config: Dict[str, Any] = None,
        interpolation_config: Dict[str, Any] = None,
    ):
        super().__init__(
            name,
            original_tz,
            feature_transformation,
            post_load_processing,
            resampling_config,
            interpolation_config,
        )
        self.file_path = file_path
        self.time_column = time_column
        self.value_column = value_column

    def _load_series(self) -> pd.Series:
        df: pd.DataFrame = pd.read_pickle(self.file_path)
        if self.post_load_processing is not None:
            df = self.post_load_processing.transform_dataframe(df)

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df.set_index(self.time_column, drop=True, inplace=True)

        return df[self.value_column]


class CSVFileSeriesSpecs(SeriesSpecs):
    """
    Spec for a CSV file source.
    This class holds the filename, from which we load the data frame, then read the column.
    Any special configuration of pd.read_csv can be given in the `read_csv_config` dict.
    """

    file_path: str
    time_column: str
    value_column: str
    read_csv_config: Dict[str, Any]

    def __init__(
        self,
        file_path: str,
        time_column: str,
        value_column: str,
        name: str,
        read_csv_config: Dict[str, Any] = None,
        original_tz: Optional[tzinfo] = None,
        feature_transformation: ReversibleTransformation = None,
        post_load_processing: Optional[Transformation] = None,
        resampling_config: Dict[str, Any] = None,
        interpolation_config: Dict[str, Any] = None,
    ):
        super().__init__(
            name,
            original_tz,
            feature_transformation,
            post_load_processing,
            resampling_config,
            interpolation_config,
        )
        self.file_path = file_path
        self.time_column = time_column
        self.value_column = value_column
        self.read_csv_config = read_csv_config

    def _load_series(self) -> pd.Series:
        if not os.path.exists(self.file_path):
            raise IncompatibleModelSpecs(
                "Filepath %s does not seem to exist." % self.file_path
            )

        if self.read_csv_config is None:
            df: pd.DataFrame = pd.read_csv(self.file_path)
        else:
            df: pd.DataFrame = pd.read_csv(self.file_path, **self.read_csv_config)

        if self.post_load_processing is not None:
            df = self.post_load_processing.transform_dataframe(df)

        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df.set_index(self.time_column, drop=True, inplace=True)

        return df[self.value_column]


class DBSeriesSpecs(SeriesSpecs):

    """Define how to query a database for time series values.
    This works via a SQLAlchemy query.
    This query should return the needed information for the forecasting pipeline:
    A "datetime" column (which will be set as index of the series) and a "value" column.
    """

    db: Engine
    query: Query

    def __init__(
        self,
        db_engine: Engine,
        query: Query,
        name: str,
        original_tz: Optional[tzinfo] = pytz.utc,  # postgres stores naive datetimes
        feature_transformation: Optional[ReversibleTransformation] = None,
        post_load_processing: Optional[Transformation] = None,
        resampling_config: Dict[str, Any] = None,
        interpolation_config: Dict[str, Any] = None,
    ):
        super().__init__(
            name,
            original_tz,
            feature_transformation,
            post_load_processing,
            resampling_config,
            interpolation_config,
        )
        self.db_engine = db_engine
        self.query = query

    def _load_series(self) -> pd.Series:
        logger.info(
            "Reading %s data from database"
            % self.query.column_descriptions[0]["entity"].__tablename__
        )

        df = pd.DataFrame(
            self.query.all(),
            columns=[col["name"] for col in self.query.column_descriptions],
        )

        self.check_data(df)

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        if self.post_load_processing is not None:
            df = self.post_load_processing.transform_dataframe(df)

        df.set_index("datetime", drop=True, inplace=True)

        return df["value"]

    def check_data(self, df: pd.DataFrame):
        """ Raise error if data is empty or contains nan values.
        Here, other than in load_series, we can show the query, which is quite helpful."""
        if df.empty:
            raise MissingData(
                "No values found in database for the requested %s data. It's no use to continue I'm afraid."
                " Here's a print-out of the database query:\n\n%s\n\n"
                % (
                    self.query.column_descriptions[0]["entity"].__tablename__,
                    render_query(self.query.statement, dialect=postgresql.dialect()),
                )
            )
        if df.isnull().values.any():
            raise NaNData(
                "Nan values found in database for the requested %s data. It's no use to continue I'm afraid."
                " Here's a print-out of the database query:\n\n%s\n\n"
                % (
                    self.query.column_descriptions[0]["entity"].__tablename__,
                    render_query(self.query.statement, dialect=postgresql.dialect()),
                )
            )


class ModelSpecs(object):
    """Describes a model and how it was trained.
    """

    outcome_var: SeriesSpecs
    model_type: Type  # e.g. statsmodels.api.OLS, sklearn.linear_model.LinearRegression, ...
    model_params: dict
    library_name: str
    frequency: timedelta
    horizon: timedelta
    lags: List[int]
    regressors: List[SeriesSpecs]
    # Start of training data set
    start_of_training: datetime
    # End of testing data set
    end_of_testing: datetime
    # This determines the cutoff point between training and testing data
    ratio_training_test_data: float
    # time this model was created, defaults to UTC now
    creation_time: datetime
    model_filename: str
    remodel_frequency: timedelta

    def __init__(
        self,
        outcome_var: Union[SeriesSpecs, pd.Series],
        model: Optional[
            Union[Type, Tuple[Type, dict]]
        ],  # Model class and optionally initialization parameters, can be set later with set_model
        start_of_training: datetime,
        end_of_testing: datetime,
        frequency: timedelta,
        horizon: timedelta,
        lags: List[int] = None,
        regressors: Union[List[SeriesSpecs], List[pd.Series]] = None,
        ratio_training_testing_data=DEFAULT_RATIO_TRAINING_TESTING_DATA,
        remodel_frequency: Union[str, timedelta] = DEFAULT_REMODELING_FREQUENCY,
        model_filename: str = None,
        creation_time: datetime = None,
    ):
        """Create a ModelSpecs instance."""
        self.outcome_var = parse_series_specs(outcome_var, "y")
        if model is not None:
            self.set_model(model)
        self.frequency = frequency
        self.horizon = horizon
        self.lags = lags
        if self.lags is None:
            self.lags = []
        if regressors is None:
            self.regressors = []
        else:
            self.regressors = [
                parse_series_specs(r, "Regressor%d" % (regressors.index(r) + 1))
                for r in regressors
            ]
        self.start_of_training = start_of_training
        self.end_of_testing = end_of_testing
        self.ratio_training_testing_data = ratio_training_testing_data
        # check if training + testing period is compatible with frequency
        if not timedelta_fits_into(
            self.frequency, self.end_of_testing - self.start_of_training
        ):
            raise IncompatibleModelSpecs(
                "Training & testing period (%s to %s) does not fit with frequency (%s)"
                % (self.start_of_training, self.end_of_testing, self.frequency)
            )

        if creation_time is None:
            self.creation_time = tz_aware_utc_now()
        else:
            self.creation_time = creation_time
        self.model_filename = model_filename
        self.remodel_frequency = remodel_frequency

    def as_dict(self):
        return vars(self)

    def __repr__(self):
        return "ModelSpecs: <%s>" % pformat(vars(self))

    def set_model(
        self, model: Union[Type, Tuple[Type, dict]], library_name: Optional[str] = None
    ):
        """
        Set the model. Model is a fittable model class, or a tuple of a model class
        and a dict of mode parameters.
        This function has the extra feature to set the library name (useful for instance
        when your model is custom, based on either statsmodels or sklearn.
        """
        self.model_type = model[0] if isinstance(model, tuple) else model
        self.model_params = model[1] if isinstance(model, tuple) else {}

        known_libraries = ["sklearn", "statsmodels"]
        if library_name is None:
            # guess from class path
            library_name = self.model_type.__module__.split(".")[0]
        if library_name not in (known_libraries):
            raise UnsupportedModel(
                "Library name is '%s',"
                " but should be one out of %s!"
                " Set it via ModelSpecs.set_model." % (library_name, known_libraries)
            )
        self.library_name = library_name


def parse_series_specs(
    specs: Union[SeriesSpecs, pd.Series], name: str = None
) -> SeriesSpecs:
    if isinstance(specs, pd.Series):
        return ObjectSeriesSpecs(specs, name)
    else:
        return specs
