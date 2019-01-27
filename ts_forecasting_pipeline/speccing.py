from typing import Dict, List, Optional, Tuple, Type, Union
from datetime import datetime, timedelta, tzinfo
from pprint import pformat
import json
import warnings
import logging

import pytz
import dateutil.parser
import numpy as np
import pandas as pd
from statsmodels.base.transform import BoxCox
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Query
from sqlalchemy.dialects import postgresql

from ts_forecasting_pipeline.utils.debug_utils import render_query
from ts_forecasting_pipeline.utils.time_utils import (
    tz_aware_utc_now,
    timedelta_to_pandas_freq_str,
    timedelta_fits_into,
)
from ts_forecasting_pipeline.exceptions import IncompatibleModelSpecs


DEFAULT_RATIO_TRAINING_TESTING_DATA = 2 / 3
DEFAULT_REMODELING_FREQUENCY = timedelta(days=1)

np.seterr(all="warn")
warnings.filterwarnings("error", message="invalid value encountered in power")

logger = logging.getLogger(__name__)


class SeriesSpecs(object):
    """Describes a time series (e.g. a pandas Series).
    In essence, a column in the regression frame, filled with numbers.

    Using this class, the column will be filled with NaN values.

    If you have data to be loaded in automatically, you should be using one of the subclasses, which allow to describe
    or pass in an actual data source to be loaded.

    When dealing with columns, our code should usually refer to this superclass so it does not need to care
    which kind of data source it is dealing with.
    """

    # The name in the resulting feature frame, and possibly in the saved model specs (named by outcome var)
    name: str
    # The name in the data source, if source is a pandas DataFrame or database Table - if None, the name will be tried
    column: Optional[str]
    # timezone of the data - useful when de-serializing (e.g. pandas serialises to UTC)
    original_tz: tzinfo

    def __init__(self, name: str, column: str = None, original_tz: tzinfo = None):
        self.name = name
        self.column = column
        if self.column is None:
            self.column = name
        self.original_tz = original_tz
        self.__series_type__ = self.__class__.__name__

    def as_dict(self):
        return vars(self)

    def load_series(self, expected_frequency: timedelta = None) -> pd.Series:
        return pd.Series()

    def __repr__(self):
        return "%s: <%s>" % (self.__class__.__name__, self.as_dict())


class ObjectSeriesSpecs(SeriesSpecs):
    """
    Spec for a pd.Series object that is being passed in and is stored directly in the specs.
    The data is not mutatable after creation.
    Note: The column argument is not used, as the series has only one column.
    """

    data: pd.Series

    def __init__(
        self, data: pd.Series, name: str, column: str = None, original_tz: tzinfo = None
    ):
        super().__init__(name, None, original_tz)
        self.data = data
        self.original_tz = data.index.tzinfo
        if self.original_tz is None:
            self.original_tz = pytz.utc

    def load_series(self, expected_frequency: timedelta = None) -> pd.Series:

        if expected_frequency is not None:
            return self.data.resample(
                timedelta_to_pandas_freq_str(expected_frequency)
            ).mean()

        return self.data


class DFFileSeriesSpecs(SeriesSpecs):
    """
    Spec for a pandas DataFrame source.
    This class holds the filename, from which we unpickle the data frame, then read the column.
    """

    file_path: str

    def __init__(
        self, file_path: str, name: str, column: str = None, original_tz: tzinfo = None
    ):
        super().__init__(name, column, original_tz)
        self.file_path = file_path

    def load_series(self, expected_frequency: timedelta = None) -> pd.Series:
        df: pd.DataFrame = pd.read_pickle(self.file_path)
        if df.index.tzinfo is None:
            self.original_tz = pytz.utc
            df.index = df.index.tz_localize(self.original_tz)
        else:
            self.original_tz = df.index.tzinfo

        if expected_frequency is not None:
            return (
                df[self.column]
                .resample(timedelta_to_pandas_freq_str(expected_frequency))
                .mean()
            )

        return df[self.column]


class DBSeriesSpecs(SeriesSpecs):

    """Define how to query a database for time series values.
    This works via a SQLAlchemy query.
    This query should return the needed information for the forecasting pipeline:
    A "datetime" column (which will be set as index) and the values column (named by name or column,
    see SeriesSpecs.__init__, defaults to "value"). For example:
    TODO: show an example"""

    db: Engine
    query: Query

    def __init__(
        self,
        db_engine: Engine,
        query: Query,
        name: str = "value",
        column: str = None,
        original_tz: tzinfo = pytz.utc,  # postgres stores naive datetimes
    ):
        super().__init__(name, column, original_tz)
        self.db_engine = db_engine
        self.query = query

    def load_series(self, expected_frequency: timedelta = None) -> pd.Series:
        logger.info(
            "Reading %s data from database"
            % self.query.column_descriptions[0]["entity"].__tablename__
        )
        """
        from sqlalchemy.dialects import postgresql
        cq = self.query.statement.compile(dialect=postgresql.dialect())
        logger.debug("Query: %s" % str(cq))
        logger.debug("Params: %s" % str(cq.params))
        """

        series_orig = pd.DataFrame(
            self.query.all(),
            columns=[col["name"] for col in self.query.column_descriptions],
        )
        series_orig["datetime"] = pd.to_datetime(series_orig["datetime"], utc=True)

        # Raise error if data is empty or contains nan values
        if series_orig.empty:
            raise ValueError(
                "No values found in database for the requested %s data. It's no use to continue I'm afraid."
                " Here's a print-out of the database query:\n\n%s\n\n"
                % (
                    self.query.column_descriptions[0]["entity"].__tablename__,
                    render_query(self.query.statement, dialect=postgresql.dialect()),
                )
            )
        if series_orig.isnull().values.any():
            raise ValueError(
                "Nan values found in database for the requested %s data. It's no use to continue I'm afraid."
                " Here's a print-out of the database query:\n\n%s\n\n"
                % (
                    self.query.column_descriptions[0]["entity"].__tablename__,
                    render_query(self.query.statement, dialect=postgresql.dialect()),
                )
            )

        # Keep the most recent observation
        series = (
            series_orig.sort_values(by=["horizon"], ascending=True)
            .drop_duplicates(subset=["datetime"], keep="first")
            .sort_values(by=["datetime"])
        )
        series.set_index("datetime", drop=True, inplace=True)

        if series.index.tzinfo is None:
            if self.original_tz is not None:
                series.index = series_orig.index.tz_localize(self.original_tz)
        else:
            series.index = series.index.tz_convert(self.original_tz)

        if expected_frequency is not None:
            series = series_orig.resample(
                timedelta_to_pandas_freq_str(expected_frequency)
            ).mean()

        return series["value"]


class Transformation(object):
    """Base class for transformations.
    Initialise with your custom transformation parameters and define custom functions to transform and back-transform.
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

    def transform(
        self, x: Union[np.array, pd.DataFrame, pd.Series]
    ) -> Tuple[np.array, dict]:
        params = {}
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = np.array(x.values, dtype=np.float64)

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


class ModelSpecs(object):
    """Describes a model and how it was trained.
    """

    outcome_var: SeriesSpecs
    model_type: Type  # e.g. statsmodels.api.OLS, sklearn.linear_model.LinearRegression, ...
    model_params: dict
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
    # Custom transformation to perform on the outcome data. Called after relevant SeriesSpecs were resolved.
    transformation: Transformation
    # Custom transformation to perform on each regressor data. Called after relevant SeriesSpecs were resolved.
    regressor_transformation: Dict[str, Transformation]

    def __init__(
        self,
        outcome_var: Union[str, SeriesSpecs, pd.Series],
        model: Union[
            Type, Tuple[Type, dict]
        ],  # Model class and optionally initialization parameters
        start_of_training: Union[str, datetime],
        end_of_testing: Union[str, datetime],
        frequency: timedelta,
        horizon: timedelta,
        lags: List[int] = None,
        regressors: Union[List[str], List[SeriesSpecs], List[pd.Series]] = None,
        ratio_training_testing_data=DEFAULT_RATIO_TRAINING_TESTING_DATA,
        remodel_frequency: Union[str, timedelta] = DEFAULT_REMODELING_FREQUENCY,
        model_filename: str = None,
        creation_time: Union[str, datetime] = None,
        transformation: Transformation = None,
        regressor_transformation: Dict[str, Transformation] = None,
    ):
        """Create a ModelSpecs instance. Accepts all parameters as string (besides transform - TODO) for
         deserialization support (JSON strings for all parameters which are not natively JSON-parseable,)"""
        self.outcome_var = parse_series_specs(outcome_var, "y")
        self.model_type = model[0] if isinstance(model, tuple) else model
        self.model_params = model[1] if isinstance(model, tuple) else {}
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
        if isinstance(start_of_training, str):
            self.start_of_training = dateutil.parser.parse(start_of_training)
        else:
            self.start_of_training = start_of_training
        if isinstance(end_of_testing, str):
            self.end_of_testing = dateutil.parser.parse(end_of_testing)
        else:
            self.end_of_testing = end_of_testing
        self.ratio_training_testing_data = ratio_training_testing_data
        # check if training+testing period is compatible with frequency
        if not timedelta_fits_into(
            self.frequency, self.end_of_testing - self.start_of_training
        ):
            raise IncompatibleModelSpecs(
                "Training & testing period (%s to %s) does not fit with frequency (%s)"
                % (self.start_of_training, self.end_of_testing, self.frequency)
            )

        if isinstance(creation_time, str):
            self.creation_time = dateutil.parser.parse(creation_time)
        elif creation_time is None:
            self.creation_time = tz_aware_utc_now()
        else:
            self.creation_time = creation_time
        self.model_filename = model_filename
        if isinstance(remodel_frequency, str):
            self.remodel_frequency = timedelta(
                days=int(remodel_frequency) / 60 / 60 / 24
            )
        else:
            self.remodel_frequency = remodel_frequency
        self.transformation = transformation
        self.regressor_transformation = regressor_transformation

    def as_dict(self):
        return vars(self)

    def __repr__(self):
        return "ModelSpecs: <%s>" % pformat(vars(self))


def parse_series_specs(
    specs: Union[str, SeriesSpecs, pd.Series], name: str = None
) -> SeriesSpecs:
    if isinstance(specs, str):
        return load_series_specs_from_json(specs)
    elif isinstance(specs, pd.Series):
        return ObjectSeriesSpecs(specs, name)
    else:
        return specs


def load_series_specs_from_json(s: str) -> SeriesSpecs:
    json_repr = json.loads(s)
    series_class = globals()[json_repr["__series_type__"]]
    if series_class == ObjectSeriesSpecs:
        # load pd.Series from string, will be UTC-indexed, so apply original_tz
        json_repr["data"] = pd.read_json(
            json_repr["data"], typ="series", convert_dates=True
        )
        json_repr["data"].index = json_repr["data"].index.tz_localize(
            json_repr["original_tz"]
        )
    return series_class(
        **{k: v for k, v in json_repr.items() if not k.startswith("__")}
    )
