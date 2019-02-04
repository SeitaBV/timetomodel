from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np

from ts_forecasting_pipeline.speccing import ObjectSeriesSpecs
from ts_forecasting_pipeline.tests.utils import MyMultiplicationTransformation


def test_load_series_without_datetime_index():
    with pytest.raises(Exception) as e_info:
        s = ObjectSeriesSpecs(data=pd.Series([1, 2, 3]), name="mydata")
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "DatetimeIndex" in str(e_info.value)


def test_load_series():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
    )
    assert (
        s.load_series(expected_frequency=timedelta(minutes=15)).loc[
            dt + timedelta(minutes=15)
        ]
        == 2
    )


def test_load_series_with_frequency_resampling():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
    )
    series = s.load_series(expected_frequency=timedelta(hours=1))
    assert len(series) == 1
    assert series[0] == 2  # the mean


def test_load_series_with_not_existing_custom_frequency_resampling():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
        resampling_config={"aggregation": "GGG"}
    )

    with pytest.raises(Exception) as e_info:
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "Cannot find resampling aggregation GGG" in str(e_info.value)


def test_load_series_with_custom_frequency_resampling():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
        resampling_config={"aggregation": "sum"}
    )

    series = s.load_series(expected_frequency=timedelta(hours=1))
    assert len(series) == 1
    assert series[0] == 6  # the sum


def test_load_series_without_data():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[np.nan, np.nan, np.nan],
        ),
        name="mydata",
    )
    with pytest.raises(Exception) as e_info:
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "Nan values" in str(e_info.value)


def test_load_series_with_missing_data():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, np.nan, 3],
        ),
        name="mydata",
    )
    with pytest.raises(Exception) as e_info:
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "Nan values" in str(e_info.value)


def test_load_series_with_transformation():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
        transformation=MyMultiplicationTransformation(factor=11)
    )
    assert (
        s.load_series(expected_frequency=timedelta(minutes=15)).loc[
            dt + timedelta(minutes=15)
        ]
        == 2 * 11
    )


