from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np

from ts_forecasting_pipeline.speccing import ObjectSeriesSpecs


def test_load_objects_without_datetime_index():
    with pytest.raises(Exception) as e_info:
        s = ObjectSeriesSpecs(data=pd.Series([1, 2, 3]), name="mydata")
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "DatetimeIndex" in str(e_info.value)


def test_load_objects():
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


def test_load_objects_with_frequency_resampling():
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


def test_load_objects_with_custom_frequency_resampling():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
    )

    with pytest.raises(Exception) as e_info:
        s.load_series(expected_frequency=timedelta(hours=1), resample_config={"aggregation": "GGG"})
    assert "Cannot find resampling aggregation GGG" in str(e_info.value)

    series = s.load_series(expected_frequency=timedelta(hours=1), resample_config={"aggregation": "sum"})
    assert len(series) == 1
    assert series[0] == 6  # the sum


def test_load_objects_without_data():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[],
        ),
        name="mydata",
    )
    with pytest.raises(Exception) as e_info:
        s.load_series(expected_frequency=timedelta(hours=1))
    assert "No values" in str(e_info.value)


def test_load_objects_without_data():
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


