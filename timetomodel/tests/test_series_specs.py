from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
import pytz

from timetomodel.speccing import ObjectSeriesSpecs, CSVFileSeriesSpecs
from timetomodel.transforming import Transformation
from timetomodel.tests.utils import MyMultiplicationTransformation
from timetomodel.exceptions import MissingData, NaNData, IncompatibleModelSpecs


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
            dt + timedelta(minutes=30)
        ]
        == 3
    )


def test_load_series_with_expected_time_window():
    dt = datetime(2019, 1, 29, 15, 15, tzinfo=pytz.utc)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
    )
    assert (
        s.load_series(
            expected_frequency=timedelta(minutes=15),
            check_time_window=(dt, dt + timedelta(minutes=30)),
        ).loc[dt + timedelta(minutes=30)]
        == 3
    )


def test_load_series_with_larger_expected_time_window():
    dt = datetime(2019, 1, 29, 15, 15, tzinfo=pytz.utc)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
    )
    with pytest.raises(MissingData) as e_info:
        s.load_series(
            expected_frequency=timedelta(minutes=15),
            check_time_window=(dt - timedelta(minutes=15), dt + timedelta(minutes=45)),
        )
    assert "starts too late" in str(e_info.value)
    assert "ends too early" in str(e_info.value)


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


def test_load_series_with_non_existing_custom_frequency_resampling():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, 2, 3],
        ),
        name="mydata",
        resampling_config={"aggregation": "GGG"},
    )

    with pytest.raises(IncompatibleModelSpecs) as e_info:
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
        resampling_config={"aggregation": "sum"},
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
    with pytest.raises(NaNData) as e_info:
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
    with pytest.raises(NaNData) as e_info:
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
        feature_transformation=MyMultiplicationTransformation(factor=11),
    )
    assert (
        s.load_series(expected_frequency=timedelta(minutes=15)).loc[
            dt + timedelta(minutes=15)
        ]
        == 2
    )
    assert (
        s.load_series(
            expected_frequency=timedelta(minutes=15), transform_features=True
        ).loc[dt + timedelta(minutes=15)]
        == 2 * 11
    )


def test_load_series_from_csv_with_post_load_processing(tmpdir):

    highscore_data = """Time,Name,Highscore,
2019-02-05T12:57:00,Mel,8,
2019-02-05T10:30:00,Jack,5,
2019-02-05T11:36:00,David,10,
2019-02-05T10:34:00,Peter,6,
2019-02-05T09:11:00,David,5,
2019-02-05T11:17:00,Ryan,9,
2019-02-05T12:27:00,Ryan,9,
"""
    f = tmpdir.join("highscore.csv")
    f.write(highscore_data)

    def to_hour(dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    class BestHighscorePerHour(Transformation):
        def transform_dataframe(self, df):
            df["Time"] = pd.to_datetime(df["Time"], utc=True)
            df["Time"] = df["Time"].apply(to_hour)

            return (
                df.sort_values(by=["Highscore"], ascending=False)
                .drop_duplicates(subset=["Time"], keep="first")
                .sort_values(by=["Time"])
            )

    s = CSVFileSeriesSpecs(
        file_path=f.realpath(),
        time_column="Time",
        value_column="Highscore",
        post_load_processing=BestHighscorePerHour(),
        name="mydata",
        feature_transformation=MyMultiplicationTransformation(factor=100),
    )

    data = s.load_series(expected_frequency=timedelta(hours=1))

    assert data[datetime(2019, 2, 5, 9)] == 5
    assert data[datetime(2019, 2, 5, 10)] == 6
    assert data[datetime(2019, 2, 5, 11)] == 10
    assert data[datetime(2019, 2, 5, 12)] == 9

    data = s.load_series(expected_frequency=timedelta(hours=1), transform_features=True)

    assert data[datetime(2019, 2, 5, 9)] == 500
    assert data[datetime(2019, 2, 5, 10)] == 600
    assert data[datetime(2019, 2, 5, 11)] == 1000
    assert data[datetime(2019, 2, 5, 12)] == 900


def test_load_series_with_non_existing_interpolation():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, np.nan, 3],
        ),
        name="mydata",
        interpolation_config={"method": "GGG"},
    )

    with pytest.raises(IncompatibleModelSpecs) as e_info:
        s.load_series(expected_frequency=timedelta(minutes=15))
    assert "Cannot call interpolate function with arguments {'method': 'GGG'}" in str(
        e_info.value
    )


def test_load_series_with_interpolation():
    dt = datetime(2019, 1, 29, 15, 15)
    s = ObjectSeriesSpecs(
        data=pd.Series(
            index=pd.date_range(dt, dt + timedelta(minutes=30), freq="15T"),
            data=[1, np.nan, 3],
        ),
        name="mydata",
        interpolation_config={"method": "time"},
    )

    series = s.load_series(expected_frequency=timedelta(minutes=15))
    assert len(series) == 3
    assert series[1] == 2  # the interpolated value
