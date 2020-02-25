from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytz

from timetomodel.utils.time_utils import (
    get_closest_quarter,
    get_most_recent_quarter,
    round_datetime,
    timedelta_fits_into,
    timedelta_to_pandas_freq_str,
)


def test_find_quarter():
    dt = datetime(2018, 1, 26, 14, 40).astimezone(pytz.timezone("Europe/Amsterdam"))
    recent = get_most_recent_quarter(dt)
    assert recent.day == dt.day
    assert recent.hour == dt.hour
    assert recent.minute == 30
    closest = get_closest_quarter(dt)
    assert closest.day == dt.day
    assert closest.hour == dt.hour
    assert closest.minute == 45


def test_timedelta_to_pd_freq_str():
    assert timedelta_to_pandas_freq_str(timedelta(seconds=5)) == "5S"
    assert timedelta_to_pandas_freq_str(timedelta(minutes=2)) == "2T"
    assert timedelta_to_pandas_freq_str(timedelta(minutes=15)) == "15T"
    assert timedelta_to_pandas_freq_str(timedelta(hours=1)) == "H"
    assert timedelta_to_pandas_freq_str(timedelta(hours=2)) == "2H"
    assert timedelta_to_pandas_freq_str(timedelta(hours=26)) == "26H"
    assert timedelta_to_pandas_freq_str(timedelta(days=1, hours=2)) == "26H"
    assert timedelta_to_pandas_freq_str(timedelta(days=1)) == "D"


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2018, 1, 26, 14, 40),
        datetime(2018, 1, 26, 14, 40, tzinfo=pytz.utc),
        pd.Timestamp(datetime(2018, 1, 26, 14, 40)),
        pd.Timestamp(datetime(2018, 1, 26, 14, 40, tzinfo=pytz.utc)),
    ],
)
def test_round_time_by_hour(dt):
    round_to_hour = round_datetime(dt, by_seconds=60 * 60)
    assert round_to_hour.day == dt.day
    assert round_to_hour.hour == 15
    assert round_to_hour.minute == 00


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2018, 1, 26, 14, 40),
        datetime(2018, 1, 26, 14, 40, tzinfo=pytz.utc),
        pd.Timestamp(datetime(2018, 1, 26, 14, 40)),
        pd.Timestamp(datetime(2018, 1, 26, 14, 40, tzinfo=pytz.utc)),
    ],
)
def test_round_time_by_15min(dt):
    round_to_hour = round_datetime(dt, by_seconds=60 * 15)
    assert round_to_hour.day == dt.day
    assert round_to_hour.hour == 14
    assert round_to_hour.minute == 45


def test_timedelta_fits():
    assert not timedelta_fits_into(timedelta(seconds=11), timedelta(minutes=4))
    assert timedelta_fits_into(timedelta(minutes=10), timedelta(hours=2))
    assert timedelta_fits_into(timedelta(minutes=3), timedelta(hours=1))
    assert timedelta_fits_into(timedelta(minutes=15), timedelta(hours=1))
    assert timedelta_fits_into(timedelta(minutes=15), timedelta(days=4))
    assert timedelta_fits_into(timedelta(hours=12), timedelta(days=2))
    assert not timedelta_fits_into(timedelta(hours=16), timedelta(days=3))
    assert timedelta_fits_into(timedelta(hours=16), timedelta(days=6))
    assert timedelta_fits_into(timedelta(minutes=15), timedelta(weeks=1))
    assert not timedelta_fits_into(timedelta(minutes=11), timedelta(hours=1))
    assert timedelta_fits_into(timedelta(minutes=11), timedelta(hours=11))
