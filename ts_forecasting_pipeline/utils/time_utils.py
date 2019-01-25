from typing import List
from datetime import datetime, timedelta

import pytz
from pandas.tseries.frequencies import to_offset


def tz_aware_utc_now() -> datetime:
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def get_most_recent_quarter(dt: datetime = None) -> datetime:
    if dt is None:
        dt = tz_aware_utc_now()  # TODO: maybe we should be able to configure a timezone?
    return dt.replace(minute=dt.minute - (dt.minute % 15), second=0, microsecond=0)


def get_closest_quarter(dt: datetime = None) -> datetime:
    cdt = get_most_recent_quarter(dt)
    if dt - cdt > timedelta(minutes=7, seconds=30):
        # round up
        return cdt + timedelta(minutes=15)
    return cdt


def day_lags(lags):
    """Translate day lags into 15-minute lags"""
    return [l * 96 for l in lags]


def to_15_min_lags(lags: List[timedelta]) -> List[int]:
    """Translate timedelta lags into 15-minute lags."""
    return [int(lag.days * 96 + lag.seconds / 900) for lag in lags]


def timedelta_to_pandas_freq_str(resolution: timedelta) -> str:
    return to_offset(resolution).freqstr
