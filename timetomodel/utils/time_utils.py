from typing import List
from datetime import datetime, timedelta

import pytz
from pandas.tseries.frequencies import to_offset


def tz_aware_utc_now() -> datetime:
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def get_most_recent_quarter(dt: datetime = None) -> datetime:
    if dt is None:
        dt = (
            tz_aware_utc_now()
        )  # TODO: maybe we should be able to configure a timezone?
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
    """Translate a timedelta to a frequency name string used by Pandas.

    Unlike timedelta objects, calendar rules matter here, so safest is to pass UTC or naive datetimes.
    See also https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    """
    return to_offset(resolution).freqstr


def naive_utc_from(dt: datetime) -> datetime:
    """Return a naive datetime, that is localised to UTC if it has a timezone."""
    if not hasattr(dt, "tzinfo") or dt.tzinfo is None:
        # let's hope this is the UTC time you expect
        return dt
    else:
        return dt.astimezone(pytz.utc).replace(tzinfo=None)


def round_datetime(dt, by_seconds=60):
    """Round a datetime by some number of seconds. Can be made nicer by e.g. Pendulum"""
    dt_naive = naive_utc_from(dt)
    seconds = (dt_naive - dt_naive.min).total_seconds()
    rounding = (seconds + by_seconds / 2) // by_seconds * by_seconds
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def timedelta_fits_into(short_td, long_td):
    """Return true if multiple short timedeltas fit exactly into long timedelta"""
    return long_td.total_seconds() % short_td.total_seconds() == 0
