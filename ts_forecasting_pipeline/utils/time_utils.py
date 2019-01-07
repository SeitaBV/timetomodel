from typing import List
from datetime import datetime, timedelta
import pytz


def tz_aware_utc_now() -> datetime:
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def get_most_recent_quarter(dt: datetime = None) -> datetime:
    if dt is None:
        dt = tz_aware_utc_now()
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
