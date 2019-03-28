from datetime import time
from cn_stock_holidays.data import get_cached
from pandas import Timestamp, date_range, DatetimeIndex
import pytz
from zipline.utils.memoize import remember_last, lazyval
import warnings

from zipline.utils.calendars import TradingCalendar
# from zipline.utils.calendars.trading_calendar import days_at_time, NANOS_IN_MINUTE
import numpy as np
import pandas as pd

NANOS_IN_MINUTE = 60000000000
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)


def days_at_time(days, t, tz, day_offset=0):
    """
    Create an index of days at time ``t``, interpreted in timezone ``tz``.
    The returned index is localized to UTC.
    Parameters
    ----------
    days : DatetimeIndex
        An index of dates (represented as midnight).
    t : datetime.time
        The time to apply as an offset to each day in ``days``.
    tz : pytz.timezone
        The timezone to use to interpret ``t``.
    day_offset : int
        The number of days we want to offset @days by
    Examples
    --------
    In the example below, the times switch from 13:45 to 12:45 UTC because
    March 13th is the daylight savings transition for US/Eastern.  All the
    times are still 8:45 when interpreted in US/Eastern.
    >>> import pandas as pd; import datetime; import pprint
    >>> dts = pd.date_range('2016-03-12', '2016-03-14')
    >>> dts_at_845 = days_at_time(dts, datetime.time(8, 45), 'US/Eastern')
    >>> pprint.pprint([str(dt) for dt in dts_at_845])
    ['2016-03-12 13:45:00+00:00',
     '2016-03-13 12:45:00+00:00',
     '2016-03-14 12:45:00+00:00']
    """
    if len(days) == 0:
        return days

    # Offset days without tz to avoid timezone issues.
    days = DatetimeIndex(days).tz_localize(None)
    delta = pd.Timedelta(
        days=day_offset,
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
    )
    return (days + delta).tz_localize(tz).tz_convert('UTC')


# lunch break for shanghai and shenzhen exchange
lunch_break_start = time(11, 30)
lunch_break_end = time(13, 1)

start_default = pd.Timestamp('1990-12-19', tz='UTC')
end_base = pd.Timestamp('today', tz='UTC')
end_default = end_base + pd.Timedelta(days=365)


class SHSZExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for Shanghai and Shenzhen (China Market)
    Open Time 9:31 AM, Asia/Shanghai
    Close Time 3:00 PM, Asia/Shanghai

    One big difference between china and us exchange is china exchange has a lunch break , so I handle it

    Sample Code in ipython:

    > from zipline.utils.calendars import *
    > from cn_stock_holidays.zipline.exchange_calendar_shsz import SHSZExchangeCalendar
    > register_calendar("SHSZ", SHSZExchangeCalendar(), force=True)
    > c=get_calendar("SHSZ")

    for the guy need to keep updating about holiday file, try to add `cn-stock-holiday-sync` command to crontab
    """

    def __init__(self, start=start_default, end=end_default):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _all_days = date_range(start, end, freq=self.day, tz='UTC')

        self._lunch_break_starts = days_at_time(_all_days, lunch_break_start, self.tz, 0)
        self._lunch_break_ends = days_at_time(_all_days, lunch_break_end, self.tz, 0)

        TradingCalendar.__init__(self, start=start_default, end=end_default)

        self.schedule = pd.DataFrame(
            index=_all_days,
            columns=['market_open', 'market_close', 'lunch_break_start', 'lunch_break_end'],
            data={
                'market_open': self._opens,
                'market_close': self._closes,
                'lunch_break_start': self._lunch_break_starts,
                'lunch_break_end': self._lunch_break_ends
            },
            dtype='datetime64[ns]',
        )

    @property
    def name(self):
        return "SHSZ"

    @property
    def tz(self):
        return pytz.timezone("Asia/Shanghai")

    @property
    def open_times(self):
        return time(9, 31)

    @property
    def close_times(self):
        return time(15, 0)

    @property
    def adhoc_holidays(self):
        return [Timestamp(t, tz=pytz.UTC) for t in get_cached(use_list=True)]

    @lazyval
    def _minutes_per_session(self):
        diff = (
               self.schedule.lunch_break_start.values.astype('datetime64[m]') - self.schedule.market_open.values.astype('datetime64[m]')) + (
               self.schedule.market_close.values.astype('datetime64[m]') - self.schedule.lunch_break_end.values.astype('datetime64[m]'))
        diff = diff.astype(np.int64)
        return diff + 2

    @property
    @remember_last
    def all_minutes(self):
        """
            Returns a DatetimeIndex representing all the minutes in this calendar.
        """
        opens_in_ns = \
            self._opens.values.astype('datetime64[ns]')

        closes_in_ns = \
            self._closes.values.astype('datetime64[ns]')

        lunch_break_start_in_ns = \
            self._lunch_break_starts.values.astype('datetime64[ns]')
        lunch_break_ends_in_ns = \
            self._lunch_break_ends.values.astype('datetime64[ns]')

        deltas_before_lunch = lunch_break_start_in_ns - opens_in_ns
        deltas_after_lunch = closes_in_ns - lunch_break_ends_in_ns

        daily_before_lunch_sizes = (deltas_before_lunch / NANOS_IN_MINUTE) + 1
        daily_after_lunch_sizes = (deltas_after_lunch / NANOS_IN_MINUTE) + 1

        daily_sizes = daily_before_lunch_sizes + daily_after_lunch_sizes

        num_minutes = np.sum(daily_sizes).astype(np.int64)

        # One allocation for the entire thing. This assumes that each day
        # represents a contiguous block of minutes.
        all_minutes = np.empty(num_minutes, dtype='datetime64[ns]')

        idx = 0
        for day_idx, size in enumerate(daily_sizes):
            # lots of small allocations, but it's fast enough for now.

            # size is a np.timedelta64, so we need to int it
            size_int = int(size)

            before_lunch_size_int = int(daily_before_lunch_sizes[day_idx])
            after_lunch_size_int = int(daily_after_lunch_sizes[day_idx])

            all_minutes[idx:(idx + before_lunch_size_int)] = \
                np.arange(
                    opens_in_ns[day_idx],
                    lunch_break_start_in_ns[day_idx] + NANOS_IN_MINUTE,
                    NANOS_IN_MINUTE
                )

            all_minutes[(idx + before_lunch_size_int):(idx + size_int)] = \
                np.arange(
                    lunch_break_ends_in_ns[day_idx],
                    closes_in_ns[day_idx] + NANOS_IN_MINUTE,
                    NANOS_IN_MINUTE
                )

            idx += size_int
        return DatetimeIndex(all_minutes).tz_localize("UTC")
