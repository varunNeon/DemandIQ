"""
Known calendar events for DemandIQ.

Holidays are known in advance, so they are legitimate inputs for future
forecast days. The list is a simplified set of Ecuadorian national holidays,
matching the store locations used by the synthetic demo data.
"""

from __future__ import annotations

import pandas as pd

KNOWN_HOLIDAYS: frozenset[str] = frozenset({
    # 2013
    "2013-01-01", "2013-02-11", "2013-04-01", "2013-05-01", "2013-05-24", "2013-07-24",
    "2013-08-10", "2013-10-09", "2013-11-02", "2013-11-03", "2013-12-25",
    # 2014
    "2014-01-01", "2014-03-03", "2014-04-18", "2014-05-01", "2014-05-24", "2014-07-24",
    "2014-08-10", "2014-10-09", "2014-11-02", "2014-11-03", "2014-12-25",
    # 2015
    "2015-01-01", "2015-02-16", "2015-04-03", "2015-05-01", "2015-05-24", "2015-07-24",
    "2015-08-10", "2015-10-09", "2015-11-02", "2015-11-03", "2015-12-25",
    # 2016
    "2016-01-01", "2016-02-08", "2016-03-25", "2016-05-01", "2016-05-24", "2016-07-24",
    "2016-08-10", "2016-10-09", "2016-11-02", "2016-11-03", "2016-12-25",
})


def future_calendar(last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """
    Build the known exogenous inputs for the `horizon` days after `last_date`.

    Holidays come from the known calendar. Promotions are unknown for the
    future, so they are set to 0; this is a stated assumption, not a fact.
    """
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({
        "date": dates,
        "onpromotion": 0,
        "is_holiday": [int(d.strftime("%Y-%m-%d") in KNOWN_HOLIDAYS) for d in dates],
    })
