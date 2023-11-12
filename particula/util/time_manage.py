"""Time management utilities."""

from datetime import datetime
import pytz


def time_str_to_epoch(
        time: str,
        time_format: str,
        timezone_identifier: str,
) -> float:
    """Convert to UTC (epoch) timezone from all inputs. Using pytz library,
    which implements the Olson time zone database. tz identifiers are strings
    from the database.
    See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    for a list of time zones.

    Parameters:
    -----------
    time : float (single value no arrays)
        Epoch time in seconds.
    time_format : str
        The format of the time string. See
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        for a list of format codes.
    timezone_identifier : str
        The time zone identifier for the current time zone.

    Returns:
    --------
    new_time : float
        The float time in the new time zone.
    # """
    time_zone = pytz.timezone(timezone_identifier)
    time_obj = datetime.strptime(time, time_format)

    time_obj = time_zone.localize(time_obj)

    return time_obj.timestamp()
