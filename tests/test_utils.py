from datetime import date, datetime, tzinfo

from epimodel import utils


def test_utc_date():
    assert utils.utc_date("2020-01-02").isoformat() == "2020-01-02T00:00:00+00:00"
    assert utils.utc_date("3/4/19").isoformat() == "2019-03-04T00:00:00+00:00"
    assert (
        utils.utc_date("2020-01-02 23:59:00-10:00").isoformat()
        == "2020-01-02T00:00:00+00:00"
    )
    assert utils.utc_date(date(2020, 5, 6)).isoformat() == "2020-05-06T00:00:00+00:00"
    assert (
        utils.utc_date(datetime(2020, 5, 6, 23, 59).astimezone()).isoformat()
        == "2020-05-06T00:00:00+00:00"
    )
    assert (
        utils.utc_date(datetime(2020, 5, 6, 0, 0).astimezone()).isoformat()
        == "2020-05-06T00:00:00+00:00"
    )
