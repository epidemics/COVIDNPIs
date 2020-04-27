import io
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


def test_read_csv_smart(regions):
    df = utils.read_csv_smart(io.StringIO("code,X,Y\n"), regions)
    assert df.index.name == "Code"
    assert "Code" not in df.columns
    assert "code" not in df.columns

    df = utils.read_csv_smart(io.StringIO("A,name,X,y,date\n"), regions)
    assert df.index.names == ["Code", "Date"]
    assert "Code" not in df.columns
    assert "code" not in df.columns
    assert "Date" not in df.columns
    assert "date" not in df.columns

    df = utils.read_csv_smart(
        io.StringIO("A,name,date\n42,CZ,2020-01-02\n43,CZ,2020-01-01"), regions
    )
    assert df["A"].dtype == int

    df = utils.read_csv_smart(io.StringIO("A,name\n42,CZ\n43,NA"), regions)
    assert df.loc["NA", "A"] == 43
