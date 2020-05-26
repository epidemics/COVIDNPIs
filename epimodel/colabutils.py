import re
import pandas as pd

import gspread
from oauth2client.client import GoogleCredentials


def import_gsheet_as_df(url: str):
    """
    Export a DataFrame from a Google Sheets tab. The first row is used
    for column names, and index is set equal to the row number for easy
    cross-referencing.
    """
    sheet_url, _, worksheet_id = url.partition("#gid=")
    worksheet_id = int(worksheet_id or 0)

    client = gspread.authorize(GoogleCredentials.get_application_default())
    spreadsheet = client.open_by_url(sheet_url)
    worksheet = _get_worksheet_by_id(spreadsheet, worksheet_id)
    values = worksheet.get_all_values()
    return pd.DataFrame(values[1:], columns=values[0], index=range(2, len(values) + 1))


def _get_worksheet_by_id(spreadsheet, worksheet_id):
    """ gspread does not provide this function, so I added it """
    for worksheet in spreadsheet.worksheets():
        if worksheet.id == worksheet_id:
            return worksheet
    raise gspread.WorksheetNotFound(f"id {worksheet_id}")


def get_csv_or_sheet(path):
    if re.search("\.csv$", str(path), re.IGNORECASE):
        return pd.read_csv(path)
    elif re.match("https://docs.google.com/spreadsheets/[\w/?#=]+$", str(path)):
        return import_gsheet_as_df(path)
    else:
        return ValueError(f"{path!r} not recognized as CSV or Google sheet")
