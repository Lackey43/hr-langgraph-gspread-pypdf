import gspread
import os
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
load_dotenv()

SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT_FILE")


def append_to_google_sheet(values: list[list], column : str):
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ]


    credentials = Credentials.from_service_account_file(
        SERVICE_ACCOUNT,
        scopes=scopes
    )


    gc = gspread.authorize(credentials)
    spreadsheet = gc.open_by_key("1OsGCgyeG76Og5sORQTi4N3ha7nxmYKnzqleaO5o13C4")
    sheet = spreadsheet.get_worksheet_by_id("469343244")
    index = sheet.get_all_values()
    sheet.append_row(values,table_range=f"{column}{len(index)}")
    return 1
