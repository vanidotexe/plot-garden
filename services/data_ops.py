import pandas as pd
from PyQt6.QtWidgets import QTableWidgetItem


def dataframe_from_table(table) -> pd.DataFrame:
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    data = []
    for r in range(table.rowCount()):
        row = []
        for c in range(table.columnCount()):
            item = table.item(r, c)
            txt = item.text().strip() if item else ""
            row.append(txt)
        data.append(row)
    df = pd.DataFrame(data, columns=headers)
    return df


def display_dataframe(table, df: pd.DataFrame):
    table.clear()
    table.setColumnCount(df.shape[1])
    table.setRowCount(df.shape[0])
    table.setHorizontalHeaderLabels(list(df.columns))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            text = '' if (pd.isna(val)) else str(val)
            item = QTableWidgetItem(text)
            table.setItem(i, j, item)


def load_excel_file(file_path: str) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)
    header_idx = next((i for i, row in raw.iterrows() if row.notna().sum() > 1), 0)
    df = pd.read_excel(file_path, header=header_idx)
    df = df.dropna(axis=1, how='all')
    cols_norm = df.columns.astype(str).str.strip().tolist()
    col_set = set(cols_norm)

    def is_repeat_header(row):
        vals = row.astype(str).str.strip().tolist()
        matches = sum(val in col_set for val in vals)
        return matches >= (len(cols_norm) / 2)

    df = df[~df.apply(is_repeat_header, axis=1)]
    df = df.dropna(axis=0, how='all').reset_index(drop=True)
    return df
