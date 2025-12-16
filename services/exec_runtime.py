from PyQt6.QtWidgets import QMessageBox
from services.data_ops import dataframe_from_table
from services import plots as _plots
import pandas as _pd

class ExecRuntime:
    def __init__(self, main_window):
        self.w = main_window

    def build_namespace(self):
        df = dataframe_from_table(self.w.table)
        df = df.apply(lambda col: _pd.to_numeric(col, errors="ignore"))
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([_plots.np.inf, -_plots.np.inf], _plots.np.nan)
        df = df.dropna(axis=0, subset=numeric_cols, how="all")
        return {
            "self": self.w,
            "df": df,
            "pd": _pd,
            "plt": _plots.plt,
            "curve_fit": _plots.curve_fit,
            "LinearRegression": _plots.LinearRegression,
            "LogisticRegression": _plots.LogisticRegression,
            "f_oneway": _plots.f_oneway,
            "KMeans": _plots.KMeans,
            "plots": _plots,
        }

    def exec_snippet(self, code: str):
        ns = self.build_namespace()
        try:
            exec(code, globals(), ns)
        except Exception as e:
            QMessageBox.critical(self.w, "Execution Error", str(e))
