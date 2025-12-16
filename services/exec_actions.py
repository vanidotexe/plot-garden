from PyQt6.QtWidgets import QFileDialog, QMessageBox
import pandas as pd
import os

class ExecActions:
    def __init__(self, main_window):
        self.w = main_window

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.w, "Open Data File", "", "Excel & CSV Files (*.xlsx *.xls *.csv)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(('.csv')):
                df = pd.read_csv(path, header=0)
            else:
                df = pd.read_excel(path)
            from services.data_ops import display_dataframe
            display_dataframe(self.w.table, df)
        except Exception as e:
            QMessageBox.critical(self.w, "Import Error", str(e))

    def export_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self.w, "Save Data File", "data.csv", "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            from services.data_ops import dataframe_from_table
            df = dataframe_from_table(self.w.table)
            if path.lower().endswith('.xlsx'):
                df.to_excel(path, index=False)
            else:
                if not path.lower().endswith('.csv'):
                    path = path + ".csv"
                df.to_csv(path, index=False)
            QMessageBox.information(self.w, "Export", f"Saved to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self.w, "Export Error", str(e))
