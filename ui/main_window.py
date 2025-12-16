import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as _plt
from PyQt6.QtWidgets import (
    QMainWindow, QTableWidget, QTableWidgetItem, QMessageBox,
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QScrollArea,
    QToolBar, QInputDialog, QLabel, QFileDialog,
    QToolButton, QMenu, QDialog, QCheckBox, QDialogButtonBox, QHBoxLayout,
    QFormLayout, QComboBox, QLineEdit, QTextEdit
)
from PyQt6.QtGui import QKeySequence, QShortcut, QTextCursor
from PyQt6.QtCore import Qt

from services import plots
from services.data_ops import display_dataframe, load_excel_file, dataframe_from_table
from ui.plot_actions import PlotActions
from ui.chat_panel import ChatPanel
from services.exec_actions import ExecActions
from services.exec_runtime import ExecRuntime


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Statistics")
        self.setMinimumSize(1000, 600)

        # Assistant offline stub
        self.chat_session = None

        # Toolbar: create actions, add to toolbar, and connect handlers
        toolbar = QToolBar("Toolbar")
        toolbar.setMovable(False)

        action_import = toolbar.addAction("Import Data")
        action_import.triggered.connect(lambda: self.exec.import_file())

        clear_action = toolbar.addAction("Reset Spreadsheet")
        clear_action.triggered.connect(self.clear_spreadsheet)

        action_save = toolbar.addAction("Export File")
        action_save.triggered.connect(lambda: self.exec.export_file())

        insertButton = QToolButton()
        insertButton.setText("Insert")
        insertButton.clicked.connect(self.insert_column)
        insertMenu = QMenu(insertButton)
        actionInsertRow = insertMenu.addAction("Insert Row")
        actionInsertColumn = insertMenu.addAction("Insert Column")
        insertButton.setMenu(insertMenu)
        insertButton.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        actionInsertRow.triggered.connect(self.insert_row)
        actionInsertColumn.triggered.connect(self.insert_column)
        toolbar.addWidget(insertButton)

        deleteButton = QToolButton()
        deleteButton.setText("Delete")
        deleteButton.clicked.connect(self.delete_column)
        deleteMenu = QMenu(deleteButton)
        actionDeleteRow = deleteMenu.addAction("Delete Row")
        actionDeleteColumn = deleteMenu.addAction("Delete Column")
        deleteButton.setMenu(deleteMenu)
        deleteButton.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        actionDeleteRow.triggered.connect(self.delete_row)
        actionDeleteColumn.triggered.connect(self.delete_column)
        toolbar.addWidget(deleteButton)

        transposeButton = toolbar.addAction("Transpose")
        transposeButton.triggered.connect(self.transpose_table)

        set_vheaders = toolbar.addAction("Set Row Headers")
        set_vheaders.triggered.connect(self.set_column_as_vheaders)

        set_headers = toolbar.addAction("Set Column Headers")
        set_headers.triggered.connect(self.set_row_as_hheaders)

        # Assistant temporarily disabled
        # chat_toggle = toolbar.addAction("Assistant")
        # chat_toggle.triggered.connect(
        #     lambda: self.chat_panel.setVisible(not self.chat_panel.isVisible())
        # )

        self.addToolBar(toolbar)

        # Chat panel (temporarily hidden/disabled)
        # self.chat_panel = ChatPanel(self)
        # self.chat_panel.setVisible(False)
        # self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chat_panel)

        # Spreadsheet table
        self.table = QTableWidget(15, 8)
        headers = [f"var{i}" for i in range(1, 9)]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self.rename_column)
        self.table.verticalHeader().sectionDoubleClicked.connect(self.rename_row)
        self.setCentralWidget(self.table)

        # Shortcuts:
        # Ctrl+C, Ctrl+V, Ctrl+X have default behavior
        # Ctrl+B deletes selected rows or columns
        self.delete_sc = QShortcut(QKeySequence("Ctrl+B"), self.table)
        self.delete_sc.activated.connect(self.delete_selected)
        self.copy_sc = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self.table)
        self.copy_sc.activated.connect(self.copy_selection)
        self.paste_sc = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self.table)
        self.paste_sc.activated.connect(self.paste_selection)
        self.cut_sc = QShortcut(QKeySequence(QKeySequence.StandardKey.Cut), self.table)
        self.cut_sc.activated.connect(self.cut_selection)

        self._clipboard_data = []

        # Sidebar widget
        # Buttons are created, connected to functions, and added to the left panel
        dock = QDockWidget(self)
        sidebar_title = QLabel("Plots")
        sidebar_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dock.setTitleBarWidget(sidebar_title)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock_layout = QVBoxLayout()

        histogram_button = QPushButton("Histogram")
        histogram_button.clicked.connect(lambda: self.plots.plot_histogram())
        dock_layout.addWidget(histogram_button)

        boxplot_button = QPushButton("Box Plot")
        boxplot_button.clicked.connect(lambda: self.plots.plot_box_plot())
        dock_layout.addWidget(boxplot_button)

        scatter_button = QPushButton("Scatter Plot")
        scatter_button.clicked.connect(lambda: self.plots.plot_scatter())
        dock_layout.addWidget(scatter_button)

        pie_button = QPushButton("Pie Chart")
        pie_button.clicked.connect(lambda: self.plots.plot_pie_chart())
        dock_layout.addWidget(pie_button)

        bar_button = QPushButton("Bar Chart")
        bar_button.clicked.connect(lambda: self.plots.plot_bar_chart())
        dock_layout.addWidget(bar_button)

        scatter_mat_button = QPushButton("Scatter Matrix")
        scatter_mat_button.clicked.connect(lambda: self.plots.plot_scatter_matrix())
        dock_layout.addWidget(scatter_mat_button)

        survival_button = QPushButton("Survival Plot")
        survival_button.clicked.connect(lambda: self.plots.plot_survival())
        dock_layout.addWidget(survival_button)

        linegraph_button = QPushButton("Line Graph")
        linegraph_button.clicked.connect(lambda: self.plots.plot_line_graph())
        dock_layout.addWidget(linegraph_button)

        anova_btn = QPushButton("ANOVA")
        anova_btn.clicked.connect(lambda: self.plots.run_anova())
        dock_layout.addWidget(anova_btn)

        linregression_button = QPushButton("Linear Regression")
        linregression_button.clicked.connect(lambda: self.plots.run_linear_regression())
        dock_layout.addWidget(linregression_button)

        logistic_reg_btn = QPushButton("Binary Log. Regression")
        logistic_reg_btn.clicked.connect(lambda: self.plots.run_logistic())
        dock_layout.addWidget(logistic_reg_btn)

        exp_reg_btn = QPushButton("Exponential Regression")
        exp_reg_btn.clicked.connect(lambda: self.plots.run_exp_regression())
        dock_layout.addWidget(exp_reg_btn)

        kmeans_button = QPushButton("K-Means")
        kmeans_button.clicked.connect(lambda: self.plots.run_kmeans())
        dock_layout.addWidget(kmeans_button)

        dock_layout.addStretch()

        dock_container = QWidget()
        dock_container.setLayout(dock_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(dock_container)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dock.setWidget(scroll_area)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # instantiate controllers
        self.plots = PlotActions(self)
        self.exec = ExecActions(self)

    def clear_spreadsheet(self):
        self.table.clear()
        self.table.setRowCount(15)
        self.table.setColumnCount(8)
        headers = [f"var{i}" for i in range(1, 9)]
        self.table.setHorizontalHeaderLabels(headers)

    def insert_row(self):
        current_row_count = self.table.rowCount()
        self.table.insertRow(current_row_count)

    def insert_column(self):
        current_column_count = self.table.columnCount()
        self.table.insertColumn(current_column_count)
        self.table.setHorizontalHeaderItem(current_column_count, QTableWidgetItem(f"var{current_column_count + 1}"))

    def delete_row(self):
        current_row_count = self.table.rowCount()
        if current_row_count > 0:
            self.table.removeRow(current_row_count - 1)

    def delete_column(self):
        current_column_count = self.table.columnCount()
        if current_column_count > 0:
            self.table.removeColumn(current_column_count - 1)

    def prompt_missing_column(self, col: str, frac: float):
        # Prompt when a column has >= 30% empty cells
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Column '{col}' missing {int(frac*100)}% data")
        layout = QVBoxLayout(dlg)
        label = QLabel(f"Column '{col}' has {frac:.0%} missing values. What would you like to do?")
        layout.addWidget(label)

        btns = QDialogButtonBox(dlg)
        keep_btn = btns.addButton("Keep", QDialogButtonBox.ButtonRole.AcceptRole)
        drop_btn = btns.addButton("Drop", QDialogButtonBox.ButtonRole.DestructiveRole)
        fill_btn = btns.addButton("Fill...", QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(btns)

        keep_btn.clicked.connect(lambda: dlg.done(1))
        drop_btn.clicked.connect(lambda: dlg.done(2))
        fill_btn.clicked.connect(lambda: dlg.done(3))

        result = dlg.exec()
        if result == 1:
            return 'keep', None
        if result == 2:
            return 'drop', None
        if result == 3:
            val, ok = QInputDialog.getText(self, "Fill Missing", f"Enter value to fill for '{col}':")
            if ok:
                return 'fill', val
        return 'keep', None

    # Import handled by ExecActions.import_file()

    # Export handled by ExecActions.export_file()

    def set_column_as_vheaders(self):
        col_labels = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Column as Vertical Header")
        layout = QVBoxLayout(dlg)
        combo = QComboBox(dlg)
        combo.addItems(col_labels)
        layout.addWidget(QLabel("Choose a column:"))
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected_col = combo.currentText()
            col_index = col_labels.index(selected_col)
            for row in range(self.table.rowCount() - 1, -1, -1):
                item = self.table.takeItem(row, col_index)
                if item:
                    self.table.setVerticalHeaderItem(row, QTableWidgetItem(item.text()))
                else:
                    self.table.removeRow(row)
            self.table.removeColumn(col_index)

    def set_row_as_hheaders(self):
        row_labels = [
            (self.table.verticalHeaderItem(i).text() if self.table.verticalHeaderItem(i) else str(i+1))
            for i in range(self.table.rowCount())
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Row as Column Headers")
        layout = QVBoxLayout(dlg)
        combo = QComboBox(dlg)
        combo.addItems(row_labels)
        layout.addWidget(QLabel("Choose a row:"))
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected_row = combo.currentText()
        row_index = row_labels.index(selected_row)
        for col in range(self.table.columnCount() - 1, -1, -1):
            item = self.table.takeItem(row_index, col)
            if item and item.text().strip():
                self.table.setHorizontalHeaderItem(col, QTableWidgetItem(item.text()))
            else:
                self.table.removeColumn(col)
        self.table.removeRow(row_index)

    # Shortcut functions #
    def copy_selection(self):
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        rows = r.rowCount()
        cols = r.columnCount()
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                item = self.table.item(r.topRow()+i, r.leftColumn()+j)
                row.append(item.text() if item else "")
            data.append(row)
        self._clipboard_data = data
        text = "\n".join("\t".join(r) for r in data)
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)

    def paste_selection(self):
        if not self._clipboard_data:
            return
        start_row = self.table.currentRow()
        start_col = self.table.currentColumn()
        if start_row < 0 or start_col < 0:
            return
        for i, row in enumerate(self._clipboard_data):
            if start_row + i >= self.table.rowCount():
                self.table.insertRow(self.table.rowCount())
            for j, val in enumerate(row):
                if start_col + j >= self.table.columnCount():
                    self.table.insertColumn(self.table.columnCount())
                self.table.setItem(start_row+i, start_col + j, QTableWidgetItem(val))

    def delete_selected(self):
        sel = self.table.selectionModel()
        rows = sel.selectedRows()
        if rows:
            for idx in sorted((r.row() for r in rows), reverse=True):
                self.table.removeRow(idx)
            return
        cols = sel.selectedColumns()
        if cols:
            for idx in sorted((c.column() for c in cols), reverse=True):
                self.table.removeColumn(idx)

    def cut_selection(self):
        self.copy_selection()
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        for i in range(r.rowCount()):
            for j in range(r.columnCount()):
                row = r.topRow() + i
                col = r.leftColumn() + j
                item = self.table.item(row, col)
                if item:
                    item.setText("")
                else:
                    self.table.setItem(row, col, QTableWidgetItem(""))

    def transpose_table(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        col_labels = [self.table.horizontalHeaderItem(j).text() if self.table.horizontalHeaderItem(j) else f"var{j+1}" for j in range(cols)]
        row_labels = [self.table.verticalHeaderItem(i).text() if self.table.verticalHeaderItem(i) else str(i+1) for i in range(rows)]
        data = {col_labels[j]: [(self.table.item(i, j).text() if self.table.item(i, j) else "") for i in range(rows)] for j in range(cols)}
        df = pd.DataFrame(data, index=row_labels)
        df_t = df.transpose()
        new_col_labels = row_labels
        new_row_labels = col_labels
        self.table.clear()
        self.table.setRowCount(len(new_row_labels))
        self.table.setColumnCount(len(new_col_labels))
        self.table.setHorizontalHeaderLabels(new_col_labels)
        self.table.setVerticalHeaderLabels(new_row_labels)
        for i, rlabel in enumerate(new_row_labels):
            for j, clabel in enumerate(new_col_labels):
                val = df_t.iat[i, j]
                self.table.setItem(i, j, QTableWidgetItem(str(val)))

    def rename_column(self, index):
        item = self.table.horizontalHeaderItem(index)
        text = item.text() if item else f"var{index + 1}"
        new_text, ok = QInputDialog.getText(self, "Rename Column", "Enter new column name:", text=text)
        if ok and new_text:
            self.table.horizontalHeaderItem(index).setText(new_text)

    def rename_row(self, index):
        item = self.table.verticalHeaderItem(index)
        text = item.text() if item else f"Row {index + 1}"
        new_text, ok = QInputDialog.getText(self, "Rename Row", "Enter new row name:", text=text)
        if ok and new_text:
            self.table.setVerticalHeaderItem(index, QTableWidgetItem(new_text))

    # Plotting and selection dialogs are handled by PlotActions.
