# Plot Garden

![Python](https://img.shields.io/badge/Python-black?logo=python&logoColor=white)
![PyQt6](https://img.shields.io/badge/PyQt6-black?logo=qt&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-black?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-black?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-black?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-black?logo=scikitlearn&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-000?logo=github&logoColor=white&labelColor=black)](LICENSE)

A lightweight desktop app for statistics and exploratory data analysis. Originally built in collaboration with doctors for clinical research workflows, and useful as a practical alternative to SPSS and GraphPad. Built with PyQt6 and the Python data stack.

## Existing functionality

- Spreadsheet-like data table with headers, inline editing, clipboard, and shortcuts.
- CSV/Excel import and export.
- Structure transforms: insert/delete rows/columns, rename headers, set row/column headers, transpose table.
- Plotting:
  - Histogram, Box Plot (single and grouped), Bar (single and grouped), Pie, Line
  - Scatter (1D multi-series, 2D pairs, overlaid), Scatter Matrix
  - Survival (Kaplan–Meier)
- Analytics:
  - K-Means (elbow + cluster scatter)
  - Linear Regression, Logistic Regression, Exponential Regression
  - One-way ANOVA
- Exec runtime scaffold for future scripted analyses in a safe namespace.

## Planned features

- Statistical tests: t-tests, chi-square, Fisher’s exact, Mann–Whitney U, Wilcoxon signed-rank, Kruskal–Wallis, Shapiro–Wilk.
- Modeling: Cox proportional hazards, regularization, ROC/PR curves, AUC, calibration.
- Survival: confidence intervals, median survival, log-rank tests, risk tables, stratified plots.
- EDA: correlation heatmaps, PCA, pairplots, violin/swarm plots, QQ plots.
- Data ops: robust type inference, categorical encodings, missing-value strategies, transformations (log, z-score), outlier handling.
- UX: project autosave, session restore, undo/redo, templates, improve current UI.
- Extensibility: optional local LLM assistant (Ollama) for code gen & plotting.

## Setup
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
python main.py
```

Notes:
- No standalone executable.
