from sklearn.cluster import KMeans
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import f_oneway, probplot
from scipy.optimize import curve_fit
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math
import os

dir = os.path.dirname(__file__)
plots_dir = os.path.join(dir, "plots")

mpl.rcParams['font.size'] = 12

mpl.rcParams['axes.titlesize'] = 13

# Tick labels
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13

# Axis labels
mpl.rcParams['axes.labelsize'] = 13


def save_on_close(fig, name, subfolder=None):
    # Save plots to plots/<subdir>
    target_dir = plots_dir
    if subfolder:
        target_dir = os.path.join(target_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    fullpath = os.path.join(target_dir, filename)

    fig.canvas.mpl_connect(
        'close_event',
        lambda event: fig.savefig(fullpath, dpi=300)
    )


def plot_histogram(chosen, numeric_data, text_data, rows=None, cols=None):
    # Check if it's a single variable or a list of variables
    if not isinstance(chosen, list):
        values = numeric_data.get(chosen, []) if isinstance(
            numeric_data, dict) else numeric_data
        if values:
            plt.figure()
            plt.hist(values, bins=10, edgecolor='black')
            plt.title(f"Histogram of {chosen}")
            plt.xlabel(chosen)
            plt.ylabel("Frequency")
            manager = plt.get_current_fig_manager()
            manager.window.setWindowTitle("Histogram Plot")
            save_on_close(plt.gcf(), "histogram", subfolder="histograms")
            plt.show()
        else:
            counts = Counter(text_data)
            labels = list(counts.keys())
            freqs = list(counts.values())
            plt.figure()
            plt.bar(labels, freqs)
            plt.title(f"Histogram of {chosen} (Categorical)")
            plt.xlabel(chosen)
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.tight_layout()
            manager = plt.get_current_fig_manager()
            manager.window.setWindowTitle("Histogram Plot")
            save_on_close(plt.gcf(), "histogram", subfolder="histograms")
            plt.show()
        return

    # Assume the list contains numerical data
    n = len(chosen)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for idx, label in enumerate(chosen):
        ax = axes[idx]
        values = numeric_data.get(label, [])
        if values:
            bins = np.histogram_bin_edges(values, bins=12)
            ax.hist(values, bins=bins, edgecolor='black')
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
        else:
            ax.set_visible(False)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Histogram Plot")
    save_on_close(plt.gcf(), "histogram", subfolder="histograms")
    plt.show()


def plot_box_plot(chosen_label, numeric_data):
    # Single row or single column
    plt.figure()
    plt.boxplot(numeric_data, labels=[f"{chosen_label} (numeric)"])
    plt.title("Box Plot")
    plt.ylabel("Values")
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Box Plot")
    save_on_close(plt.gcf(), "boxplot", subfolder="boxplots")
    plt.show()


def plot_grouped_box_plot(numeric_label, grouping_label, grouped_data):
    n_groups = len(grouped_data)

    # Increase figure width if grouping by many variables (otherwise too narrow)
    fig_width = max(8, n_groups * 1.2)
    plt.figure(figsize=(fig_width, 6))

    data_to_plot = [v for v in grouped_data.values() if v]
    group_names = [k for k, v in grouped_data.items() if v]

    colors = ['lightblue', 'lightgreen']
    bplot = plt.boxplot(data_to_plot, labels=group_names, patch_artist=True)
    for i, box in enumerate(bplot['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor('black')

    plt.title(f"Box Plot of {numeric_label} grouped by {grouping_label}")
    plt.ylabel("Values")

    # Tick labels under each box plot; color them to improve readability when many groups
    tick_colors = ['blue', 'green']
    ax = plt.gca()
    for i, tick_label in enumerate(ax.get_xticklabels()):
        tick_label.set_color(tick_colors[i % len(tick_colors)])

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Box Plot")
    save_on_close(plt.gcf(), "boxplot", subfolder="boxplots")
    plt.grid(True)
    plt.show()


def plot_box_plots(chosen_labels, numeric_data_dict):
    n = len(chosen_labels)
    # Maximum 3 box plots per row side-by-side; then wrap to the next row
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for idx, label in enumerate(chosen_labels):
        ax = axes[idx]
        data = numeric_data_dict.get(label, [])
        if data:
            ax.boxplot(data)
            ax.set_title(label)
            ax.set_ylabel("Values")
            ax.set_xticks([])
        else:
            # Hide subplot if no data for the variable
            ax.set_visible(False)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Box Plot")
    save_on_close(plt.gcf(), "boxplot", subfolder="boxplots")
    plt.show()


def plot_scatter(x_label, y_label, x_data, y_data):
    # For a single x,y pair
    plt.figure(figsize=(6, 6))
    plt.scatter(x_data, y_data, alpha=0.7, edgecolor='black')
    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.grid(True)
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Scatter Plot")
    save_on_close(plt.gcf(), "scatterplot", subfolder="scatterplots")
    plt.show()


def plot_scatter_1d(data_dict):
    labels = list(data_dict.keys())
    # Use matplotlib integrated color cycle
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Case for a single variable
    if len(labels) == 1:
        key = labels[0]
        vals = data_dict[key]
        plt.figure(figsize=(8, 2))
        plt.scatter(vals, np.zeros(len(vals)),
                    color=palette[0], alpha=0.7, edgecolor='black')
        plt.yticks([])
        plt.xlabel(key)
        plt.title(f"1D Scatter of {key}")
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.window.setWindowTitle("Scatter Plot")
        save_on_close(plt.gcf(), "scatterplot", subfolder="scatterplots")
        plt.show()
        return

    # Case for multiple variables
    n = len(labels)
    fig_height = max(4, n * 1.2)
    plt.figure(figsize=(8, fig_height))
    for idx, key in enumerate(labels):
        vals = data_dict[key]
        y = np.full(len(vals), idx)
        plt.scatter(vals, y,
                    label=key,
                    color=palette[idx % len(palette)],
                    alpha=0.7,
                    edgecolor='black')
    plt.yticks(range(n), labels)
    plt.xlabel("Value")
    plt.title("1D Scatter of Selected Variables")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Scatter Plot")
    save_on_close(plt.gcf(), "scatterplot", subfolder="scatterplots")
    plt.show()


def plot_scatter_overlaid(pairs):
    # For multiple x,y pairs
    plt.figure(figsize=(8, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

    for i, (x_label, y_label, x, y) in enumerate(pairs):
        plt.scatter(
            x, y,
            label=f"{x_label} vs {y_label}",
            alpha=0.7,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)]
        )

    plt.title("Overlaid 2D Scatter")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Scatter Plot")
    save_on_close(plt.gcf(), "scatterplot", subfolder="scatterplots")
    plt.show()


def plot_pie_chart(chosen, data_dict):
    # For a single variable
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Pie Chart of {chosen}")
    plt.axis('equal')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Pie Chart")
    save_on_close(plt.gcf(), "piechart", subfolder="piecharts")
    plt.show()


def plot_pie_charts(pie_list, pies_per_row=4):
    # For multiple variables
    n = len(pie_list)
    rows = math.ceil(n / pies_per_row)
    cols = min(n, pies_per_row)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    for idx, (label, counts) in enumerate(pie_list):
        ax = axes[idx]
        ax.pie(list(counts.values()), labels=list(
            counts.keys()), autopct='%1.1f%%', startangle=90)
        ax.set_title(label)
        ax.axis('equal')

    # plt.subplots returns rows*cols axes; hide those with idx >= n
    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Pie Chart")
    save_on_close(plt.gcf(), "piechart", subfolder="piecharts")
    plt.show()


def plot_bar_chart(labels, values):
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x, values, 0.6, edgecolor='black')
    plt.xticks(x, labels)
    plt.title("Bar Chart of Selected Variables")
    plt.xlabel("Variable")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Bar Chart")
    save_on_close(plt.gcf(), "barchart", subfolder="barcharts")
    plt.show()


def plot_grouped_bar_chart(var_labels, group_labels, grouped_means, grouping_vars=None):
    n_groups = len(group_labels)
    n_vars = len(var_labels)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(6, n_groups*1.2), 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    bar_w = 0.8 / n_vars
    for i, var in enumerate(var_labels):
        means = [grouped_means[g][i] for g in group_labels]
        ax.bar(
            x + i*bar_w,
            means,
            width=bar_w,
            label=var,
            edgecolor='black',
            color=colors[i % len(colors)]
        )

    # Place x-axis ticks at the center of grouped bars and rotate text to avoid overlap
    ax.set_xticks(x + 0.4 - bar_w/2)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')

    if grouping_vars:
        grp_title = ", ".join(grouping_vars)
        ax.set_xlabel(grp_title)
        ax.set_title(
            f"Bar Chart of {', '.join(var_labels)}\ngrouped by {grp_title}")
    else:
        ax.set_xlabel("Group")
        ax.set_title("Grouped Bar Chart")

    ax.set_ylabel("Mean Value")
    ax.legend(title="Variable")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Bar Chart")
    save_on_close(plt.gcf(), "barchart", subfolder="barcharts")
    plt.grid(True)
    plt.show()


def plot_scatter_matrix(labels, data_dict):
    # Scatter matrix: diagonal shows histograms of corresponding variables; off-diagonals are 2D scatters
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
    for i, yi in enumerate(labels):
        for j, xi in enumerate(labels):
            ax = axes[i, j]
            x = data_dict[xi]
            y = data_dict[yi]
            if i == j:
                ax.hist(x, bins='auto', edgecolor='black')
            else:
                ax.scatter(x, y, s=10, alpha=0.6)
            if i < n-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(xi)
            if j > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(yi)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Scatter Matrix")
    save_on_close(plt.gcf(), "scattermatrix", subfolder="scattermatrices")
    plt.show()


def plot_kaplan_meier(df, time_col, event_col, group_col=None):
    # Plot Kaplan-Meier curve (adapted from scikit-survival documentation)
    plt.figure()
    groups = df[group_col].dropna().unique() if group_col else [None]

    for grp in groups:
        if grp is None:
            mask = df.index == df.index
            label = "Overall"
        else:
            mask = df[group_col] == grp
            label = str(grp)

        times = df.loc[mask, time_col].astype(float).to_numpy()
        events = df.loc[mask, event_col].astype(bool).to_numpy()

        km_times, km_surv = kaplan_meier_estimator(events, times)
        plt.step(km_times, km_surv, where="post", label=label)

    plt.xlabel(time_col)
    plt.ylabel("Survival Probability")
    title = "Kaplan–Meier Curve"
    if group_col:
        title += f" by {group_col}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Survival Plot - Kaplan-Meier")
    save_on_close(plt.gcf(), "kaplan_meier", subfolder="survival_plots")
    plt.show()


def plot_kmeans_elbow(df, x_col, y_col, max_k=10):
    # Elbow method to find optimal number of clusters for K-means
    data = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    Ks = list(range(1, max_k + 1))
    inertias = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0).fit(data)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(Ks, inertias, marker='o')
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia")
    plt.title("K-Means Inertia for K clusters")
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("K-Means Inertia")
    save_on_close(plt.gcf(), "kmeans_elbow", subfolder="kmeans_plots")
    plt.show()


def plot_kmeans_scatter(df, x_col, y_col, k):
    # After choosing optimal K, run KMeans and plot clusters and centroids (scikit-learn)
    data = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    km = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = km.labels_
    centers = km.cluster_centers_

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1],
                marker='X', s=50, edgecolor='k')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"K-Means Clustering (K={k})")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("K-Means Clusters")
    save_on_close(plt.gcf(), "kmeans_scatter", subfolder="kmeans_plots")
    plt.show()


def plot_linear_regression(df, x_col, y_col):
    # Linear regression using scikit-learn
    data = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    coef = model.coef_[0]
    intercept = model.intercept_

    plt.figure()
    plt.scatter(X[:, 0], y, label="Data")
    order = X[:, 0].argsort()
    plt.plot(X[order, 0], y_pred[order],
             label=f"y = {coef:.3f} x + {intercept:.3f}", linewidth=2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Linear Regression: {y_col} vs {x_col}")
    plt.legend()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Linear Regression")
    save_on_close(plt.gcf(), "linear_regression", subfolder="regression_plots")
    plt.show()


def plot_line_graph(chosen_labels, numeric_data_dict):
    n = len(chosen_labels)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_1d(axes).flatten()

    for idx, label in enumerate(chosen_labels):
        ax = axes[idx]
        data = numeric_data_dict.get(label, [])
        if data:
            ax.plot(data, marker='o')
            ax.set_title(label)
        else:
            ax.set_visible(False)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Line Graph")
    save_on_close(plt.gcf(), "line_graph", subfolder="line_graphs")
    plt.show()


def perform_anova(groups: dict[str, list[float]]):
    # One-way ANOVA using scipy.stats.f_oneway
    f_stat, p_val = f_oneway(*groups.values())

    # For each group: compute mean and residuals (y - mean)
    residuals = []
    for vals in groups.values():
        mu = np.mean(vals)
        residuals.extend(y - mu for y in vals)
    residuals = np.array(residuals)
    N, k = len(residuals), len(groups)
    # Compute mean squared error, estimate deviation via sqrt, then standardize residuals
    mse = np.sum(residuals**2) / (N - k)
    sigma_hat = np.sqrt(mse)
    std_resid = residuals / sigma_hat

    # Use a 2x2 subplot grid and access via axs[row, col]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Box plot reused so we can save into the proper subdirectory
    data = [v for v in groups.values() if v]
    names = [k for k, v in groups.items() if v]
    bplot = axs[0, 0].boxplot(data, labels=names, patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for i, box in enumerate(bplot['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_edgecolor('black')
    tick_colors = ['blue', 'green']
    for i, lbl in enumerate(axs[0, 0].get_xticklabels()):
        lbl.set_color(tick_colors[i % len(tick_colors)])
    axs[0, 0].set_title("Box Plot of ANOVA Data")
    axs[0, 0].set_xlabel("Group")
    axs[0, 0].set_ylabel("Value")

    # Bar chart for F-stat and P-value
    labels = ["F-statistic", "P-value"]
    vals = [f_stat, p_val]
    x = np.arange(len(labels))
    bars = axs[0, 1].bar(x, vals, width=0.6, edgecolor='black')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].set_title("ANOVA: F vs P")
    axs[0, 1].set_ylabel("Value")
    for bar, v in zip(bars, vals):
        axs[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.3g}",
            ha="center", va="bottom"
        )
    ymax = max(vals)
    # Increased y-limit so F-stat label doesn’t hit the plot edge
    axs[0, 1].set_ylim(0, ymax * 1.15)

    axs[1, 0].hist(residuals, bins='auto', edgecolor='black')
    axs[1, 0].set_title("Histogram of ANOVA Residuals")
    axs[1, 0].set_xlabel("Residual")
    axs[1, 0].set_ylabel("Frequency")

    # QQ plot
    probplot(std_resid, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("QQ Plot of std residuals")

    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Anova 1 way")
    save_on_close(fig, "anova_1way", subfolder="anova_plots")
    plt.show()


def plot_logistic_regression(df, x_col, y_col):
    X = df[[x_col]].astype(float).to_numpy()
    y = df[y_col].astype(int).to_numpy()

    model = LogisticRegression()
    model.fit(X, y)

    x_min, x_max = X.min(), X.max()
    X_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    probs = model.predict_proba(X_grid)[:, 1]

    fig = plt.figure(figsize=(6, 4))

    plt.scatter(X, y, alpha=0.6, edgecolor='k', label='data')
    plt.plot(X_grid, probs, linewidth=2, label=f"P({y_col}=1)")
    plt.xlabel(x_col)
    plt.ylabel(f"Probability of {y_col}=1")
    plt.title(f"Logistic Regression: {y_col} vs {x_col}")
    plt.legend()
    plt.tight_layout()

    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Logistic Regression")
    save_on_close(fig, f"logistic_{x_col}_vs_{y_col}",
                  subfolder="regression_plots")
    plt.show()


def exp_func(x, a, b):
    return a * np.exp(b * x)


def plot_exponential_reg(df, x_col, y_col):
    data = df[[x_col, y_col]].dropna().to_numpy(dtype=float)
    x = data[:, 0]
    y = data[:, 1]

    params, _ = curve_fit(exp_func, x, y, maxfev=10000)
    a, b = params

    x_sorted = np.sort(x)
    y_pred = exp_func(x_sorted, a, b)

    plt.figure()
    plt.scatter(x, y, label="Data", edgecolor='black', alpha=0.7)
    plt.plot(x_sorted, y_pred, color='red',
             label=f"y = {a:.2f} * e^({b:.2f}x)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Exponential Regression: {y_col} vs {x_col}")
    plt.legend()
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.window.setWindowTitle("Exponential Regression")
    save_on_close(plt.gcf(), "exponential_regression",
                  subfolder="regression_plots")
    plt.show()
