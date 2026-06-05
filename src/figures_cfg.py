import os, warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("figures", exist_ok=True)

# ── Palette ──────────────────────────────────────────────────
PRIMARY = "#003366"
MUTED   = "#6E7888"
INK     = "#121826"
LIGHT   = "#F0F2F5"
WHITE   = "#FFFFFF"

BLUE_RAMP = ["#C8D9EF", "#8AADD8", "#4D80C0", PRIMARY, "#073870"]
CAT_PAL   = [PRIMARY, "#3A78C9", MUTED, "#8FA8CC", "#2D4E7A", "#A0AABC"]

_EXTRACTOR = {
    "year":      lambda idx: idx.year,
    "month":     lambda idx: idx.month,
    "week":      lambda idx: idx.isocalendar().week.astype(int),
    "dayofweek": lambda idx: idx.dayofweek,
    "hour":      lambda idx: idx.hour,
}

_TICK_LABELS = {
    "month":     (range(1, 13),  ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"]),
    "dayofweek": (range(0, 7),   ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]),
    "hour":      (range(0, 24),  [f"{h:02d}:00" for h in range(24)]),
    "week":      (range(1, 54),  [str(w) for w in range(1, 54)]),
    "year":      (None, None),
}

_AXIS_LABEL = {
    "month":     "Month",
    "dayofweek": "Day of week",
    "hour":      "Hour of day",
    "week":      "Week",
    "year":      "Year",
}

_SEASON_VAL_LABELS = {
    "dayofweek": {i: d for i, d in enumerate(
                    ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])},
    "month":     {i: m for i, m in enumerate(
                    ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"], start=1)},
}


def apply_theme():
    sns.set_theme(style="white")
    mpl.rcParams.update({
        # Sans-serif everywhere
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica",
                                 "Arial", "DejaVu Sans"],
        "text.color":           INK,
        # Axes
        "axes.facecolor":       WHITE,
        "figure.facecolor":     WHITE,
        "axes.edgecolor":       LIGHT,
        "axes.labelcolor":      MUTED,
        "axes.labelsize":       10,
        "axes.labelweight":     "normal",
        "axes.titlesize":       11,
        "axes.titleweight":     "bold",
        "axes.titlecolor":      INK,
        "axes.titlepad":        12,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     False,
        "axes.spines.bottom":   True,
        "axes.linewidth":       0.6,
        "axes.grid":            True,
        "axes.grid.axis":       "y",       # horizontal grid only
        "grid.color":           LIGHT,
        "grid.linewidth":       0.8,
        # Ticks — minimal
        "xtick.color":          MUTED,
        "ytick.color":          MUTED,
        "xtick.labelsize":      8.5,
        "ytick.labelsize":      8.5,
        "xtick.bottom":         True,
        "ytick.left":           False,     # no left tick marks
        "xtick.major.size":     3,
        "xtick.major.width":    0.6,
        "ytick.major.size":     0,
        # Legend
        "legend.frameon":       False,
        "legend.fontsize":      8.5,
        "legend.title_fontsize":9,
        "legend.labelcolor":    INK,
        # Lines
        "lines.linewidth":      2.0,
        "lines.solid_capstyle": "round",
        # Output
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    WHITE,
        "savefig.pad_inches":   0.15,
    })

def plot_feature_ts(df, feature, title=None, subtitle=None):
    """
    Plot a single feature from a datetime-indexed dataframe.
    Styled to match the dissertation theme.
    """
    if title is None:
        title = feature

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(df.index, df[feature], color=PRIMARY, linewidth=0.5, alpha=0.9)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PRIMARY)
    ax.spines["bottom"].set_linewidth(1.0)

    # Grid
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_facecolor("white")

    # Ticks
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))

    # Labels
    ax.set_ylabel(feature, color=MUTED, fontsize=10, fontfamily="sans-serif")
    ax.set_xlabel("Date", color=MUTED, fontsize=10, fontfamily="sans-serif")

    # Title + optional subtitle
    ax.set_title(title, color=INK, fontweight="bold", fontsize=11, loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                color=MUTED, fontsize=8.5, va="bottom")

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    
    return fig

def plot_features_ts(df, features, rows, cols, suptitle=None, subtitle=None):
    """
    Plot multiple features in a grid layout.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.2))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        ax.plot(df.index, df[feature], color=PRIMARY, linewidth=0.5, alpha=0.9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(PRIMARY)
        ax.spines["bottom"].set_linewidth(1.0)

        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.set_facecolor("white")

        ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
        ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))

        ax.set_ylabel(feature, color=MUTED, fontsize=10)
        ax.set_xlabel("Date", color=MUTED, fontsize=10)
        ax.set_title(feature, color=INK, fontweight="bold", fontsize=11, loc="left", pad=14)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, x=0.0, ha="left", fontsize=13,
                     fontweight="bold", color=INK, y=1.01)
    if subtitle:
        fig.text(0.0, 0.99, subtitle, ha="left", fontsize=9, color=MUTED, va="top")

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    
    return fig

def _make_corr_cmap():
    """
    Build a diverging colormap anchored to the dissertation palette:
      negative  → warm off-white
      zero      → LIGHT (#F0F2F5)
      positive  → PRIMARY (#003366)
    Returns a LinearSegmentedColormap.
    """
    colors = [
        "#C94040",   # strong negative  (deep red)
        "#E8A0A0",   # mild negative
        LIGHT,       # zero / uncorrelated
        "#8AADD8",   # mild positive
        PRIMARY,     # strong positive  (dissertation blue)
    ]
    return LinearSegmentedColormap.from_list("corr_theme", colors, N=256)
 
def plot_correlation_matrix(
    df,
    title=None,
    subtitle=None,
    method="pearson",
    annot=True,
    annot_threshold=0.0,
    mask_upper=False,
    figsize=None,
):
    """
    Plot a styled correlation matrix matching the dissertation theme.
 
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns will be correlated.
    title : str, optional
        Bold title shown top-left.  Defaults to "Correlation Matrix".
    subtitle : str, optional
        Muted subtitle shown just below the title.
    method : {'pearson', 'spearman', 'kendall'}
        Correlation method passed to df.corr().
    annot : bool
        Whether to annotate each cell with the correlation value.
    annot_threshold : float
        Only annotate cells whose |r| >= this value.  0.0 = annotate all.
    mask_upper : bool
        If True, show only the lower triangle (including diagonal).
    figsize : tuple, optional
        Override the auto-computed figure size.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    corr = df.corr(method=method)
    n = len(corr)
 
    if figsize is None:
        side = max(5, n * 0.72)
        figsize = (side, side * 0.88)
 
    cmap = _make_corr_cmap()
 
    # ── Mask ─────────────────────────────────────────────────
    mask = np.zeros_like(corr, dtype=bool)
    if mask_upper:
        mask[np.triu_indices_from(mask, k=1)] = True
 
    # ── Figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
 
    # Draw heatmap manually (no seaborn dependency for this func)
    data = corr.values.copy()
    data_masked = np.where(mask, np.nan, data)
 
    im = ax.imshow(
        data_masked,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
 
    # ── Grid lines (cell separators) ─────────────────────────
    for i in range(n + 1):
        ax.axhline(i - 0.5, color=WHITE, linewidth=1.2, zorder=2)
        ax.axvline(i - 0.5, color=WHITE, linewidth=1.2, zorder=2)
 
    # ── Annotations ──────────────────────────────────────────
    if annot:
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    continue
                val = data[i, j]
                if abs(val) < annot_threshold:
                    continue
                # Choose ink vs white based on cell darkness
                cell_brightness = abs(val)
                text_color = WHITE if cell_brightness > 0.55 else INK
                ax.text(
                    j, i,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=max(6.5, 9.5 - n * 0.18),
                    color=text_color,
                    fontfamily="sans-serif",
                    zorder=3,
                )
 
    # ── Tick labels ──────────────────────────────────────────
    labels = list(corr.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right",
                       fontsize=8.5, color=MUTED)
    ax.set_yticklabels(labels, fontsize=8.5, color=MUTED)
    ax.tick_params(axis="both", which="both", length=0, pad=6)
 
    # ── Spines ───────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    # ── Colorbar ─────────────────────────────────────────────
    cbar = fig.colorbar(
        im, ax=ax,
        fraction=0.035, pad=0.03,
        ticks=[-1, -0.5, 0, 0.5, 1],
    )
    cbar.ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"],
                             fontsize=8, color=MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
 
    # ── Title + subtitle ─────────────────────────────────────
    _title = title if title is not None else "Correlation Matrix"
    ax.set_title(
        _title,
        color=INK, fontweight="bold", fontsize=11,
        loc="left", pad=14,
        fontfamily="sans-serif",
    )
    if subtitle:
        ax.text(
            0, 1.01, subtitle,
            transform=ax.transAxes,
            color=MUTED, fontsize=8.5, va="bottom",
            fontfamily="sans-serif",
        )
 
    # ── Grid off (we drew our own separators) ────────────────
    ax.grid(False)
 
    fig.tight_layout()
    return fig

def plot_seasonal(df, feature, season, period, agg="mean", title=None, subtitle=None):
    """
    Overlay seasonal plot — one line per season group, x-axis over period.

    Args:
        df (pd.DataFrame): Datetime-indexed dataframe.
        feature (str): Column to plot.
        season (str): Grouping variable; one line per unique value.
            One of: 'year', 'month', 'week', 'dayofweek', 'hour'.
        period (str): X-axis variable; position within each season.
            One of: 'year', 'month', 'week', 'dayofweek', 'hour'.
        agg (str): Aggregation passed to groupby. Default 'mean'.
        title (str, optional): Plot title.
        subtitle (str, optional): Muted subtitle below title.

    Returns:
        matplotlib.figure.Figure

    Examples:
        >>> plot_seasonal(df, "OT", season="year",      period="month")
        >>> plot_seasonal(df, "OT", season="year",      period="hour")
        >>> plot_seasonal(df, "OT", season="month",     period="hour")
        >>> plot_seasonal(df, "OT", season="dayofweek", period="hour")
    """
    if title is None:
        title = f"{feature}: by {_AXIS_LABEL[period]} per {_AXIS_LABEL[season]}"

    season_vals      = sorted(_EXTRACTOR[season](df.index).unique())
    colors           = cm.tab10(np.linspace(0, 0.9, min(len(season_vals), 10)))
    color_cycle      = [colors[i % 10] for i in range(len(season_vals))]
    val_labels       = _SEASON_VAL_LABELS.get(season)
    period_extractor = _EXTRACTOR[period]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    for val, color in zip(season_vals, color_cycle):
        mask    = _EXTRACTOR[season](df.index) == val
        subset  = df.loc[mask, feature]
        grouped = subset.groupby(period_extractor(subset.index)).agg(agg)
        label   = val_labels[val] if val_labels else str(val)
        ax.plot(grouped.index, grouped.values,
                label=label, color=color, linewidth=1.0, alpha=0.85)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PRIMARY)
    ax.spines["bottom"].set_linewidth(1.0)

    # Grid
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_facecolor(WHITE)

    # Ticks
    ticks, labels = _TICK_LABELS.get(period, (None, None))
    if ticks is not None:
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))

    # Labels
    ax.set_xlabel(_AXIS_LABEL[period], color=MUTED, fontsize=10)
    ax.set_ylabel(feature,             color=MUTED, fontsize=10)

    # Title + subtitle
    ax.set_title(title, color=INK, fontweight="bold", fontsize=11,
                 loc="left", pad=14 if not subtitle else 26)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                color=MUTED, fontsize=8.5, va="bottom")

    # Legend
    ax.legend(
        title=_AXIS_LABEL[season], title_fontsize=8, fontsize=8,
        bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False, labelcolor=MUTED,
    )

    fig.tight_layout()
    return fig

def plot_seasonal_grid(df, features, season, period, rows, cols,
                       agg="mean", suptitle=None, subtitle=None):
    """
    Grid of seasonal overlay plots — one subplot per feature.

    Args:
        df (pd.DataFrame): Datetime-indexed dataframe.
        features (list[str]): Columns to plot, one per subplot.
        season (str): Grouping variable; one line per unique value.
            One of: 'year', 'month', 'week', 'dayofweek', 'hour'.
        period (str): X-axis variable; position within each season.
            One of: 'year', 'month', 'week', 'dayofweek', 'hour'.
        rows (int): Number of subplot rows.
        cols (int): Number of subplot columns.
        agg (str): Aggregation passed to groupby. Default 'mean'.
        suptitle (str, optional): Bold figure-level title.
        subtitle (str, optional): Muted figure-level subtitle.

    Returns:
        matplotlib.figure.Figure
    """
    season_vals      = sorted(_EXTRACTOR[season](df.index).unique())
    colors           = cm.tab10(np.linspace(0, 0.9, min(len(season_vals), 10)))
    color_cycle      = [colors[i % 10] for i in range(len(season_vals))]
    val_labels       = _SEASON_VAL_LABELS.get(season)
    period_extractor = _EXTRACTOR[period]
    ticks, tlabels   = _TICK_LABELS.get(period, (None, None))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.2))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, feature in zip(axes, features):
        for val, color in zip(season_vals, color_cycle):
            mask    = _EXTRACTOR[season](df.index) == val
            subset  = df.loc[mask, feature]
            grouped = subset.groupby(period_extractor(subset.index)).agg(agg)
            label   = val_labels[val] if val_labels else str(val)
            ax.plot(grouped.index, grouped.values,
                    label=label, color=color, linewidth=1.0, alpha=0.85)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(PRIMARY)
        ax.spines["bottom"].set_linewidth(1.0)

        # Grid
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.set_facecolor(WHITE)

        # Ticks
        if ticks is not None:
            ax.set_xticks(list(ticks))
            ax.set_xticklabels(tlabels, rotation=30, ha="right", fontsize=8)
        ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
        ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))

        # Labels
        ax.set_xlabel(_AXIS_LABEL[period], color=MUTED, fontsize=10)
        ax.set_ylabel(feature,             color=MUTED, fontsize=10)
        ax.set_title(feature, color=INK, fontweight="bold",
                     fontsize=11, loc="left", pad=14)

    # Hide unused axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    # Shared legend — taken from last active axis
    handles, labels = axes[min(len(features), rows * cols) - 1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title=_AXIS_LABEL[season], title_fontsize=8, fontsize=8,
        bbox_to_anchor=(1.01, 0.5), loc="center left",
        frameon=False, labelcolor=MUTED,
    )

    # Suptitle + subtitle
    if suptitle:
        fig.suptitle(suptitle, x=0.0, ha="left", fontsize=13,
                     fontweight="bold", color=INK, y=1.01)
    if subtitle:
        fig.text(0.0, 0.99, subtitle, ha="left", fontsize=9,
                 color=MUTED, va="top")

    fig.tight_layout()
    return fig

def plot_monthly_box(df, feature, title=None, subtitle=None):
    """
    Monthly boxplot for a single feature.

    Args:
        df (pd.DataFrame): Datetime-indexed dataframe.
        feature (str): Column to plot.
        title (str, optional): Plot title. Defaults to "<feature>: Monthly Distribution".
        subtitle (str, optional): Muted subtitle below title.

    Returns:
        matplotlib.figure.Figure
    """
    if title is None:
        title = f"{feature}: Monthly Distribution"

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    groups = [df.loc[df.index.month == m, feature].dropna().values
              for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        medianprops=dict(color=PRIMARY, linewidth=1.5),
        boxprops=dict(facecolor=LIGHT, color=MUTED, linewidth=0.8),
        whiskerprops=dict(color=MUTED, linewidth=0.8),
        capprops=dict(color=MUTED, linewidth=0.8),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=MUTED, markeredgewidth=0,
                        alpha=0.4),
        widths=0.55,
    )

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PRIMARY)
    ax.spines["bottom"].set_linewidth(1.0)

    # Grid
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_facecolor(WHITE)

    # Ticks
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune="both"))

    # Labels
    ax.set_xlabel("Month", color=MUTED, fontsize=10)
    ax.set_ylabel(feature, color=MUTED, fontsize=10)

    # Title + subtitle
    ax.set_title(title, color=INK, fontweight="bold", fontsize=11,
                 loc="left", pad=14 if not subtitle else 26)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                color=MUTED, fontsize=8.5, va="bottom")

    fig.tight_layout()
    return fig