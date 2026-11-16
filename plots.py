"""
Plotting of the figures to be used for the report. Further plots shall be made for future analyses. 
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullLocator

RESULTS = "results/results.csv"
OUT_DIR = "figures"
D_VALUES = [2, 4, 8, 16, 32, 64]

# One style per model, shared by every panel. Linear AE is dashed with hollow
# markers so it stays visible on top of PCA, which it overlaps almost exactly.
STYLE = {
    "PCA":       dict(color="#1f77b4", marker="o", linestyle="-",  markerfacecolor="#1f77b4", label="PCA"),
    "lin AE":    dict(color="#ff7f0e", marker="s", linestyle="--", markerfacecolor="none",    label="Linear AE"),
    "nonlin AE": dict(color="#2ca02c", marker="^", linestyle="-",  markerfacecolor="#2ca02c", label="Nonlinear AE"),
}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def aggregate(df):
    """Mean and std across seeds per (model, d). PCA is deterministic, so its std is 0."""
    agg = (df.groupby(["model", "d"])
             .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                  acc_mean=("acc", "mean"), acc_std=("acc", "std"))
             .reset_index())
    return agg.fillna(0.0)


def format_d_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(D_VALUES)
    ax.set_xticklabels([str(d) for d in D_VALUES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Latent dimension $d$")
    ax.grid(True, alpha=0.3, linewidth=0.6)


def plot_metric(ax, agg, metric, ylabel):
    for model, style in STYLE.items():
        sub = agg[agg["model"] == model].sort_values("d")
        ax.errorbar(sub["d"], sub[f"{metric}_mean"], yerr=sub[f"{metric}_std"],
                    capsize=2.5, linewidth=1.4, markersize=4.5, **style)
    format_d_axis(ax)
    ax.set_ylabel(ylabel)


def main_figure(agg):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    plot_metric(axes[0], agg, "mse", "Test reconstruction MSE")
    plot_metric(axes[1], agg, "acc", "Linear-probe test accuracy")

    for ax, tag in zip(axes, ["(a)", "(b)"]):
        ax.set_title(tag, loc="left", fontweight="bold", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    agg = aggregate(pd.read_csv(RESULTS))
    fig = main_figure(agg)
    fig.savefig(os.path.join(OUT_DIR, "main_figure.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "main_figure.png"), dpi=200, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/main_figure.pdf and .png")


if __name__ == "__main__":
    main()