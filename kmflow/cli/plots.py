from __future__ import annotations

from typing import Optional, List
from pathlib import Path
from math import ceil

import typer
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.plots_utils as plots_utils
import kmflow.utils.pca_utils as pca_utils

app = typer.Typer(help="Plotting commands (read CSV from --input or stdin; save PNGs or display).")


@app.command("barplot")
def barplot(
    category_col: str = typer.Argument(..., help="Categorical column (x-axis when vertical)."),
    numeric_col: str = typer.Argument(..., help="Numeric column to plot."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Bar plot of <numeric_col> by <category_col>.
    """
    default_name = f"{category_col.capitalize()}_by_{numeric_col.capitalize()}_barplot.png"
    with tqdm(total=2, desc="Barplot", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.bar_plot(
            df=df,
            category_col=category_col,
            numeric_col=numeric_col,
            orient=orientation,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Barplot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Barplot displayed (not saved).")


@app.command("histogram")
def hist(
    x_axis: str = typer.Argument(..., help="Column to histogram."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    num_bins: int = typer.Option(10, "--bins", "-b", help="Number of bins."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Histogram of <x_axis>.
    """
    default_name = f"{x_axis}_hist.png"
    with tqdm(total=2, desc="Histogram", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.histogram(
            df=df,
            num_bins=num_bins,
            x_axis=x_axis,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Histogram saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Histogram displayed (not saved).")


@app.command("scatter")
def scatterplot(
    x_axis: str = typer.Argument(..., help="X-axis column."),
    y_axis: str = typer.Argument(..., help="Y-axis column."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x and y axes ranges."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Scatter plot of x vs y.
    """
    default_name = f"{x_axis}_vs_{y_axis}_scatter.png"
    with tqdm(total=2, desc="Scatter", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.scatter_plot(
            df=df,
            x_axis=x_axis,
            y_axis=y_axis,
            scale=scale,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Scatter saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Scatter displayed (not saved).")


@app.command("boxplot")
def boxplot(
    numeric_col: str = typer.Argument(..., help="Numeric column for the box plot."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    category_col: Optional[str] = typer.Option(
        None, "--category-col", "-c", help="Column to group by (one box per category)."
    ),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Comma-separated regex pattern(s) to filter categories. For ex: 'Price, Quantity'",
    ),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Box plot of `numeric_col`, optionally grouped by `category_col` and filtered by `patterns`.
    """
    name_mid = (
        f"filtered_{category_col}"
        if patterns
        else (f"by_{category_col}" if category_col else "all")
    )
    default_name = f"{name_mid}_{numeric_col}_boxplot.png"

    # normalize patterns from ["'a, b'"] → ["a", "b"] if provided once
    patt = patterns[0].split(", ") if (patterns and len(patterns) == 1) else patterns

    with tqdm(total=2, desc="Boxplot", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.box_plot(
            df=df,
            numeric_col=numeric_col,
            category_col=category_col,
            patterns=patt,
            orient=orientation,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Boxplot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Boxplot displayed (not saved).")


@app.command("violin")
def violinplot(
    numeric_col: str = typer.Argument(..., help="Numeric column for the violin plot."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    category_col: Optional[str] = typer.Option(
        None, "--category-col", "-c", help="Column to group by (one violin per category)."
    ),
    patterns: Optional[List[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Comma-separated regex pattern(s) to filter categories. For ex: 'Price, Quantity'",
    ),
    orientation: str = typer.Option("v", "--orientation", "-a", help="Orientation: 'v' or 'h'."),
    inner: str = typer.Option(
        "box", "--inner", "-i", help="Interior representation inside the violins."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Violin plot of `numeric_col`, optionally grouped by `category_col` and filtered by `patterns`.
    """
    name_mid = (
        f"filtered_{category_col}"
        if patterns
        else (f"by_{category_col}" if category_col else "all")
    )
    default_name = f"{name_mid}_{numeric_col}_violin.png"

    patt = patterns[0].split(", ") if (patterns and len(patterns) == 1) else patterns

    with tqdm(total=2, desc="Violin", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.violin_plot(
            df=df,
            numeric_col=numeric_col,
            category_col=category_col,
            patterns=patt,
            orient=orientation,
            inner=inner,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Violin saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Violin displayed (not saved).")


@app.command("heatmap")
def corr_heatmap(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Correlation matrix heatmap for all numeric features in the data.
    """
    default_name = "heatmap.png"
    with tqdm(total=2, desc="Heatmap", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.correlation_heatmap(
            df=df,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Heatmap saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Heatmap displayed (not saved).")


@app.command("qq")
def qq_plt(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-col",
        "-nc",
        help="Comma-separated list of numeric columns for Q-Q plot, ex: 'Price, Quantity', (omit when using --all).",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    all_cols: bool = typer.Option(
        False, "--all", "-a", help="Generate Q-Q plots for all numeric columns."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save PNG(s) (default) or display only."
    ),
):
    """
    Generate a Q-Q plot for one numeric column, multiple columns, or all numeric columns.
    """
    df = cli_utils.read_df(input_path)

    # All numeric columns mode
    if all_cols:
        cols = df.select_dtypes(include="number").columns.tolist()
        if not cols:
            raise typer.BadParameter("No numeric columns found.")

        with tqdm(total=2, desc="Q-Q Plots (all)", colour="green") as pbar:
            plots_utils._apply_cubehelix_style()
            n = len(cols)
            ncols = 3
            nrows = ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes_flat = axes.flatten() if n > 1 else [axes]
            for i, col in enumerate(cols):
                plots_utils.qq_plot(df, col, output_path=None, save=False, ax=axes_flat[i])
                axes_flat[i].set_title(col)
            for ax in axes_flat[n:]:
                ax.set_visible(False)
            fig.suptitle("Q-Q Plots")
            fig.tight_layout()
            pbar.update(1)

            if save:
                out = plots_utils._ensure_unique_path(Path.cwd() / "qq_all.png")
                fig.savefig(out)
                plt.close(fig)
                logger.success(f"Q-Q plots saved to {out!r}")
            else:
                plt.show()
                plt.close(fig)
                logger.success("Q-Q plots displayed (not saved).")
            pbar.update(1)
        return

    # Specific columns mode
    if not numeric_cols:
        raise typer.BadParameter("Specify --numeric-col (Shorthand: -nc) or use --all.")

    for col in numeric_cols:
        default_name = f"{col}_qq.png"
        with tqdm(total=2, desc=f"Q-Q Plot: {col}", colour="green") as pbar:
            plots_utils._apply_cubehelix_style()
            fig, ax = plt.subplots()
            plots_utils.qq_plot(df, col, output_path=None, save=False, ax=ax)
            pbar.update(1)
            if save:
                out = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
                fig.savefig(out)
                plt.close(fig)
                logger.success(f"Q-Q Plot: {col} saved to {out!r}.")
            else:
                plt.show()
                plt.close(fig)
                logger.success(f"Q-Q Plot: {col} displayed (not saved).")
            pbar.update(1)


@app.command("inertia")
def inertia(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV file of inertia vs k; omit to read stdin."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Elbow plot of K-Means inertia versus number of clusters.
    """
    default_name = "inertia.png"
    with tqdm(total=2, desc="Inertia", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.inertia_plot(inertia_df=df, output_path=out_path, save=save)
        pbar.update(1)
    if save:
        logger.success(f"Inertia plot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Inertia plot displayed (not saved).")


@app.command("silhouette")
def silhouette(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV file of silhouette scores; omit to read stdin."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Plot silhouette score versus number of clusters.
    """
    default_name = "silhouette.png"
    with tqdm(total=2, desc="Silhouette", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.silhouette_plot(silhouette_df=df, output_path=out_path, save=save)
        pbar.update(1)
    if save:
        logger.success(f"Silhouette plot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Silhouette plot displayed (not saved).")


@app.command("scree")
def scree(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV of PCA variance; omit to read stdin."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Scree plot of proportion variance explained by each principal component.
    """
    default_name = "scree.png"
    with tqdm(total=2, desc="Scree", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.scree_plot(df=df, output_path=out_path, save=save)
        pbar.update(1)
    if save:
        logger.success(f"Scree plot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Scree plot displayed (not saved).")


@app.command("cpv")
def cpv(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV of PCA cumulative variance; omit to read stdin."
    ),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Plot cumulative proportion of variance explained by principal components.
    """
    default_name = "cumulative_prop_var.png"
    with tqdm(total=2, desc="Cumulative PV", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        plots_utils.cumulative_var_plot(df=df, output_path=out_path, save=save)
        pbar.update(1)
    if save:
        logger.success(f"Cumulative variance plot saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Cumulative variance plot displayed (not saved).")


@app.command("cluster")
def cluster(
    x_axis: str = typer.Argument(..., help="Feature for X axis."),
    y_axis: str = typer.Argument(..., help="Feature for Y axis."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV of clustered data; omit to read stdin."
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x/y axes."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Scatter plot of X vs. Y colored by cluster labels.
    """
    default_name = f"{x_axis}_vs_{y_axis}_cluster.png"
    with tqdm(total=2, desc="Cluster Scatter", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        ax = plots_utils.cluster_scatter(
            df=df,
            x_axis=x_axis,
            y_axis=y_axis,
            cluster_col=cluster_col,
            scale=scale,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)
    if save:
        logger.success(f"Cluster scatter saved to {out_path!r}")
    else:
        plt.show()
        logger.success("Cluster scatter displayed (not saved).")


@app.command("3d-cluster")
def cluster3d(
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Clustered CSV; omit to read stdin."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Exactly three numeric columns (e.g. 'weight, height, width'); "
        "if omitted, the first three numeric columns are used.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    scale: float = typer.Option(
        1.0, "--scale", "-s", help="Multiplier for x, y, and z axis ranges."
    ),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save PNG (default) or open interactive plot."
    ),
):
    """
    3D scatter of three features colored by cluster labels.
    """
    with tqdm(total=2, desc="3D Cluster Scatter", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        cols = numeric_cols or df.select_dtypes(include="number").columns[:3].tolist()
        if len(cols) != 3:
            raise typer.BadParameter("Must specify exactly three numeric columns for 3D.")
        default_name = f"3d_cluster_{cols[0]}_{cols[1]}_{cols[2]}.png"
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        fig = plots_utils.cluster_scatter_3d(
            df=df,
            numeric_cols=cols,
            cluster_col=cluster_col,
            scale=scale,
            output_path=out_path,
            save=save,
        )
        pbar.update(1)

        # annotate/tune if desired (kept simple)
        if save:
            logger.success(f"3D cluster scatter saved to {out_path!r}")
        else:
            fig.show(renderer="browser")
            logger.success("3D cluster scatter opened in browser (not saved).")
        pbar.update(1)


@app.command("cluster-subplot")
def batch_cluster_plot(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Clustered CSV; omit to read stdin."
    ),
    x_axis: Optional[str] = typer.Argument(..., help="Feature for X axis."),
    y_axis: Optional[str] = typer.Argument(..., help="Feature for Y axis."),
    cluster_prefix: str = typer.Option(
        "cluster_", "--cluster-col", "-cluster-col", help="Prefix for cluster columns."
    ),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Multiplier for x and y axes."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    Create a grid of 2D cluster-colored scatter plots for each column starting with `cluster_prefix`.
    """
    with tqdm(total=2, desc="Cluster Subplot", colour="green") as pbar:
        df = cli_utils.read_df(input_path)

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        if not numeric_columns:
            raise typer.BadParameter("No numeric columns found in the data.")

        x_col = x_axis or numeric_columns[0]
        y_col = y_axis or (numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0])

        cluster_cols = sorted(
            [c for c in df.columns if c.startswith(cluster_prefix)],
            key=lambda c: int(c.replace(cluster_prefix, "")),
        )
        if not cluster_cols:
            raise typer.BadParameter(f"No columns found with prefix {cluster_prefix!r}")

        default_name = f"{x_col}_vs_{y_col}_batch.png"
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None

        fig = plots_utils.plot_batch_clusters(
            df=df,
            x_axis=x_col,
            y_axis=y_col,
            cluster_cols=cluster_cols,
            output_path=out_path,
            save=save,
            scale=scale,
        )
        pbar.update(1)

        if save:
            logger.success(f"Cluster subplot saved to {out_path!r}")
        else:
            plt.show()
            logger.success("Cluster subplot displayed (not saved).")
        pbar.update(1)


@app.command("biplot")
def plot_biplot(
    hue_column: str = typer.Argument(..., help="Cluster column to color points by."),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Choose numeric columns, comma-separated or repeatable. Defaults to all numeric columns if omitted.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    skip_scores: bool = typer.Option(
        True, "--skip-scores", "-ss", help="Assume df already has PC columns."
    ),
    pc_x: int = typer.Option(0, "--pc-x", "-x", help="Index of PC for x-axis (0-based)."),
    pc_y: int = typer.Option(1, "--pc-y", "-y", help="Index of PC for y-axis (0-based)."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Arrow length multiplier."),
    save: bool = typer.Option(True, "--no-save", "-n", help="Save PNG (default) or display only."),
):
    """
    2D Biplot: combines PC scores and loading vectors.
    """
    default_name = f"biplot_pc{pc_x + 1}-{pc_y + 1}.png"
    with tqdm(total=2, desc="Biplot", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        num_cols = numeric_cols or None

        summary = pca_utils.compute_pca(df=df, numeric_cols=num_cols, hue_column=hue_column)
        loadings = summary["loadings"]
        pve = summary["pve"]
        hue_ser = df[hue_column] if hue_column else None

        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None
        fig = plots_utils.biplot(
            df=df,
            loadings=loadings,
            pve=pve,
            skip_scores=skip_scores,
            pc_x=pc_x,
            pc_y=pc_y,
            scale=scale,
            hue=hue_ser,
            save=save,
            output_path=out_path,
        )
        pbar.update(1)

    if save:
        logger.success(f"Biplot saved to {out_path!r}")
    else:
        plt.show()
        plt.close(fig)
        logger.success("Biplot displayed (not saved).")


@app.command("3d-biplot")
def plot_3d_biplot(
    hue_column: Optional[str] = typer.Argument(
        ..., help="Column to color points by (e.g. cluster)."
    ),
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
    numeric_cols: str | None = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Exactly three features for PCA; if omitted, uses all numeric columns minus hue.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    skip_scores: bool = typer.Option(
        True, "--skip-scores", "-ss", help="Assume input already has PC columns."
    ),
    pc_x: int = typer.Option(0, "--pc-x", "-x", help="PC for X axis (0-based)."),
    pc_y: int = typer.Option(1, "--pc-y", "-y", help="PC for Y axis (0-based)."),
    pc_z: int = typer.Option(2, "--pc-z", "-z", help="PC for Z axis (0-based)."),
    scale: float = typer.Option(1.0, "--scale", "-s", help="Loading arrow scale."),
    save: bool = typer.Option(
        True, "--no-save", "-n", help="Save PNG (default) or open interactive plot."
    ),
):
    """
    3D PCA biplot: sample scores + loading vectors in three dimensions.
    """
    with tqdm(total=2, desc="3D Biplot", colour="green") as pbar:
        df = cli_utils.read_df(input_path)

        if not numeric_cols:
            numerics = df.select_dtypes(include="number").columns.tolist()
            if hue_column in numerics:
                numerics.remove(hue_column)
        else:
            numerics = numeric_cols

        if not skip_scores and len(numerics) < 3:
            raise typer.BadParameter("Need at least three numeric columns for PCA.")

        summary = pca_utils.compute_pca(df, numeric_cols=numerics, hue_column=hue_column)
        loadings = summary["loadings"]
        pve = summary["pve"]

        default_name = "3d_biplot.png"
        out_path = plots_utils._ensure_unique_path(Path.cwd() / default_name) if save else None

        fig = plots_utils.biplot_3d(
            df=df,
            loadings=loadings,
            pve=pve,
            output_path=out_path,
            skip_scores=skip_scores,
            pc_x=pc_x,
            pc_y=pc_y,
            pc_z=pc_z,
            scale=scale,
            hue=(df[hue_column] if hue_column else None),
            save=save,
        )
        pbar.update(1)

        if save:
            logger.success(f"3D biplot saved to {out_path!r}")
        else:
            fig.show(renderer="browser")
            logger.success("3D biplot opened in browser (not saved).")
        pbar.update(1)


if __name__ == "__main__":
    app()
