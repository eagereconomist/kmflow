from pathlib import Path

import typer
from typing import Optional
from tqdm import tqdm

from kmflow.utils import cli_utils, process_utils, pca_utils

app = typer.Typer(
    help="PCA: compute loadings, scores, and variance summaries, writing CSVs to a directory."
)


@app.command("pca")
def run_pca(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, data is read from stdin.",
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
    n_components: int = typer.Option(
        None,
        "--n-components",
        "-c",
        help="Number of PCs to compute (defaults to all).",
    ),
    random_state: int = typer.Option(
        4572,
        "--seed",
        "-seed",
        help="Random seed for reproducibility.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        dir_okay=True,
        file_okay=False,
        help="Directory where PCA summary CSVs will be saved. Defaults to ./<stem>_pca",
    ),
):
    """
    Compute PCA loadings, scores, explained variance, and cumulative variance,
    then write four CSVs into a directory (created if needed).
    """
    # 1) Read input
    df = cli_utils.read_df(input_path)
    stem = input_path.stem if input_path is not None else "stdout"

    # 2) Parse numeric_cols → list[str] | None
    numeric_cols_arg = None if not numeric_cols.strip() else cli_utils.comma_split(numeric_cols)

    # 3) Compute PCA
    summary = pca_utils.compute_pca(
        df=df,
        numeric_cols=numeric_cols_arg,
        n_components=n_components,
        random_state=random_state,
    )

    # 4) Decide and create output directory
    outdir = output_dir or (Path.cwd() / f"{stem}_pca")
    outdir.mkdir(parents=True, exist_ok=True)

    # 5) Prepare outputs
    tasks = [
        (
            "PCA Loadings",
            summary["loadings"].reset_index().rename(columns={"index": "component"}),
            "pca_loadings",
        ),
        (
            "PCA Scores",
            summary["scores"].reset_index(drop=True),
            f"pca_scores_{summary['scores'].shape[1]}pc" if n_components else "pca_scores",
        ),
        (
            "Proportion Variance",
            summary["pve"].reset_index().rename(columns={"index": "component"}),
            "pca_proportion_var",
        ),
        (
            "Cumulative Variance",
            summary["cpve"].reset_index().rename(columns={"index": "component"}),
            "pca_cumulative_var",
        ),
    ]

    # 6) Write files with a progress bar
    for desc, df_out, suffix in tqdm(tasks, desc="Writing PCA CSVs", colour="green"):
        path = process_utils.write_csv(df_out, prefix=stem, suffix=suffix, output_dir=outdir)
        typer.secho(f"Saved {desc} -> {path!r}", fg="green")


if __name__ == "__main__":
    app()
