from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.utils.kmeans_utils as kmeans_utils
import kmflow.utils.cli_utils as cli_utils

app = typer.Typer(help="K-Means clustering commands.")


@app.command("fit-km")
def fit_km_cli(
    k: int = typer.Argument(..., help="Number of clusters to fit."),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, data is read from stdin.",
    ),
    seed: int = typer.Option(
        4572,
        "--seed",
        "--random-seed",
        "-seed",
        help="Random seed for reproducibility.",
    ),
    n_init: int = typer.Option(
        50, "--n-init", "-n-init", help="Number of K-Means initializations."
    ),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="K-Means algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option("k-means++", "--init", "-init", help="'k-means++' or 'random'."),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
):
    """
    Read CSV (file via --input or stdin), fit a single K-Means model, append one cluster column,
    and write the resulting CSV to stdout.
    """
    with tqdm(total=3, desc="Fit K-Means", colour="green") as pbar:
        # 1) load
        df = cli_utils.read_df(input_path)
        pbar.update(1)

        # 2) fit
        df_out = kmeans_utils.fit_kmeans(
            df=df,
            k=k,
            numeric_cols=numeric_cols or None,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm=algorithm,
            cluster_col="cluster",
        )
        pbar.update(1)

        # 3) write
        cli_utils.write_df(df_out)
        logger.success("K-means CSV written to stdout.")
        pbar.update(1)


@app.command("batch-km")
def batch_km_cli(
    start: int = typer.Option(1, "--start", "-start", help="Minimum k (inclusive)."),
    stop: int = typer.Option(20, "--stop", "-stop", help="Maximum k (inclusive)."),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, data is read from stdin.",
    ),
    seed: int = typer.Option(4572, "--seed", "-seed", "--random-seed", help="Random seed."),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Runs per k."),
    algorithm: str = typer.Option("lloyd", "--algorithm", "-algo", help="'lloyd' or 'elkan'."),
    init: str = typer.Option("k-means++", "--init", "-init", help="'k-means++' or 'random'."),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated numeric columns; omit to use all numeric columns.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
):
    """
    Run K-Means for k in [start…stop], appending one cluster column per k
    (e.g., 'cluster_3', 'cluster_4', ...), and write the CSV to stdout.
    """
    with tqdm(total=3, desc="Batch K-Means", colour="green") as pbar:
        # 1) load
        df = cli_utils.read_df(input_path)
        pbar.update(1)

        # 2) batch-fit
        df_out = kmeans_utils.batch_kmeans(
            df=df,
            k_range=range(start, stop + 1),
            numeric_cols=numeric_cols or None,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm=algorithm,
            cluster_col="cluster",
        )
        pbar.update(1)

        # 3) write
        cli_utils.write_df(df_out)
        logger.success("Batch K-means clustering CSV written to stdout.")
        pbar.update(1)


if __name__ == "__main__":
    app()
