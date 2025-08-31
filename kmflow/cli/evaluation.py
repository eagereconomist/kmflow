from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.config as config
import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.evaluation_utils as evaluation_utils

app = typer.Typer(help="K-Means evaluation metrics CLI.")


@app.command("benchmark")
def benchmark(
    input_dir: Path = typer.Argument(
        ...,
        help="Processed-data root under data/ (e.g. 'processed').",
    ),
    decimals: int = typer.Option(
        3,
        "--decimals",
        "-d",
        help="Round metric values to this many decimal places.",
    ),
):
    """
    Load all Calinski-Harabasz & Davies-Bouldin CSVs under data/<input_dir>/… and merge to one table.
    Writes CSV to stdout.
    """
    processed_root = config.DATA_DIR / input_dir

    with tqdm(total=4, desc="Benchmark", colour="green") as pbar:
        # 1) load calinski
        calinski_df = evaluation_utils.load_calinski_results(processed_root)
        pbar.update(1)

        # 2) load davies
        davies_df = evaluation_utils.load_davies_results(processed_root)
        pbar.update(1)

        # 3) merge & round
        merged = evaluation_utils.merge_benchmarks(calinski_df, davies_df)
        merged["calinski"] = merged["calinski"].round(decimals)
        merged["davies"] = merged["davies"].round(decimals)
        pbar.update(1)

        # 4) write stdout
        cli_utils.write_df(merged)
        logger.success("Benchmark table written to stdout.")
        pbar.update(1)


@app.command("inertia")
def inertia(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read from stdin."
    ),
    start: int = typer.Option(1, "--start", "-start", help="Minimum k (inclusive)."),
    stop: int = typer.Option(20, "--stop", "-stop", help="Maximum k (inclusive)."),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
):
    """
    Compute K-Means inertia over k = start…stop.
    Writes CSV to stdout.
    """
    # 1) load
    df = cli_utils.read_df(input_path)
    numeric_cols_arg = None if not numeric_cols.strip() else cli_utils.comma_split(numeric_cols)

    # 2) compute
    ks = tqdm(range(start, stop + 1), desc="Inertia", colour="green")
    inertia_df = evaluation_utils.compute_inertia_scores(
        df=df,
        k_range=ks,
        numeric_cols=numeric_cols_arg,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    # 3) write
    cli_utils.write_df(inertia_df)
    logger.success("Inertia table written to stdout.")


@app.command("silhouette")
def silhouette(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read from stdin."
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
):
    """
    Compute K-Means silhouette score for k = 2…n_samples-1.
    Writes CSV to stdout.
    """
    # 1) load
    df = cli_utils.read_df(input_path)
    numeric_cols_arg = None if not numeric_cols.strip() else cli_utils.comma_split(numeric_cols)

    ks = tqdm(
        range(2, df.select_dtypes(include="number").shape[0]), desc="Silhouette", colour="green"
    )
    silhouette_df = evaluation_utils.compute_silhouette_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    # write
    cli_utils.write_df(silhouette_df)
    logger.success("Silhouette table written to stdout.")


@app.command("calinski")
def calinski(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read from stdin."
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
):
    """
    Compute K-Means Calinski-Harabasz score for k = 2…n_samples-1.
    Writes CSV to stdout.
    """
    # 1) load
    df = cli_utils.read_df(input_path)
    numeric_cols_arg = None if not numeric_cols.strip() else cli_utils.comma_split(numeric_cols)

    ks = tqdm(
        range(2, df.select_dtypes(include="number").shape[0]), desc="Calinski", colour="green"
    )
    calinski_df = evaluation_utils.compute_calinski_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    # write
    cli_utils.write_df(calinski_df)
    logger.success("Calinski-Harabasz table written to stdout.")


@app.command("davies")
def davies(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read from stdin."
    ),
    random_state: int = typer.Option(
        4572, "--seed", "-seed", help="Random seed for reproducibility."
    ),
    n_init: int = typer.Option(50, "--n-init", "-n-init", help="Number of initializations per k."),
    algorithm: str = typer.Option(
        "lloyd", "--algorithm", "-algo", help="KMeans algorithm: 'lloyd' or 'elkan'."
    ),
    init: str = typer.Option(
        "k-means++", "--init", "-init", help="Initialization: 'k-means++' or 'random'."
    ),
    numeric_cols: str = typer.Option(
        "",
        "--numeric-cols",
        "-nc",
        help="Comma-separated list of numeric columns; omit to use all numeric columns.",
    ),
):
    """
    Compute K-Means Davies-Bouldin score for k = 2…n_samples-1.
    Writes CSV to stdout.
    """
    # 1) load
    df = cli_utils.read_df(input_path)
    numeric_cols_arg = None if not numeric_cols.strip() else cli_utils.comma_split(numeric_cols)

    ks = tqdm(range(2, df.select_dtypes(include="number").shape[0]), desc="Davies", colour="green")
    davies_df = evaluation_utils.compute_davies_scores(
        df=df,
        numeric_cols=numeric_cols_arg,
        k_values=ks,
        random_state=random_state,
        n_init=n_init,
        algorithm=algorithm,
        init=init,
    )

    # write
    cli_utils.write_df(davies_df)
    logger.success("Davies-Bouldin table written to stdout.")


if __name__ == "__main__":
    app()
