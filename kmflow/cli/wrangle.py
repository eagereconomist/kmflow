from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from loguru import logger
from tqdm import tqdm
import typer

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.wrangle_utils as wrangle_utils

app = typer.Typer(help="Data preprocessing commands.")


@app.command("outlier")
def iqr_outliers(
    source: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, data is read from stdin.",
    ),
    export_outliers: bool = typer.Option(
        False,
        "--export-outliers",
        "-eo",
        help="Write detected outliers as a CSV to stdout.",
    ),
    remove_outliers: bool = typer.Option(
        False,
        "--remove-outliers",
        "-ro",
        help="Write cleaned CSV (rows with outliers removed) to stdout.",
    ),
):
    """
    Identify IQR-based outliers and write exactly one CSV to stdout.

    Choose ONE:
      --export-outliers   -> prints the outlier table
      --remove-outliers   -> prints the input data with those rows removed
    """
    # Enforce exactly one output mode
    if export_outliers and remove_outliers:
        raise typer.BadParameter(
            "Choose exactly one of --export-outliers or --remove-outliers (not both)."
        )
    if not export_outliers and not remove_outliers:
        raise typer.BadParameter("No output selected. Use --export-outliers or --remove-outliers.")

    with tqdm(total=2, desc="IQR Outliers", colour="green") as pbar:
        df = cli_utils.read_df(source)
        pbar.update(1)

        out = wrangle_utils.find_iqr_outliers(df)
        pbar.update(1)

    if out.empty:
        logger.info("No IQR-based outliers detected.")
        if export_outliers:
            cli_utils.write_df(out.reset_index())
            logger.success("Empty IQR outlier table written to stdout.")
        return

    out_df = out.reset_index().rename(
        columns={"level_0": "row_index", "level_1": "column", 0: "outlier_value"}
    )

    if export_outliers:
        cli_utils.write_df(out_df)
        logger.success("Detected IQR outlier data written to stdout.")
        return

    # remove_outliers
    rows = out_df["row_index"].unique().tolist()
    cleaned = wrangle_utils.drop_row(df, rows)
    cli_utils.write_df(cleaned)
    logger.success("IQR outliers removed and data written to stdout.")


@app.command("preprocess")
def preprocess(
    source: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, data is read from stdin.",
    ),
    dropped_columns: str = typer.Option(
        "",
        "--dropped-column",
        "-dc",
        help="Columns to drop, comma-separated.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    dotless_columns: str = typer.Option(
        "",
        "--dotless-column",
        "-dot",
        help="Columns whose dots to remove, comma-separated.",
        callback=lambda x: cli_utils.comma_split(x) if isinstance(x, str) else x,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Show first 5 rows to stderr (does not affect CSV on stdout).",
    ),
):
    """
    Apply column drops and dotless renaming. Output is CSV on stdout.
    """
    df = cli_utils.read_df(source)

    steps: list[tuple[str, Callable, list]] = []
    for col in dropped_columns:
        steps.append(("drop_column", wrangle_utils.drop_column, [col]))
    for col in dotless_columns:
        steps.append(("dotless_column", wrangle_utils.dotless_column, [col]))

    for name, func, args in tqdm(steps, desc="Data Preprocessing Steps", colour="green"):
        logger.info(f"Applying {name}...")
        df = func(df, *args)

    if preview:
        typer.echo(df.head(), err=True)

    cli_utils.write_df(df)
    logger.success("Preprocessed CSV written to stdout.")


if __name__ == "__main__":
    app()
