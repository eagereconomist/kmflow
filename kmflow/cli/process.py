from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.process_utils as process_utils

app = typer.Typer(help="Apply a single scaler/transform to a CSV (stdin -> stdout).")


@app.command("norm")
def normalize(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
):
    with tqdm(total=2, desc="Applying Normalize Scaler to Data", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_df = process_utils.apply_normalizer(df)
        pbar.update(1)
    cli_utils.write_df(out_df)
    logger.success("Normalize CSV written to stdout.")


@app.command("std")
def standardize(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
):
    with tqdm(total=2, desc="Applying Standardize Scaler to Data", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_df = process_utils.apply_standardization(df)
        pbar.update(1)
    cli_utils.write_df(out_df)
    logger.success("Standardize CSV written to stdout.")


@app.command("minmax")
def minmax(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
):
    with tqdm(total=2, desc="Applying MinMax Scaler to Data", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_df = process_utils.apply_minmax(df)
        pbar.update(1)
    cli_utils.write_df(out_df)
    logger.success("MinMax CSV written to stdout.")


@app.command("log1p")
def log_scale(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
):
    with tqdm(total=2, desc="Applying log(1 + x) Scaler to Data", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_df = process_utils.apply_log1p(df)
        pbar.update(1)
    cli_utils.write_df(out_df)
    logger.success("log1p CSV written to stdout.")


@app.command("yj")
def yeo_johnson_scale(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i", help="CSV path; omit to read stdin."
    ),
):
    with tqdm(total=2, desc="Applying Yeo-Johnson Scaler to Data", colour="green") as pbar:
        df = cli_utils.read_df(input_path)
        pbar.update(1)
        out_df = process_utils.apply_yeo_johnson(df)
        pbar.update(1)
    cli_utils.write_df(out_df)
    logger.success("Yeo-Johnson CSV written to stdout.")


if __name__ == "__main__":
    app()
