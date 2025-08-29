from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import typer
from loguru import logger
from tqdm import tqdm

import kmflow.utils.cli_utils as cli_utils
import kmflow.utils.cluster_prep_utils as cluster_prep_utils

app = typer.Typer(help="Profile and label K-Means clusters.")


@app.command("cluster-profiles")
def cluster_profiles(
    raw_file: Path = typer.Argument(..., help="Raw CSV (use '-' for stdin)."),
    cluster_file: Path = typer.Argument(..., help="Clustered CSV (use '-' for stdin)."),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
    key_col: Optional[str] = typer.Option(
        None,
        "--key-col",
        "-k",
        help="If present, merge on this column instead of by row order.",
    ),
):
    """
    Generate per-cluster summary profiles.
    Writes CSV to stdout.
    """
    with tqdm(total=4, desc="Cluster Profiles", colour="green") as pbar:
        # 1) read raw
        raw_df = cli_utils.read_df(raw_file)
        pbar.update(1)

        # 2) read cluster
        cluster_df = cli_utils.read_df(cluster_file)
        pbar.update(1)

        # 3) merge + compute
        if key_col:
            merged = raw_df.merge(
                cluster_df[[key_col, cluster_col]],
                on=key_col,
                how="inner",
            )
        else:
            merged = cluster_prep_utils.merge_cluster_labels(raw_df, cluster_df, cluster_col)
        profiles = cluster_prep_utils.get_cluster_profiles(merged, cluster_col)
        pbar.update(1)

        # 4) write
        cli_utils.write_df(profiles)
        logger.success("Cluster profiles written to stdout.")
        pbar.update(1)


@app.command("map-clusters")
def map_clusters(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="CSV path. If omitted, reads from stdin.",
    ),
    cluster_col: str = typer.Argument(..., help="Column with cluster labels."),
):
    """
    Prompt for human labels per cluster ID, then count.
    Writes CSV to stdout.
    """
    df = cli_utils.read_df(input_path)
    unique_ids = sorted(df[cluster_col].unique())

    # 1) prompt for mapping
    mapping: Dict[int, str] = {}
    for cid in unique_ids:
        mapping[cid] = typer.prompt(f"Label for cluster {cid}")

    # 2) apply mapping & count
    labels = cluster_prep_utils.clusters_to_labels(df[cluster_col], mapping)
    counts = cluster_prep_utils.count_labels(labels, label_col="cluster_label")

    # 3) echo summary
    typer.echo("\nCluster → Label mapping:")
    for cid, label in mapping.items():
        typer.echo(f"  {cid} → {label}")

    typer.echo("\nCounts per label:")
    typer.echo(counts.to_markdown(index=False))

    # 4) write to stdout
    cli_utils.write_df(counts)
    logger.success("Label counts written to stdout.")


if __name__ == "__main__":
    app()
