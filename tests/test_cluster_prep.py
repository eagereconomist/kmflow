import tempfile
from pathlib import Path
import pandas as pd
from typer.testing import CliRunner

from kmflow.cli.cluster_prep import app

runner = CliRunner()


def _make_raw_and_cluster(tmpdir: Path):
    # simple tiny fixture
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [1.0, 0.9, 1.1, 1.2],
        }
    )
    clustered = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "cluster": [0, 1, 0, 1],
        }
    )
    raw_p = tmpdir / "raw.csv"
    clu_p = tmpdir / "clustered.csv"
    raw.to_csv(raw_p, index=False)
    clustered.to_csv(clu_p, index=False)
    return raw_p, clu_p


class TestClusterPrepCLI:
    def test_cluster_profiles_stdout(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw_p, clu_p = _make_raw_and_cluster(td)
            result = runner.invoke(
                app,
                ["cluster-profiles", str(raw_p), str(clu_p), "cluster"],
            )
            assert result.exit_code == 0, result.output
            # should be CSV on stdout
            assert "cluster" in result.stdout

    def test_map_clusters_interactive_counts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            _, clu_p = _make_raw_and_cluster(td)
            user_inputs = "A\nB\n"
            result = runner.invoke(
                app,
                ["map-clusters", str(clu_p), "cluster"],
                input=user_inputs,
            )
            assert result.exit_code == 0, result.output
            assert "| cluster_label | count |" in result.stdout
