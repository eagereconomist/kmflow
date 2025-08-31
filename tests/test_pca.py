import io
import pandas as pd
from typer.testing import CliRunner
from pathlib import Path

from kmflow.cli.pca import app

runner = CliRunner()


class TestPCACLI:
    def test_pca_writes_directory(self, tmp_path):
        # feed stdin and request default outdir
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        stdin = io.StringIO(df.to_csv(index=False))
        result = runner.invoke(app, ["--seed", "1234"], input=stdin.getvalue())
        assert result.exit_code == 0, result.output

        candidates = list(Path(".").glob("*_pca"))
        assert candidates, "Expected an output directory *_pca to be created"
        outdir = candidates[0]
        expect = {"pca_loadings", "pca_scores", "pca_proportion_var", "pca_cumulative_var"}
        found = {
            p.stem.split("_stdin_")[-1].split("stdin_")[-1]
            if "stdin" in p.stem
            else p.stem.split("_", 1)[-1]
            for p in outdir.glob("*.csv")
        }
        assert expect.issubset(found)
