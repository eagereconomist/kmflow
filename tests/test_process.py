from typer.testing import CliRunner
from kmflow.cli.process import app
import pandas as pd
import io

runner = CliRunner()


class TestProcessCLI:
    def test_standardize_stdout(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        stdin = io.StringIO(df.to_csv(index=False))
        result = runner.invoke(app, ["std"], input=stdin.getvalue())
        assert result.exit_code == 0, result.output
        out = result.stdout.strip().splitlines()
        assert out[0] == "x,y"
        assert "," in out[1]
