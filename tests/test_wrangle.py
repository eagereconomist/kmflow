import io
import unittest
import pandas as pd
from typer.testing import CliRunner

from kmflow.cli import wrangle as wrangle_cli


class TestWrangleCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet("tests/fixtures/tennis_racquets.parquet")
        cls.df = df[["length", "staticweight", "balance"]].head(100)

    def test_outlier_remove_stdout(self):
        runner = CliRunner()
        csv = self.df.to_csv(index=False)
        result = runner.invoke(
            wrangle_cli.app,
            ["outlier", "--remove-outliers"],
            input=csv,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        out_df = pd.read_csv(io.StringIO(result.stdout))
        self.assertLessEqual(len(out_df), len(self.df))
