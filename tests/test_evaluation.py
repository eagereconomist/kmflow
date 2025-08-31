import io
import unittest
import pandas as pd
from typer.testing import CliRunner

from kmflow.cli import evaluation as eval_cli


class TestEvaluationCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet("tests/fixtures/tennis_racquets.parquet")
        cls.df = df[["length", "staticweight", "balance"]].head(25)

    def test_inertia_stdout_small_range(self):
        runner = CliRunner()
        csv = self.df.to_csv(index=False)
        result = runner.invoke(
            eval_cli.app,
            ["inertia", "--start", "2", "--stop", "5", "--seed", "17", "--n-init", "3"],
            input=csv,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        out_df = pd.read_csv(io.StringIO(result.stdout))
        self.assertIn("k", out_df.columns)
        self.assertIn("inertia", out_df.columns)
        self.assertTrue(out_df["k"].between(2, 5).all())

    def test_silhouette_stdout(self):
        runner = CliRunner()
        csv = self.df.to_csv(index=False)
        result = runner.invoke(
            eval_cli.app,
            ["silhouette", "--seed", "17", "--n-init", "3"],
            input=csv,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        out_df = pd.read_csv(io.StringIO(result.stdout))
        self.assertIn("n_clusters", out_df.columns)
        self.assertIn("silhouette_score", out_df.columns)
