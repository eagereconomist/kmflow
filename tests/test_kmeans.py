import io
import unittest
import pandas as pd
from typer.testing import CliRunner

from kmflow.cli import kmeans as kmeans_cli


class TestKMeansCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet("tests/fixtures/tennis_racquets.parquet")
        cls.df = df[["length", "staticweight", "balance"]].head(60)

    def test_fit_km_stdout(self):
        runner = CliRunner()
        csv = self.df.to_csv(index=False)
        result = runner.invoke(
            kmeans_cli.app,
            [
                "fit-km",
                "3",
                "--random-seed",
                "42",
                "--n-init",
                "5",
                "--algorithm",
                "lloyd",
                "--init",
                "k-means++",
            ],
            input=csv,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("cluster", result.stdout.splitlines()[0])
