import os
import unittest
import pandas as pd
from typer.testing import CliRunner

from kmflow.cli import plots as plots_cli


class TestPlotsCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = pd.read_parquet("tests/fixtures/tennis_racquets.parquet")
        cls.df = df.head(120)

    def test_histogram_creates_file(self):
        runner = CliRunner()
        csv = self.df.to_csv(index=False)
        with runner.isolated_filesystem():
            result = runner.invoke(
                plots_cli.app,
                ["histogram", "beamwidth", "--bins", "30"],
                input=csv,
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(os.path.exists("beamwidth_hist.png"))
