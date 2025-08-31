from pathlib import Path
import unittest


class TestCodeIsTested(unittest.TestCase):
    def test_fixture_exists(self):
        self.assertTrue(Path("tests/fixtures/tennis_racquets.parquet").exists())


if __name__ == "__main__":
    unittest.main()
