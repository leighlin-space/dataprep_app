"""
Run the full test suite: python tests/run_tests.py

Uses only the standard library (unittest) — no extra install needed.
On your machine, with duckdb already installed via requirements.txt,
every test runs for real, including the DuckDB-connection tests that
can't run in a sandbox without network access to install duckdb.
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
