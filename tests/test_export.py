import sys
import unittest
from pathlib import Path
import pandas as pd
import io

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import export


class TestDataframeToExcelBytes(unittest.TestCase):
    def test_basic_export_produces_valid_bytes(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = export.dataframe_to_excel_bytes(df)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)
        # Should be a real, openable xlsx
        readback = pd.read_excel(io.BytesIO(result))
        self.assertEqual(readback.shape, (3, 2))

    def test_regression_nullable_dtype_with_nan(self):
        """
        Regression test: pandas nullable/Arrow-backed dtypes can leave a
        missing value as an actual float NaN even after astype(str),
        which crashed the old column-width calculation. This must not
        raise TypeError: object of type 'float' has no len().
        """
        df = pd.DataFrame({
            "amount": pd.array([1.5, None, 100.25], dtype="Float64"),
            "name": pd.array(["Alice", None, "Bob"], dtype="string"),
        }).convert_dtypes()
        try:
            result = export.dataframe_to_excel_bytes(df)
        except TypeError as e:
            self.fail(f"dataframe_to_excel_bytes raised TypeError on nullable dtypes: {e}")
        self.assertGreater(len(result), 0)

    def test_export_with_title(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = export.dataframe_to_excel_bytes(df, title="My Exhibit")
        self.assertGreater(len(result), 0)

    def test_export_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame({"a": pd.array([], dtype="Int64"), "b": pd.array([], dtype="string")})
        try:
            result = export.dataframe_to_excel_bytes(df)
        except Exception as e:
            self.fail(f"Empty dataframe export raised: {e}")
        self.assertGreater(len(result), 0)

    def test_export_all_null_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": pd.array([None, None, None], dtype="Float64")})
        try:
            result = export.dataframe_to_excel_bytes(df)
        except Exception as e:
            self.fail(f"All-null column export raised: {e}")
        self.assertGreater(len(result), 0)

    def test_export_numpy_nan_directly(self):
        """Classic (non-nullable-dtype) NaN path should also work."""
        import numpy as np
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", np.nan, "z"]})
        result = export.dataframe_to_excel_bytes(df)
        self.assertGreater(len(result), 0)


class TestSafeStrLen(unittest.TestCase):
    def test_regular_string(self):
        self.assertEqual(export._safe_str_len("hello"), 5)

    def test_nan_float(self):
        self.assertEqual(export._safe_str_len(float("nan")), 3)

    def test_pd_na(self):
        self.assertEqual(export._safe_str_len(pd.NA), 3)

    def test_number(self):
        self.assertEqual(export._safe_str_len(12345), 5)


if __name__ == "__main__":
    unittest.main()
