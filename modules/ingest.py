"""
File ingestion: turn an uploaded file into a pandas dataframe with
reasonable type inference, regardless of source format.
"""

import io
import pandas as pd

def load_file(uploaded_file) -> pd.DataFrame:
    """
    uploaded_file: a Streamlit UploadedFile (has .name and read()-able bytes)
    Returns a raw (not-yet-cleaned) dataframe.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    buffer = io.BytesIO(data)

    if name.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    elif name.endswith(".json"):
        df = pd.read_json(buffer)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(buffer)
    elif name.endswith((".tsv", ".txt")):
        df = pd.read_csv(buffer, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    # Best-effort dtype tightening (nullable ints, proper strings, etc.)
    df = df.convert_dtypes()
    return df
