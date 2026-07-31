"""
Excel export (Phase 5) — turn a resulting dataframe into a formatted
exhibit: header styling, auto column widths, number formatting.
"""

import pandas as pd
import io


def _safe_str_len(value) -> int:
    """len(str(value)), tolerant of NaN/pd.NA/mixed-type columns that trip up astype(str)."""
    if pd.isna(value):
        return 3  # width of "nan"/"N/A"-ish placeholder
    return len(str(value))


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Exhibit", title: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        start_row = 2 if title else 0
        df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1D9E75", "font_color": "white",
            "border": 1, "align": "center", "valign": "vcenter",
        })
        title_fmt = workbook.add_format({"bold": True, "font_size": 14})
        number_fmt = workbook.add_format({"num_format": "#,##0.00"})

        if title:
            worksheet.write(0, 0, title, title_fmt)

        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(start_row, col_idx, col_name, header_fmt)

            col_data = df[col_name]
            max_len = max(
                (_safe_str_len(v) for v in col_data) if len(col_data) else [0],
                default=0,
            )
            max_len = max(max_len, len(str(col_name))) + 2
            width = min(max_len, 40)

            if pd.api.types.is_numeric_dtype(col_data):
                worksheet.set_column(col_idx, col_idx, width, number_fmt)
            else:
                worksheet.set_column(col_idx, col_idx, width)

        worksheet.freeze_panes(start_row + 1, 0)

    return buffer.getvalue()