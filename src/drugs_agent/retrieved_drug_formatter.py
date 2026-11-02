"""Format retrieved drug records as the input table for the drug agent's LLM.

This only presents rows. Search, classification and choosing which rows answer the
question are not done here.
"""

import json
from typing import Any

from tabulate import tabulate

# Keep the stock list short enough for a doctor to read. Search returns the closest matches first.
MAX_DRUG_ROWS = 15


# ---------------------------------------------------------------------------
# INPUT PREPARATION — Present retrieved data without selecting or inventing rows
# ---------------------------------------------------------------------------
def format_retrieved_drug_table(
    drug_records: list[dict[str, Any]], max_rows: int = MAX_DRUG_ROWS
) -> str:
    """Return the first `max_rows` records as a Markdown table, preserving arrival order.

    Columns follow first appearance across the shown records. Blank cells represent
    absent fields. Present cells are JSON literals, so lists and string values retain
    their types and special characters. Pipes are JSON-escaped to protect table
    boundaries. If rows were cut, a final NOTE tells the model how many were shown.
    """
    if not drug_records:
        return "No drug records were retrieved."

    shown_records = drug_records[:max_rows]
    column_names = list(dict.fromkeys(key for record in shown_records for key in record))
    if not column_names:
        return "Retrieved drug records contain no fields."

    table_rows = []
    for record in shown_records:
        table_row = []
        for column_name in column_names:
            if column_name not in record:
                table_row.append("")
            else:
                cell = json.dumps(record[column_name], ensure_ascii=False)
                table_row.append(cell.replace("|", r"|"))
        table_rows.append(table_row)

    table = tabulate(
        table_rows,
        headers=column_names,
        tablefmt="github",
        disable_numparse=True,
    )
    if len(drug_records) > max_rows:
        table += (
            f"\n\nNOTE: Showing the closest {max_rows} of {len(drug_records)} retrieved records."
        )
    return table
