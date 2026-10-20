"""Render a drug answer as a Markdown table."""

from tabulate import tabulate

from .schema import DrugAnswer, DrugMatch

NO_MATCH_MESSAGE = "No matching drugs were found."
EMPTY_CELL = "-"
TABLE_HEADERS = (
    "Drug",
    "Class",
    "Subclass",
    "Therapeutic category",
    "Route",
    "Form",
    "Strengths",
)


def _markdown_cell(cell_value: str | None) -> str:
    """Show a missing value as a dash and keep a pipe from splitting the cell."""
    if not cell_value:
        return EMPTY_CELL
    return cell_value.replace("|", "\\|")


def _drug_table_row(drug_match: DrugMatch) -> list[str]:
    return [
        _markdown_cell(drug_match.drug_name),
        _markdown_cell(drug_match.drug_class),
        _markdown_cell(drug_match.drug_sub_class),
        _markdown_cell(drug_match.therapeutic_category),
        _markdown_cell(drug_match.route_of_administration),
        _markdown_cell(drug_match.formulation),
        _markdown_cell(", ".join(drug_match.dosage_strengths)),
    ]


def format_drug_answer_as_markdown_table(drug_answer: DrugAnswer) -> str:
    """Return one Markdown table with a row per matched drug, or a no-match message."""
    if not drug_answer.found or not drug_answer.matches:
        return NO_MATCH_MESSAGE
    return tabulate(
        [_drug_table_row(drug_match) for drug_match in drug_answer.matches],
        headers=TABLE_HEADERS,
        tablefmt="pipe",
        disable_numparse=True,
    )
