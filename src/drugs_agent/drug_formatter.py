"""Render a drug answer as a Markdown list grouped by class."""

import re

from .schema import DrugAnswer, DrugMatch

NO_MATCH_MESSAGE = "No matching drugs were found."
LIST_HEADING = "**Available in the catalog**"
UNGROUPED_HEADING = "Other"
_MARKDOWN_EMPHASIS = re.compile(r"([*_])")


def _escape_emphasis(text: str) -> str:
    """Keep an asterisk or underscore in a value from starting bold or italic text."""
    return _MARKDOWN_EMPHASIS.sub(r"\\\1", text)


def _group_heading(drug_match: DrugMatch) -> str:
    """Name the group a drug belongs to, preferring the most readable label it carries.

    The subclass comes first because the catalog writes it in sentence case, while
    ``drug_class`` repeats it in capitals.
    """
    return (
        drug_match.drug_sub_class
        or drug_match.drug_class
        or drug_match.therapeutic_category
        or UNGROUPED_HEADING
    )


def _drug_details(drug_match: DrugMatch) -> str:
    """Join the route, form and strengths a drug has, leaving out the ones it lacks."""
    route_and_form = " ".join(
        detail for detail in (drug_match.route_of_administration, drug_match.formulation) if detail
    )
    strengths = ", ".join(drug_match.dosage_strengths)
    return ", ".join(detail for detail in (route_and_form, strengths) if detail)


def format_drug_answer_as_markdown_list(drug_answer: DrugAnswer) -> str:
    """Return the matched drugs as bullets grouped by class, or a no-match message.

    Groups and the drugs inside them keep the order they were matched in, which runs
    from the most relevant class searched to the least.
    """
    if not drug_answer.found or not drug_answer.matches:
        return NO_MATCH_MESSAGE

    drugs_by_group: dict[str, list[DrugMatch]] = {}
    for drug_match in drug_answer.matches:
        drugs_by_group.setdefault(_group_heading(drug_match), []).append(drug_match)

    markdown_lines = [LIST_HEADING]
    for group_heading, grouped_drugs in drugs_by_group.items():
        markdown_lines.append("")
        markdown_lines.append(f"*{_escape_emphasis(group_heading)}*")
        for drug_match in grouped_drugs:
            bullet = f"- **{_escape_emphasis(drug_match.drug_name)}**"
            drug_details = _drug_details(drug_match)
            if drug_details:
                bullet += f" — {_escape_emphasis(drug_details)}"
            markdown_lines.append(bullet)
    return "\n".join(markdown_lines)
