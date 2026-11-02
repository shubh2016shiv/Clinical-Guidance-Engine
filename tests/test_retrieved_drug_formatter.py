"""Verify LLM input tables preserve retrieved data and Markdown row boundaries."""

import copy
import json
import unittest

from src.drugs_agent.retrieved_drug_formatter import format_retrieved_drug_table


class RetrievedDrugFormatterTests(unittest.TestCase):
    def test_columns_cover_sparse_records_without_changing_order_or_source_data(self) -> None:
        records = [
            {"drug_name": "First", "formulation": "tablet"},
            {"drug_name": "Second", "dosage_strengths": ["5 mg", "10 mg"]},
        ]
        original = copy.deepcopy(records)

        lines = format_retrieved_drug_table(records).splitlines()

        self.assertEqual(
            [cell.strip() for cell in lines[0].split("|")[1:-1]],
            ["drug_name", "formulation", "dosage_strengths"],
        )
        first = [cell.strip() for cell in lines[2].split("|")[1:-1]]
        second = [cell.strip() for cell in lines[3].split("|")[1:-1]]
        self.assertEqual(
            [json.loads(cell) if cell else None for cell in first], ["First", "tablet", None]
        )
        self.assertEqual(
            [json.loads(cell) if cell else None for cell in second],
            ["Second", None, ["5 mg", "10 mg"]],
        )
        self.assertEqual(records, original)

    def test_special_characters_and_json_types_round_trip_without_extra_table_rows(self) -> None:
        record = {
            "drug_name": 'A|B\nC\\D "quoted" µg',
            "code": "00123",
            "strengths": ["5|10 mg", "20 mg"],
            "count": 0,
            "available": False,
        }

        lines = format_retrieved_drug_table([record]).splitlines()

        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[2].count("|"), len(record) + 1)
        cells = [json.loads(cell.strip()) for cell in lines[2].split("|")[1:-1]]
        self.assertEqual(cells, list(record.values()))
        self.assertIn(r"\u007c", lines[2])
        self.assertIn(r"\n", lines[2])

    def test_empty_results_are_explicit_and_do_not_invent_a_record(self) -> None:
        self.assertEqual(format_retrieved_drug_table([]), "No drug records were retrieved.")
        self.assertEqual(
            format_retrieved_drug_table([{}]), "Retrieved drug records contain no fields."
        )
