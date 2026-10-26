import unittest

from src.drugs_agent import DrugAnswer, DrugMatch, format_drug_answer_as_markdown_list


def drug_match(name="Lisinopril", **fields):
    values = {
        "drug_class": "ACE INHIBITORS, PLAIN",
        "drug_sub_class": "ACE inhibitors, plain",
        "therapeutic_category": "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM",
        "route_of_administration": "oral",
        "formulation": "tablet",
        "dosage_strengths": ["10 mg", "20 mg"],
    }
    values.update(fields)
    return DrugMatch(drug_name=name, **values)


class DrugFormatterTests(unittest.TestCase):
    def test_a_match_becomes_a_bullet_under_its_class(self) -> None:
        markdown = format_drug_answer_as_markdown_list(
            DrugAnswer(found=True, matches=[drug_match()])
        )

        heading, blank, group, bullet = markdown.splitlines()
        self.assertEqual(heading, "**Available in the catalog**")
        self.assertEqual(blank, "")
        self.assertEqual(group, "*ACE inhibitors, plain*")
        self.assertEqual(bullet, "- **Lisinopril** — oral tablet, 10 mg, 20 mg")

    def test_drugs_of_one_class_share_a_single_heading(self) -> None:
        answer = DrugAnswer(found=True, matches=[drug_match("A"), drug_match("B")])

        markdown = format_drug_answer_as_markdown_list(answer)

        self.assertEqual(markdown.count("*ACE inhibitors, plain*"), 1)
        self.assertEqual(
            [line for line in markdown.splitlines() if line.startswith("- ")],
            ["- **A** — oral tablet, 10 mg, 20 mg", "- **B** — oral tablet, 10 mg, 20 mg"],
        )

    def test_groups_keep_the_order_their_drugs_were_matched_in(self) -> None:
        answer = DrugAnswer(
            found=True,
            matches=[
                drug_match("A"),
                drug_match("B", drug_sub_class="Thiazides, plain"),
                drug_match("C"),
            ],
        )

        groups = [
            line
            for line in format_drug_answer_as_markdown_list(answer).splitlines()
            if line.startswith("*") and not line.startswith("**")
        ]

        self.assertEqual(groups, ["*ACE inhibitors, plain*", "*Thiazides, plain*"])

    def test_a_drug_with_no_details_is_a_bare_bullet(self) -> None:
        answer = DrugAnswer(
            found=True,
            matches=[
                drug_match(route_of_administration=None, formulation=None, dosage_strengths=[])
            ],
        )

        self.assertIn("- **Lisinopril**", format_drug_answer_as_markdown_list(answer))
        self.assertNotIn("—", format_drug_answer_as_markdown_list(answer))

    def test_details_leave_out_only_the_fields_that_are_missing(self) -> None:
        answer = DrugAnswer(
            found=True, matches=[drug_match(formulation=None, dosage_strengths=["250 mg"])]
        )

        self.assertIn(
            "- **Lisinopril** — oral, 250 mg", format_drug_answer_as_markdown_list(answer)
        )

    def test_the_class_is_used_when_a_drug_has_no_subclass(self) -> None:
        answer = DrugAnswer(found=True, matches=[drug_match(drug_sub_class=None)])

        self.assertIn("*ACE INHIBITORS, PLAIN*", format_drug_answer_as_markdown_list(answer))

    def test_a_drug_with_no_class_at_all_is_grouped_as_other(self) -> None:
        answer = DrugAnswer(
            found=True,
            matches=[drug_match(drug_sub_class=None, drug_class=None, therapeutic_category=None)],
        )

        self.assertIn("*Other*", format_drug_answer_as_markdown_list(answer))

    def test_emphasis_in_a_value_cannot_break_the_bullet(self) -> None:
        answer = DrugAnswer(found=True, matches=[drug_match("A*B_C")])

        self.assertIn(r"- **A\*B\_C**", format_drug_answer_as_markdown_list(answer))

    def test_strengths_are_not_reformatted_as_numbers(self) -> None:
        answer = DrugAnswer(found=True, matches=[drug_match(dosage_strengths=["500", "0.50"])])

        self.assertIn("500, 0.50", format_drug_answer_as_markdown_list(answer))

    def test_no_match_returns_a_message_and_no_list(self) -> None:
        self.assertEqual(
            format_drug_answer_as_markdown_list(DrugAnswer(found=False, matches=[])),
            "No matching drugs were found.",
        )

    def test_found_with_no_matches_is_also_no_match(self) -> None:
        self.assertEqual(
            format_drug_answer_as_markdown_list(DrugAnswer(found=True, matches=[])),
            "No matching drugs were found.",
        )


if __name__ == "__main__":
    unittest.main()
