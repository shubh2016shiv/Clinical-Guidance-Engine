"""Exercise the drug ETL: search text rendering, drug name parsing, dedup, failures."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "drug_data_ingestion_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from config_loader import (  # noqa: E402
    SEARCH_TEXT_FIELDS,
    DataSourceConfig,
    ETLConfig,
    IngestionConfig,
    MilvusCollectionConfig,
    MilvusIndexConfig,
    OpenAIConfig,
    SearchTextConfig,
)
from drugs_etl_pipeline import (  # noqa: E402
    DataDeduplicator,
    DataTransformer,
    PharmaceuticalETLPipeline,
    StandardizedDrugRecord,
)
from pydantic import ValidationError  # noqa: E402


def record(**overrides):
    fields = {
        "drug_name": "Metformin Tablet Oral",
        "base_drug_name": "Metformin",
        "drug_class": "Antidiabetic",
        "drug_sub_class": "Biguanide",
        "therapeutic_category": "Endocrine",
        "dosages": ["500 mg", "850 mg"],
        "route_of_administration": "oral",
        "formulation": "tablet",
    }
    fields.update(overrides)
    return StandardizedDrugRecord(**fields)


def etl_config(input_files=(), max_errors=2):
    """A minimal config that logs to the null device.

    The pipeline opens a FileHandler on log_file and holds it open, which on Windows
    would block cleanup of any temporary directory the log lived in.
    """
    return IngestionConfig(
        openai=OpenAIConfig(api_key="test-key"),
        data_sources=DataSourceConfig(input_files=list(input_files)),
        etl=ETLConfig(log_file=os.devnull),
        error_handling={"max_errors": max_errors},
    )


class SearchTextTests(unittest.TestCase):
    """The text that gets embedded, stored in Milvus and written to the CSV."""

    def test_complete_record_renders_every_segment(self) -> None:
        text = record().generate_search_text(SearchTextConfig())

        self.assertEqual(
            text,
            "Drug: Metformin | Dosages: [500 mg, 850 mg] | Form: tablet | Route: oral "
            "| Class: Antidiabetic | Subclass: Biguanide | Therapeutic: Endocrine",
        )

    def test_sparse_record_drops_empty_segments_instead_of_leaving_blank_labels(self) -> None:
        sparse = record(
            dosages=[],
            formulation=None,
            route_of_administration=None,
            therapeutic_category=None,
        )

        text = sparse.generate_search_text(SearchTextConfig())

        self.assertEqual(text, "Drug: Metformin | Class: Antidiabetic | Subclass: Biguanide")
        for filler in ("Dosages: []", "Form:", "Route:", "Therapeutic:"):
            self.assertNotIn(filler, text)

    def test_the_same_string_is_embedded_and_written_to_the_csv(self) -> None:
        config = SearchTextConfig()
        drug = record()

        self.assertEqual(drug.generate_search_text(config), drug.to_dict(config)["search_text"])

    def test_config_is_required_so_no_call_site_can_fall_back_to_another_format(self) -> None:
        with self.assertRaises(TypeError):
            record().generate_search_text()
        with self.assertRaises(TypeError):
            record().to_dict()

    def test_separator_and_dosage_delimiter_are_honoured(self) -> None:
        config = SearchTextConfig(
            segments=["Drug: {drug_name}", "Dosages: {dosages}"],
            separator=" ;; ",
            dosage_delimiter=" + ",
        )

        self.assertEqual(
            record().generate_search_text(config),
            "Drug: Metformin ;; Dosages: 500 mg + 850 mg",
        )

    def test_drug_name_falls_back_to_the_full_name_when_no_base_name_was_parsed(self) -> None:
        drug = record(base_drug_name=None)

        text = drug.generate_search_text(SearchTextConfig(segments=["Drug: {drug_name}"]))

        self.assertEqual(text, "Drug: Metformin Tablet Oral")

    def test_renderer_supplies_every_placeholder_the_config_advertises(self) -> None:
        """Guards SEARCH_TEXT_FIELDS and the renderer's value dict from drifting apart."""
        segments = [f"{name}={{{name}}}" for name in sorted(SEARCH_TEXT_FIELDS)]

        text = record().generate_search_text(SearchTextConfig(segments=segments))

        for name in SEARCH_TEXT_FIELDS:
            self.assertIn(f"{name}=", text)

    def test_unknown_placeholder_is_rejected_when_the_config_loads(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SearchTextConfig(segments=["Drug: {drug_nmae}"])

        self.assertIn("drug_nmae", str(caught.exception))


class ConfigStrictnessTests(unittest.TestCase):
    """A key that looks like it configures something must actually configure it."""

    def test_the_yaml_spelling_of_the_index_type_is_honoured(self) -> None:
        """The field is aliased 'index_type'; the YAML writes 'type'."""
        index = MilvusIndexConfig(**{"type": "IVF_FLAT"})

        self.assertEqual(index.type, "IVF_FLAT")
        self.assertEqual(index.params, {"nlist": 128})

    def test_the_alias_spelling_still_works(self) -> None:
        self.assertEqual(MilvusIndexConfig(**{"index_type": "IVF_FLAT"}).type, "IVF_FLAT")

    def test_an_unknown_key_is_rejected_rather_than_silently_dropped(self) -> None:
        with self.assertRaises(ValidationError):
            DataSourceConfig(encodingg="utf-8")

    def test_a_stale_nested_block_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            MilvusCollectionConfig(schema={"fields": []})


class DrugNameParsingTests(unittest.TestCase):
    """Characterisation tests: these pin current behaviour, including its limits."""

    def setUp(self) -> None:
        self.transformer = DataTransformer(ETLConfig().parser)

    def test_strength_route_and_formulation_are_split_out(self) -> None:
        parsed = self.transformer.parse_drug_details("Cimetidine Tablet 800 MG Oral")

        self.assertEqual(parsed["base_name"], "Cimetidine")
        self.assertEqual(parsed["strengths"], ["800 MG"])
        self.assertEqual(parsed["route"], "oral")
        self.assertEqual(parsed["formulation"], "tablet")

    def test_release_descriptors_are_stripped_from_the_base_name(self) -> None:
        parsed = self.transformer.parse_drug_details("Aspirin 81 MG Delayed Release Oral Tablet")

        self.assertEqual(parsed["base_name"], "Aspirin")
        self.assertEqual(parsed["strengths"], ["81 MG"])
        self.assertEqual(parsed["formulation"], "tablet")

    def test_route_is_only_detected_as_the_final_token(self) -> None:
        """Known limitation: "Oral Tablet" ends in the formulation, so the route is missed."""
        parsed = self.transformer.parse_drug_details("Aspirin 81 MG Delayed Release Oral Tablet")

        self.assertIsNone(parsed["route"])

    def test_a_compound_strength_is_split_into_two(self) -> None:
        """Known limitation: "250 MG/5ML" should be one strength, not two."""
        parsed = self.transformer.parse_drug_details("Amoxicillin 250 MG/5ML Oral Suspension")

        self.assertEqual(parsed["base_name"], "Amoxicillin")
        self.assertEqual(parsed["strengths"], ["250 MG", "5 ML"])

    def test_a_multiword_base_name_loses_its_qualifier(self) -> None:
        """Known limitation: 'Insulin Glargine' narrows to 'Insulin'.

        Latent with the current sources, which spell these names with commas
        ("Insulin Glargine, Human") and so never reach this branch.
        """
        parsed = self.transformer.parse_drug_details(
            "Insulin Glargine 100 UNT/ML Subcutaneous Solution"
        )

        self.assertEqual(parsed["base_name"], "Insulin")

    def test_a_bare_name_parses_to_itself(self) -> None:
        parsed = self.transformer.parse_drug_details("Metformin")

        self.assertEqual(parsed["base_name"], "Metformin")
        self.assertEqual(parsed["strengths"], [])

    def test_blank_input_yields_an_empty_base_name(self) -> None:
        for blank in ("", "   ", None):
            self.assertEqual(self.transformer.parse_drug_details(blank)["base_name"], "")


class DeduplicationTests(unittest.TestCase):
    def test_key_ignores_case_spacing_and_punctuation(self) -> None:
        keys = {
            DataDeduplicator.create_deduplication_key(name)
            for name in ("Metformin HCl", "metformin hcl", "Met-formin  HCL")
        }

        self.assertEqual(keys, {"metforminhcl"})

    def test_merge_fills_only_missing_fields_and_unions_the_rest(self) -> None:
        primary = record(
            drug_sub_class=None,
            therapeutic_category=None,
            dosages=["500 mg"],
            source_files={"a.csv"},
        )
        other = record(
            drug_class="SHOULD NOT OVERWRITE",
            drug_sub_class="Biguanide",
            therapeutic_category="Endocrine",
            dosages=["850 mg"],
            source_files={"b.csv"},
        )

        primary.merge_with(other)

        self.assertEqual(primary.drug_class, "Antidiabetic")
        self.assertEqual(primary.drug_sub_class, "Biguanide")
        self.assertEqual(primary.therapeutic_category, "Endocrine")
        self.assertEqual(primary.dosages, ["500 mg", "850 mg"])
        self.assertEqual(primary.source_files, {"a.csv", "b.csv"})

    def test_duplicates_are_merged_into_one_record(self) -> None:
        records = [
            record(drug_sub_class=None, source_files={"a.csv"}),
            record(therapeutic_category=None, source_files={"b.csv"}),
            record(base_drug_name="Aspirin", source_files={"c.csv"}),
        ]

        deduplicated = DataDeduplicator.deduplicate_records(records)

        self.assertEqual(len(deduplicated), 2)
        merged = next(r for r in deduplicated if r.base_drug_name == "Metformin")
        self.assertEqual(merged.source_files, {"a.csv", "b.csv"})


class PipelineFailureTests(unittest.TestCase):
    """A run that extracts nothing must fail, not report success."""

    def test_a_run_with_no_readable_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PharmaceuticalETLPipeline(etl_config(max_errors=99))

            with self.assertRaises(RuntimeError) as caught:
                pipeline.run_pipeline([str(Path(tmp) / "missing.csv")])

        self.assertIn("No records were extracted", str(caught.exception))

    def test_exceeding_max_errors_abandons_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PharmaceuticalETLPipeline(etl_config(max_errors=1))
            missing = [str(Path(tmp) / f"missing{i}.csv") for i in range(3)]

            with self.assertRaises(RuntimeError) as caught:
                pipeline.run_pipeline(missing)

        message = str(caught.exception)
        self.assertIn("exceeds error_handling.max_errors", message)
        self.assertIn("missing0.csv", message)

    def test_a_readable_file_still_produces_records_when_another_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "drugs.csv"
            source.write_text(
                "drug_name,drug_class\nMetformin 500 MG Oral,Antidiabetic\n",
                encoding="utf-8",
            )
            pipeline = PharmaceuticalETLPipeline(etl_config(max_errors=5))

            records = pipeline.run_pipeline([str(source), str(Path(tmp) / "missing.csv")])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].base_drug_name, "Metformin")


if __name__ == "__main__":
    unittest.main()
