"""
Pharmaceutical Data ETL Pipeline

Extracts, transforms, and standardizes drug data from multiple sources.
Processes drug classification data from Excel and CSV files to create
a unified, deduplicated dataset for pharmaceutical knowledge base.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from config_loader import IngestionConfig, ETLParserConfig, SearchTextConfig

logger = logging.getLogger(__name__)


def _setup_logging(
    log_level: str = "ERROR",
    log_file: Optional[str] = None,
    format_str: str = "%(message)s",
):
    """Setup logging configuration from config values."""
    # Convert string log level to logging constant
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = log_level_map.get(log_level.upper(), logging.ERROR)

    handlers = [logging.StreamHandler()]

    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,  # Override existing configuration
    )


@dataclass
class StandardizedDrugRecord:
    """Standardized drug record with all relevant fields."""

    drug_name: str
    base_drug_name: Optional[str] = None  # Core drug name without strength/route
    drug_class: Optional[str] = None
    drug_sub_class: Optional[str] = None
    therapeutic_category: Optional[str] = None
    dosages: List[str] = field(default_factory=list)
    route_of_administration: Optional[str] = None
    formulation: Optional[str] = None
    source_files: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Normalize and clean fields after initialization."""
        self.drug_name = self._normalize_drug_name(self.drug_name)
        self.base_drug_name = self._clean_field(self.base_drug_name)
        self.drug_class = self._clean_field(self.drug_class)
        self.drug_sub_class = self._clean_field(self.drug_sub_class)
        self.therapeutic_category = self._clean_field(self.therapeutic_category)
        self.route_of_administration = self._clean_field(self.route_of_administration)
        self.formulation = self._clean_field(self.formulation)
        self.dosages = self._normalize_dosages(self.dosages)

    @staticmethod
    def _normalize_drug_name(name: str) -> str:
        """Normalize drug name for consistency."""
        if not name or pd.isna(name):
            return ""

        # Convert to string and strip whitespace
        normalized = str(name).strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Capitalize properly (handle cases like "mg", "MG")
        # Keep dosage units lowercase
        words = normalized.split()
        capitalized_words = []

        for word in words:
            # Keep dosage units lowercase
            if re.match(
                r"^\d+\.?\d*(mg|ml|mcg|g|l|tablet|capsule|oral|injection)$",
                word.lower(),
            ):
                capitalized_words.append(word.lower())
            else:
                capitalized_words.append(word.capitalize())

        return " ".join(capitalized_words)

    @staticmethod
    def _clean_field(value: Optional[str]) -> Optional[str]:
        """Clean and normalize field values."""
        if not value or pd.isna(value):
            return None

        cleaned = str(value).strip()

        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Return None for empty or placeholder values
        if cleaned.lower() in ["", "na", "n/a", "none", "null", "unknown"]:
            return None

        return cleaned

    @staticmethod
    def _normalize_dosages(dosages: List[str]) -> List[str]:
        """Normalize and deduplicate dosages."""
        if not dosages:
            return []

        normalized = set()

        for dosage in dosages:
            if not dosage or pd.isna(dosage):
                continue

            # Clean dosage string
            cleaned = str(dosage).strip().lower()

            # Skip empty or placeholder values
            if cleaned in ["", "na", "n/a", "none", "null"]:
                continue

            normalized.add(cleaned)

        # Sort for consistency
        return sorted(list(normalized))

    def merge_with(self, other: "StandardizedDrugRecord") -> None:
        """Merge another record into this one, filling missing fields."""
        if not self.base_drug_name and other.base_drug_name:
            self.base_drug_name = other.base_drug_name

        if not self.drug_class and other.drug_class:
            self.drug_class = other.drug_class

        if not self.drug_sub_class and other.drug_sub_class:
            self.drug_sub_class = other.drug_sub_class

        if not self.therapeutic_category and other.therapeutic_category:
            self.therapeutic_category = other.therapeutic_category

        if not self.route_of_administration and other.route_of_administration:
            self.route_of_administration = other.route_of_administration

        if not self.formulation and other.formulation:
            self.formulation = other.formulation

        # Merge dosages
        if other.dosages:
            combined_dosages = set(self.dosages) | set(other.dosages)
            self.dosages = sorted(list(combined_dosages))

        # Merge source files
        self.source_files.update(other.source_files)

    def generate_search_text(
        self, search_text_config: Optional["SearchTextConfig"] = None
    ) -> str:
        """Generate rich search text for embedding using config template if provided."""
        # Use config if provided, otherwise fall back to hardcoded format
        if search_text_config:
            # Extract all template placeholder keys from template string
            template_placeholders = set(
                re.findall(r"\{(\w+)\}", search_text_config.template)
            )

            # Build field values
            drug_name_display = (
                self.base_drug_name if self.base_drug_name else self.drug_name
            )

            dosages_str = ""
            if self.dosages:
                dosages_str = search_text_config.dosage_delimiter.join(self.dosages)

            # Build template values
            # Map config key "route_of_administration" to template key "route"
            template_values = {
                "drug_name": drug_name_display,
                "dosages": dosages_str,
                "formulation": self.formulation or "",
                "route": self.route_of_administration or "",
                "drug_class": self.drug_class or "",
                "drug_sub_class": self.drug_sub_class or "",
                "therapeutic_category": self.therapeutic_category or "",
            }

            # Map config field names to template field names
            # Config uses "route_of_administration" but template uses "route"
            field_mapping = {
                "route": "route_of_administration",  # Map template key to config key
            }

            # Filter fields based on include_fields config and ensure all template keys are present
            filtered_values = {}

            # First, process fields from template_values
            for field_name, value in template_values.items():
                # Get config field name (may differ from template field name)
                config_field_name = field_mapping.get(field_name, field_name)
                include_field = search_text_config.include_fields.get(
                    config_field_name, True
                )

                if include_field:
                    if search_text_config.include_empty_fields or value:
                        # Use placeholder if empty and include_empty_fields is False
                        filtered_values[field_name] = (
                            value
                            if value
                            else search_text_config.empty_field_placeholder
                        )
                    elif value:
                        filtered_values[field_name] = value

            # Ensure all template placeholders are present in filtered_values
            # Use empty placeholder for missing keys to prevent KeyError
            for placeholder_key in template_placeholders:
                if placeholder_key not in filtered_values:
                    # Check if field should be included
                    config_field_name = field_mapping.get(
                        placeholder_key, placeholder_key
                    )
                    include_field = search_text_config.include_fields.get(
                        config_field_name, True
                    )

                    if include_field:
                        # Field should be included but is missing, use empty placeholder
                        filtered_values[placeholder_key] = (
                            search_text_config.empty_field_placeholder
                        )
                    else:
                        # Field excluded from config, use empty placeholder for template compatibility
                        filtered_values[placeholder_key] = (
                            search_text_config.empty_field_placeholder
                        )

            # Format using template
            try:
                return search_text_config.template.format(**filtered_values)
            except KeyError as e:
                # If template has keys not in filtered_values, fall back to default
                logger.warning(f"Template key error: {e}, using default format")
                # Fall through to default format

        # Default format (backward compatible)
        parts = []

        # Drug name is always present (use base_drug_name if available, otherwise drug_name)
        drug_name_display = (
            self.base_drug_name if self.base_drug_name else self.drug_name
        )
        parts.append(f"Drug: {drug_name_display}")

        # Add dosages if available
        if self.dosages:
            dosages_str = ", ".join(self.dosages)
            parts.append(f"Dosages: [{dosages_str}]")

        # Add classifications
        if self.drug_class:
            parts.append(f"Class: {self.drug_class}")

        if self.drug_sub_class:
            parts.append(f"Subclass: {self.drug_sub_class}")

        if self.therapeutic_category:
            parts.append(f"Therapeutic: {self.therapeutic_category}")

        # Add route of administration if available
        if self.route_of_administration:
            parts.append(f"Route of Administration: {self.route_of_administration}")

        # Add formulation if available
        if self.formulation:
            parts.append(f"Formulation: {self.formulation}")

        return " | ".join(parts)

    def to_dict(self, search_text_config: Optional["SearchTextConfig"] = None) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "drug_name": self.drug_name,
            "base_drug_name": self.base_drug_name,
            "drug_class": self.drug_class,
            "drug_sub_class": self.drug_sub_class,
            "therapeutic_category": self.therapeutic_category,
            "dosages": self.dosages,
            "route_of_administration": self.route_of_administration,
            "formulation": self.formulation,
            "search_text": self.generate_search_text(search_text_config),
            "source_files": list(self.source_files),
        }


class DataExtractor:
    """Extract data from various file formats."""

    @staticmethod
    def extract_from_csv(file_path: str) -> pd.DataFrame:
        """Extract data from CSV file with encoding fallback."""
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            return df
        except UnicodeDecodeError:
            # Try different encoding
            df = pd.read_csv(file_path, encoding="latin-1")
            return df
        except Exception as e:
            logger.error(f"Failed to extract CSV {file_path}: {e}")
            raise

    @staticmethod
    def extract_from_excel(
        file_path: str, sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract data from Excel file."""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                # Read first sheet by default
                df = pd.read_excel(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to extract Excel {file_path}: {e}")
            raise

    @staticmethod
    def detect_and_extract(file_path: str) -> pd.DataFrame:
        """Automatically detect file type and extract."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return DataExtractor.extract_from_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return DataExtractor.extract_from_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class DataTransformer:
    """Transform raw data into standardized format."""

    def __init__(self, parser_config: Optional["ETLParserConfig"] = None):
        """
        Initialize DataTransformer with optional parser configuration.

        Args:
            parser_config: ETLParserConfig containing recognized_routes and recognized_formulations
        """
        if parser_config:
            self.recognized_routes = set(
                route.lower() for route in parser_config.recognized_routes
            )
            self.recognized_formulations = set(
                form.lower() for form in parser_config.recognized_formulations
            )
        else:
            # Default fallback values for backward compatibility
            self.recognized_routes = {
                "oral",
                "intravenous",
                "subcutaneous",
                "intramuscular",
                "topical",
                "inhalation",
                "nasal",
                "ophthalmic",
                "otic",
                "rectal",
                "vaginal",
                "transdermal",
                "sublingual",
                "buccal",
                "injection",
                "infusion",
            }
            self.recognized_formulations = {
                "tablet",
                "capsule",
                "solution",
                "suspension",
                "cream",
                "ointment",
                "gel",
                "lotion",
                "patch",
                "spray",
                "powder",
                "syrup",
                "elixir",
                "aerosol",
                "film",
                "suppository",
                "insert",
                "device",
            }

    def parse_drug_details(self, drug_name: str) -> Dict[str, Any]:
        """
        Enhanced parser to extract drug components: base name, strength(s), route, and formulation.

        Handles formats like:
        - "Cimetidine Tablet 800 MG Oral"
        - "Aspirin 81 MG Delayed Release Oral Tablet"
        - "Insulin Glargine 100 UNT/ML Subcutaneous Solution"
        - "Amoxicillin 250 MG/5ML Oral Suspension"

        Returns:
        - dict with keys: base_name, strengths, route, formulation, original_name
        """
        if not drug_name or pd.isna(drug_name):
            return {
                "base_name": "",
                "strengths": [],
                "route": None,
                "formulation": None,
                "original_name": drug_name,
            }

        drug_name = str(drug_name).strip()
        tokens = drug_name.split()

        if len(tokens) < 2:
            return {
                "base_name": drug_name,
                "strengths": [],
                "route": None,
                "formulation": None,
                "original_name": drug_name,
            }

        # Use instance variables for routes and formulations
        routes = self.recognized_routes
        formulations = self.recognized_formulations

        # Extract route (usually last token)
        route = None
        route_index = None
        if tokens[-1].lower() in routes:
            route = tokens[-1].lower()
            route_index = len(tokens) - 1

        # Extract formulation (look backwards from route)
        formulation = None
        formulation_index = None
        search_start = route_index if route_index else len(tokens)

        for i in range(search_start - 1, -1, -1):
            if tokens[i].lower() in formulations:
                formulation = tokens[i].lower()
                formulation_index = i
                break

        # Extract strength(s) - look for patterns like "800 MG" or "100 UNT/ML"
        strength_pattern = r"\b(\d+\.?\d*)\s*([A-Za-z]+(?:/[A-Za-z]+)?)\b"
        strength_units = {
            "mg",
            "ml",
            "mcg",
            "g",
            "l",
            "kg",
            "unt",
            "unit",
            "units",
            "iu",
            "meq",
            "mmol",
            "%",
            "mg/ml",
            "mcg/ml",
            "unt/ml",
            "mg/g",
        }

        strengths = []
        strength_positions = []

        # Find all strength matches in the entire string
        full_text = " ".join(tokens)
        for match in re.finditer(strength_pattern, full_text, re.IGNORECASE):
            value = match.group(1)
            unit = match.group(2).lower()

            # Check if unit is valid
            if unit in strength_units or "/" in unit:
                strength_str = f"{value} {unit.upper()}"
                strengths.append(strength_str)

                # Find position in tokens
                match_text = match.group(0)
                for idx, token in enumerate(tokens):
                    if match_text.lower() in " ".join(tokens[idx : idx + 3]).lower():
                        strength_positions.extend(range(idx, min(idx + 3, len(tokens))))

        # Determine where base name ends
        # It should be before the first strength or formulation
        base_name_end = len(tokens)

        if strength_positions:
            base_name_end = min(base_name_end, min(strength_positions))

        if formulation_index is not None:
            base_name_end = min(base_name_end, formulation_index)

        # Extract base name
        base_name_tokens = tokens[:base_name_end]

        # Remove common descriptors that might slip into base name
        descriptors = {
            "delayed",
            "release",
            "extended",
            "immediate",
            "modified",
            "sustained",
        }
        base_name_tokens = [t for t in base_name_tokens if t.lower() not in descriptors]

        base_name = " ".join(base_name_tokens).strip()

        # If no base name extracted, use first 1-2 tokens
        if not base_name and len(tokens) >= 1:
            base_name = tokens[0]

        return {
            "base_name": base_name,
            "strengths": strengths,
            "route": route,
            "formulation": formulation,
            "original_name": drug_name,
        }

    def extract_dosage_from_drug_name(
        self, drug_name: str
    ) -> Tuple[str, Optional[str]]:
        """
        Extract dosage and clean name using enhanced parser.
        Returns: (clean_name, primary_strength)
        """
        parsed = self.parse_drug_details(drug_name)

        # Reconstruct clean name without strengths
        clean_parts = []

        if parsed["base_name"]:
            clean_parts.append(parsed["base_name"])

        if parsed["formulation"]:
            clean_parts.append(parsed["formulation"].capitalize())

        if parsed["route"]:
            clean_parts.append(parsed["route"].capitalize())

        clean_name = " ".join(clean_parts) if clean_parts else parsed["original_name"]

        # Return primary strength (first one if multiple)
        primary_strength = parsed["strengths"][0] if parsed["strengths"] else None

        return clean_name, primary_strength

    def transform_format_1(
        self, df: pd.DataFrame, source_file: str
    ) -> List[StandardizedDrugRecord]:
        """
        Transform dataset with format: drug_name, drug_class, drug_sub_class
        (Expected from drug_class_subclass_reference.xlsx)
        """
        records = []

        # Expected columns (case-insensitive mapping)
        column_mapping = {
            "drug_name": ["drug_name", "drugname", "name", "drug"],
            "drug_class": ["drug_class", "drugclass", "class"],
            "drug_sub_class": [
                "drug_sub_class",
                "drugsubclass",
                "sub_class",
                "subclass",
            ],
        }

        # Find actual column names
        df_columns_lower = {col.lower(): col for col in df.columns}
        actual_columns = {}

        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df_columns_lower:
                    actual_columns[standard_name] = df_columns_lower[possible_name]
                    break

        if "drug_name" not in actual_columns:
            raise ValueError(f"Could not find drug_name column in {source_file}")

        for idx, row in df.iterrows():
            drug_name_raw = row.get(actual_columns["drug_name"])

            if not drug_name_raw or pd.isna(drug_name_raw):
                continue

            # Parse drug details using enhanced parser
            parsed_details = self.parse_drug_details(str(drug_name_raw))

            # Use clean name as drug_name, keep original as base_drug_name
            drug_name_clean = parsed_details["base_name"]
            if parsed_details["formulation"]:
                drug_name_clean += f" {parsed_details['formulation'].capitalize()}"
            if parsed_details["route"]:
                drug_name_clean += f" {parsed_details['route'].capitalize()}"

            record = StandardizedDrugRecord(
                drug_name=drug_name_clean,
                base_drug_name=parsed_details["base_name"],
                drug_class=row.get(actual_columns.get("drug_class"))
                if "drug_class" in actual_columns
                else None,
                drug_sub_class=row.get(actual_columns.get("drug_sub_class"))
                if "drug_sub_class" in actual_columns
                else None,
                dosages=parsed_details["strengths"],
                route_of_administration=parsed_details["route"],
                formulation=parsed_details["formulation"],
                source_files={source_file},
            )

            records.append(record)

        return records

    def transform_format_2(
        self, df: pd.DataFrame, source_file: str
    ) -> List[StandardizedDrugRecord]:
        """
        Transform dataset with format:
        ingredient_rxcui, primary_drug_name, therapeutic_class_l2, drug_class_l3, drug_subclass_l4
        (Expected from rxnorm_complete_drug_classification.csv)
        """
        records = []

        # Expected columns (case-insensitive mapping)
        column_mapping = {
            "drug_name": ["primary_drug_name", "drug_name", "ingredient_name", "name"],
            "therapeutic_category": [
                "therapeutic_class_l2",
                "therapeutic_class",
                "therapeutic_category",
            ],
            "drug_class": ["drug_class_l3", "drug_class", "class"],
            "drug_sub_class": ["drug_subclass_l4", "drug_sub_class", "subclass"],
        }

        # Find actual column names
        df_columns_lower = {col.lower(): col for col in df.columns}
        actual_columns = {}

        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df_columns_lower:
                    actual_columns[standard_name] = df_columns_lower[possible_name]
                    break

        if "drug_name" not in actual_columns:
            raise ValueError(f"Could not find drug_name column in {source_file}")

        for idx, row in df.iterrows():
            drug_name_raw = row.get(actual_columns["drug_name"])

            if not drug_name_raw or pd.isna(drug_name_raw):
                continue

            # Parse drug details using enhanced parser
            parsed_details = self.parse_drug_details(str(drug_name_raw))

            # Use clean name as drug_name, keep original as base_drug_name
            drug_name_clean = parsed_details["base_name"]
            if parsed_details["formulation"]:
                drug_name_clean += f" {parsed_details['formulation'].capitalize()}"
            if parsed_details["route"]:
                drug_name_clean += f" {parsed_details['route'].capitalize()}"

            record = StandardizedDrugRecord(
                drug_name=drug_name_clean,
                base_drug_name=parsed_details["base_name"],
                drug_class=row.get(actual_columns.get("drug_class"))
                if "drug_class" in actual_columns
                else None,
                drug_sub_class=row.get(actual_columns.get("drug_sub_class"))
                if "drug_sub_class" in actual_columns
                else None,
                therapeutic_category=row.get(actual_columns.get("therapeutic_category"))
                if "therapeutic_category" in actual_columns
                else None,
                dosages=parsed_details["strengths"],
                route_of_administration=parsed_details["route"],
                formulation=parsed_details["formulation"],
                source_files={source_file},
            )

            records.append(record)

        return records

    def auto_detect_and_transform(
        self, df: pd.DataFrame, source_file: str
    ) -> List[StandardizedDrugRecord]:
        """Automatically detect dataset format and transform."""
        df_columns_lower = {col.lower() for col in df.columns}

        # Check for format 2 indicators
        if any(
            col in df_columns_lower
            for col in ["therapeutic_class_l2", "primary_drug_name", "ingredient_rxcui"]
        ):
            return self.transform_format_2(df, source_file)
        else:
            return self.transform_format_1(df, source_file)


class DataDeduplicator:
    """Deduplicate and merge drug records."""

    @staticmethod
    def create_deduplication_key(drug_name: str) -> str:
        """Create normalized key for deduplication."""
        # Remove all non-alphanumeric characters and convert to lowercase
        key = re.sub(r"[^a-z0-9]", "", drug_name.lower())
        return key

    @staticmethod
    def deduplicate_records(
        records: List[StandardizedDrugRecord],
    ) -> List[StandardizedDrugRecord]:
        """Deduplicate records by drug name and merge information."""
        # Group by deduplication key (use base_drug_name if available, otherwise drug_name)
        grouped_records = defaultdict(list)

        for record in records:
            # Use base_drug_name for deduplication if available, otherwise use drug_name
            dedup_key_name = (
                record.base_drug_name if record.base_drug_name else record.drug_name
            )
            if not dedup_key_name:
                continue

            key = DataDeduplicator.create_deduplication_key(dedup_key_name)
            grouped_records[key].append(record)

        # Merge duplicates
        deduplicated_records = []

        for key, duplicate_records in grouped_records.items():
            if len(duplicate_records) == 1:
                deduplicated_records.append(duplicate_records[0])
            else:
                # Merge all duplicates into first record
                primary_record = duplicate_records[0]

                for duplicate in duplicate_records[1:]:
                    primary_record.merge_with(duplicate)

                deduplicated_records.append(primary_record)

        return deduplicated_records


class PharmaceuticalETLPipeline:
    """Main ETL pipeline orchestrator."""

    def __init__(self, config: Optional["IngestionConfig"] = None):
        """
        Initialize ETL pipeline with optional configuration.

        Args:
            config: IngestionConfig containing all pipeline configuration
        """
        self.config = config

        # Setup logging from config if provided
        if config:
            _setup_logging(log_level=config.etl.log_level, log_file=config.etl.log_file)

        # Initialize components with config
        parser_config = config.etl.parser if config else None
        self.extractor = DataExtractor()
        self.transformer = DataTransformer(parser_config=parser_config)
        self.deduplicator = DataDeduplicator()
        self.raw_records: List[StandardizedDrugRecord] = []
        self.standardized_records: List[StandardizedDrugRecord] = []

    def extract_from_file(self, file_path: str) -> pd.DataFrame:
        """Extract data from single file."""
        return self.extractor.detect_and_extract(file_path)

    def transform_dataframe(
        self, df: pd.DataFrame, source_file: str
    ) -> List[StandardizedDrugRecord]:
        """Transform dataframe to standardized records."""
        return self.transformer.auto_detect_and_transform(df, source_file)

    def run_pipeline(self, file_paths: List[str]) -> List[StandardizedDrugRecord]:
        """Execute complete ETL pipeline."""
        print("Stage: Extract and Transform")

        # Step 1: Extract and Transform
        all_records = []

        for file_path in file_paths:
            try:
                # Extract
                df = self.extract_from_file(file_path)

                # Transform
                records = self.transform_dataframe(df, Path(file_path).name)
                all_records.extend(records)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        self.raw_records = all_records

        # Step 2: Deduplicate
        print("Stage: Deduplicate")
        self.standardized_records = self.deduplicator.deduplicate_records(all_records)

        # Step 3: Statistics
        self._print_statistics()

        print("Stage: Complete")

        return self.standardized_records

    def _print_statistics(self):
        """Print pipeline statistics."""
        print("\nStatistics:")
        print(f"  Raw records: {len(self.raw_records)}")
        print(f"  Standardized records: {len(self.standardized_records)}")
        print(
            f"  Duplicates removed: {len(self.raw_records) - len(self.standardized_records)}"
        )

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export standardized records to pandas DataFrame."""
        if not self.standardized_records:
            return pd.DataFrame()

        # Use search_text config if available
        search_text_config = self.config.search_text if self.config else None
        data = [
            record.to_dict(search_text_config=search_text_config)
            for record in self.standardized_records
        ]
        df = pd.DataFrame(data)

        return df

    def save_to_csv(self, output_path: str):
        """Save standardized records to CSV."""
        print("Stage: Export")
        df = self.export_to_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # Import here to avoid circular import issues
    from config_loader import ConfigLoader

    # Load configuration
    config_file = Path(__file__).parent / "ingestion_config.yaml"
    config_loader = ConfigLoader(config_file)
    config = config_loader.load()

    # Initialize pipeline with config
    pipeline = PharmaceuticalETLPipeline(config=config)

    # Use file paths from config
    base_path = Path(__file__).parent.parent
    file_paths = []

    for file_path in config.data_sources.input_files:
        # Resolve relative paths relative to config file location
        config_dir = config_file.parent
        resolved_path = (config_dir / file_path).resolve()
        if resolved_path.exists():
            file_paths.append(str(resolved_path))
        else:
            # Try relative to project root
            project_path = (base_path / file_path).resolve()
            if project_path.exists():
                file_paths.append(str(project_path))
            else:
                logger.warning(f"Input file not found: {file_path}")

    if not file_paths:
        raise ValueError("No valid input files found. Check configuration.")

    # Run pipeline
    standardized_records = pipeline.run_pipeline(file_paths)

    # Save output using config paths
    output_dir = base_path / config.data_sources.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / config.data_sources.standardized_data_file)
    pipeline.save_to_csv(output_path)
