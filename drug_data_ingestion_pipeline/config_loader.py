"""
Configuration Loader for Pharmaceutical Data Ingestion Pipeline

Loads and validates configuration from YAML file using Pydantic for robustness.

This module provides a comprehensive configuration management system that:
- Loads configuration from YAML files
- Validates configuration values using Pydantic models
- Resolves environment variables
- Provides type safety and runtime validation
- Includes sensible defaults and error handling
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = logging.getLogger(__name__)


class OpenAIConfig(BaseModel):
    """OpenAI embedding configuration.

    ``model_name`` and ``dimensions`` must match what the application uses to
    embed queries (``OPENAI_EMBEDDING_MODEL`` and ``MILVUS_EMBEDDING_DIMENSION``).
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str
    model_name: str = Field(default="text-embedding-3-small")
    dimensions: int = Field(default=768, ge=1)
    batch_size: int = Field(default=100, ge=1, le=2048)
    retry_attempts: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=60, gt=0)


class MilvusConnectionConfig(BaseModel):
    """Milvus connection configuration"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=19530, ge=1, le=65535)
    alias: str = Field(default="default")
    timeout: int = Field(default=30, gt=0)  # seconds; passed to connections.connect
    user: str = Field(default="")
    password: str = Field(default="")
    secure: bool = Field(default=False)
    server_pem_path: str = Field(default="")
    server_name: str = Field(default="")


class MilvusIndexConfig(BaseModel):
    """Milvus index configuration"""

    # populate_by_name lets the YAML use either "type" or the "index_type" alias.
    # Without it the YAML's "type:" key was silently discarded and the default used.
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    type: str = Field(default="HNSW", alias="index_type")
    metric_type: str = Field(default="IP")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_index_type(cls, v):
        """Validate index type is supported"""
        valid_types = ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "AUTOINDEX"]
        if v not in valid_types:
            raise ValueError(f"Invalid index type: {v}. Valid: {valid_types}")
        return v

    @field_validator("metric_type")
    @classmethod
    def validate_metric_type(cls, v):
        """Validate metric type is supported by Milvus"""
        valid_metrics = ["L2", "IP"]
        if v not in valid_metrics:
            raise ValueError(f"Invalid metric type: {v}. Milvus supports: {valid_metrics}")
        return v

    @model_validator(mode="after")
    def set_default_params(self):
        """Fill in the usual params for the chosen index type when none are given."""
        if not self.params:
            self.params = {
                "HNSW": {"M": 16, "efConstruction": 200},
                "IVF_FLAT": {"nlist": 128},
                "IVF_SQ8": {"nlist": 128},
                "IVF_PQ": {"nlist": 128, "m": 8, "nbits": 8},
            }.get(self.type, {})
        return self


class MilvusCollectionConfig(BaseModel):
    """Milvus collection configuration"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="pharmaceutical_drugs")
    description: str = Field(default="Pharmaceutical drug database with embeddings for LLM lookup")
    drop_existing: bool = Field(default=False)
    embedding_dimension: int = Field(default=768, ge=1)


class MilvusInsertionConfig(BaseModel):
    """Milvus insertion configuration"""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=10, ge=1)


class MilvusPerformanceConfig(BaseModel):
    """Milvus performance tuning configuration"""

    model_config = ConfigDict(extra="forbid")

    load_on_startup: bool = Field(default=True)


class MilvusConfig(BaseModel):
    """Complete Milvus configuration"""

    model_config = ConfigDict(extra="forbid")

    connection: MilvusConnectionConfig = Field(default_factory=MilvusConnectionConfig)
    collection: MilvusCollectionConfig = Field(default_factory=MilvusCollectionConfig)
    index: MilvusIndexConfig = Field(default_factory=MilvusIndexConfig)
    insertion: MilvusInsertionConfig = Field(default_factory=MilvusInsertionConfig)
    performance: MilvusPerformanceConfig = Field(default_factory=MilvusPerformanceConfig)


class ETLParserConfig(BaseModel):
    """ETL parser configuration"""

    model_config = ConfigDict(extra="forbid")

    recognized_routes: list[str] = Field(
        default_factory=lambda: [
            # Enteral / GI tract
            "oral",
            "buccal",
            "sublingual",
            "enteral",  # via GI tract (tube, etc)
            "gastric",  # via stomach (e.g., gastrostomy)
            "nasogastric",  # via NG-tube
            "jejunal",  # via jejunal tube
            "colonic",  # via colon
            "rectal",
            "vaginal",
            "urethral",
            # Parenteral / injection / non-GI systemic
            "intravenous",
            "intra‐arterial",
            "intraosseous",
            "intramuscular",
            "subcutaneous",
            "intradermal",
            "epidural",
            "intrathecal",
            "intracerebral",
            "intracerebroventricular",
            "intraarticular",
            "intrasynovial",
            "intraperitoneal",
            "intravesical",
            "intravitreal",
            "intratumoral",
            "intrapleural",
            "intra-amniotic",
            # Topical / local non-systemic (or transdermal systemic)
            "topical",
            "transdermal",
            "cutaneous",
            "dermal",
            "ocular",  # eye
            "otic",  # ear
            "nasal",
            "inhalation",  # via lungs
            "pulmonary",  # specifically lung inhalation
            "nasal inhalation",
            "ophthalmic",
            "auricular",  # ear (alternative term)
            "mucosal",  # generic mucous membrane
            "sublabial",  # under lip
            "sublingual mucosal",
            "buccal mucosal",
            # Other/less common
            "extracorporeal",  # outside body (dialysis etc)
            "implant",  # via implanted device
            "infusion",  # continuous infusion (could be intravenous etc)
            "aerosol",  # aerosol administration
            "inhalational",  # variant
        ]
    )

    recognized_formulations: list[str] = Field(
        default_factory=lambda: [
            # Solid oral
            "tablet",
            "film-coated tablet",
            "chewable tablet",
            "dispersible tablet",
            "effervescent tablet",
            "extended-release tablet",
            "delayed-release tablet",
            "oral tablet",
            "capsule",
            "hard capsule",
            "soft capsule",
            "liquid-filled capsule",
            "pellet",
            "lozenge",
            "troche",
            "wafer",
            "oral powder",
            "granules",
            "oral granules",
            "oral film",
            # Liquid oral
            "solution",
            "oral solution",
            "syrup",
            "elixir",
            "suspension",
            "emulsion",
            "oral spray",
            "mouthwash",
            # Parenteral / Injectable
            "injection",
            "injectable solution",
            "injectable suspension",
            "infusion solution",
            "infusion",
            "lyophilized powder for reconstitution",
            # Topical / Skin
            "cream",
            "ointment",
            "gel",
            "lotion",
            "paste",
            "foam",
            "spray",
            "powder topical",
            "topical solution",
            "topical suspension",
            "medicated patch",
            "transdermal patch",
            "transdermal system",
            "medicated tape",
            "medicated pad",
            # Ophthalmic / Otic / Nasal
            "eye drops",
            "ophthalmic solution",
            "ophthalmic suspension",
            "ophthalmic gel",
            "ophthalmic ointment",
            "ear drops",
            "otic solution",
            "otic suspension",
            "nasal spray",
            "nasal drops",
            "nasal gel",
            "nasal powder",
            # Rectal / Vaginal / Urethral
            "suppository",
            "rectal cream",
            "rectal gel",
            "rectal ointment",
            "rectal solution",
            "rectal spray",
            "vaginal tablet",
            "vaginal cream",
            "vaginal gel",
            "vaginal film",
            "vaginal ring",
            "vaginal insert",
            "urethral gel",
            "urethral suppository",
            # Inhalation
            "inhaler",
            "metered-dose inhaler",
            "dry powder inhaler",
            "inhalation powder",
            "inhalation solution",
            "nebuliser solution",
            "inhalation suspension",
            # Implantable / Other
            "implant",
            "pellet implant",
            "reservoir device",
            "osmotic pump",
        ]
    )


class ETLConfig(BaseModel):
    """ETL pipeline configuration"""

    model_config = ConfigDict(extra="forbid")

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/etl_pipeline.log")
    parser: ETLParserConfig = Field(default_factory=ETLParserConfig)


class DataSourceConfig(BaseModel):
    """Data source configuration"""

    model_config = ConfigDict(extra="forbid")

    input_files: list[str] = Field(default_factory=list)
    output_directory: str = Field(default="output")
    standardized_data_file: str = Field(default="standardized_drugs.csv")
    encoding: str = Field(default="utf-8")
    fallback_encoding: str = Field(default="latin-1")


# Placeholders a search_text segment may use. These are the keys
# StandardizedDrugRecord.generate_search_text() supplies; tests assert the two agree.
SEARCH_TEXT_FIELDS = frozenset(
    {
        "drug_name",
        "dosages",
        "formulation",
        "route",
        "drug_class",
        "drug_sub_class",
        "therapeutic_category",
    }
)


class SearchTextConfig(BaseModel):
    """How a drug record becomes the text that gets embedded.

    This is the single definition of that text: the same string is embedded, stored
    in the Milvus search_text column, and written to the standardized CSV.

    Segments render in order and are joined with ``separator``. A segment whose
    placeholders are all empty is dropped, so a record with no dosage or route does
    not carry empty labels into its embedding.

    Changing any value here changes every embedding, so the collection must be
    re-ingested afterwards.
    """

    model_config = ConfigDict(extra="forbid")

    segments: list[str] = Field(
        default_factory=lambda: [
            "Drug: {drug_name}",
            "Dosages: [{dosages}]",
            "Form: {formulation}",
            "Route: {route}",
            "Class: {drug_class}",
            "Subclass: {drug_sub_class}",
            "Therapeutic: {therapeutic_category}",
        ]
    )
    separator: str = Field(default=" | ")
    dosage_delimiter: str = Field(default=", ")

    @field_validator("segments")
    @classmethod
    def validate_placeholders(cls, segments: list[str]) -> list[str]:
        """Reject unknown placeholders now, before any embedding has been paid for."""
        for segment in segments:
            unknown = set(re.findall(r"\{(\w+)\}", segment)) - SEARCH_TEXT_FIELDS
            if unknown:
                raise ValueError(
                    f"Unknown placeholder(s) {sorted(unknown)} in search_text segment "
                    f"{segment!r}. Available placeholders: {sorted(SEARCH_TEXT_FIELDS)}"
                )
        return segments


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""

    model_config = ConfigDict(extra="forbid")

    # How many input files may fail before the ETL run is abandoned.
    max_errors: int = Field(default=2, ge=0)


class IngestionConfig(BaseModel):
    """Complete ingestion configuration"""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",  # Allow extra fields in YAML that aren't defined in the model
        arbitrary_types_allowed=True,
    )

    # Main configuration sections
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    etl: ETLConfig = Field(default_factory=ETLConfig)
    openai: OpenAIConfig
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    search_text: SearchTextConfig = Field(default_factory=SearchTextConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)

    @field_validator("openai", mode="before")
    @classmethod
    def validate_openai_config(cls, v):
        """Ensure openai config has required api_key"""
        if isinstance(v, dict) and "api_key" not in v:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config."
            )
        return v


class ConfigLoader:
    """
    Load and validate configuration from YAML file using Pydantic.

    Automatically loads .env file from the same directory as the config file
    to populate environment variables (e.g., OPENAI_API_KEY).

    The .env file should be located in the same directory as ingestion_config.yaml.
    If not found there, it will attempt to load from the current working directory.
    """

    def __init__(self, config_path: str | Path = "ingestion_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

        # Load .env file from the same directory as config file
        self._load_env_file()

    def _load_env_file(self):
        """Load .env file from the same directory as the config file."""
        if not HAS_DOTENV:
            self.logger.warning(
                "python-dotenv not installed. Install it to load .env files automatically."
            )
            return

        # Try to load .env from config file's directory
        config_dir = self.config_path.parent.resolve()
        env_file = config_dir / ".env"

        if env_file.exists():
            load_dotenv(env_file, override=False)  # Don't override existing env vars
            self.logger.info(f"Loaded .env file from: {env_file}")
        else:
            # Try loading from current working directory as fallback
            cwd_env = Path.cwd() / ".env"
            if cwd_env.exists():
                load_dotenv(cwd_env, override=False)
                self.logger.info(f"Loaded .env file from: {cwd_env}")
            else:
                self.logger.info(
                    "[DEBUG] No .env file found. Environment variables must be set manually."
                )

    def load(self) -> IngestionConfig:
        """Load configuration from YAML file with validation"""

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.logger.info(f"Loading configuration from: {self.config_path}")

        try:
            with open(self.config_path, encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e

        # Resolve environment variables (now .env file is already loaded)
        self._resolve_environment_variables()

        # Build and validate configuration using Pydantic
        try:
            config = IngestionConfig(**self.config_data)
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

        # Additional custom validations
        self._validate_file_paths(config)
        self._validate_insertion_batch_size(config)
        self._validate_dimensions(config)

        self.logger.info("Configuration loaded and validated successfully")
        return config

    def _resolve_environment_variables(self):
        """Resolve ${VAR_NAME} environment variables in config"""

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                env_value = os.getenv(var_name)
                if env_value is None:
                    raise ValueError(f"Environment variable not set: {var_name}")
                self.logger.debug(f"Resolved environment variable: {var_name}")
                return env_value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        self.config_data = resolve_value(self.config_data)

    def _validate_file_paths(self, config: IngestionConfig):
        """Warn about input files that cannot be found from the current directory.

        Paths in the config are relative, so a miss here may only mean the pipeline
        was started from a different directory. The ETL run itself fails if no file
        yields records, so this stays a warning.
        """
        if not config.data_sources.input_files:
            raise ValueError("No input files specified in configuration")

        missing = [p for p in config.data_sources.input_files if not Path(p).exists()]
        if missing:
            self.logger.warning(f"Input files not found from {Path.cwd()}: {missing}")

    def _validate_insertion_batch_size(self, config: IngestionConfig):
        """Validate the Milvus insertion batch size is positive"""
        if config.milvus.insertion.batch_size < 1:
            raise ValueError("Milvus insertion batch size must be positive")

    def _validate_dimensions(self, config: IngestionConfig):
        """Validate that embedding dimensions match between OpenAI and Milvus"""
        openai_dims = config.openai.dimensions
        milvus_dims = config.milvus.collection.embedding_dimension

        if openai_dims != milvus_dims:
            error_msg = (
                f"Embedding dimension mismatch: openai.dimensions={openai_dims} "
                f"but milvus.collection.embedding_dimension={milvus_dims}. "
                f"They must match for embeddings to be inserted into Milvus."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Embedding dimensions validated: {openai_dims} (OpenAI and Milvus match)")

    def save_config(self, config: IngestionConfig, output_path: str | Path):
        """Save configuration to YAML file"""

        output_path = Path(output_path)

        # Convert config to dict, masking sensitive data
        config_dict = config.model_dump()

        # Mask API key for security
        if "openai" in config_dict and "api_key" in config_dict["openai"]:
            config_dict["openai"]["api_key"] = "${OPENAI_API_KEY}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise OSError(f"Failed to save configuration to {output_path}: {e}") from e

        self.logger.info(f"Configuration saved to: {output_path}")

    def get_config_template(self) -> str:
        """Generate a configuration template as YAML string"""
        template_config = IngestionConfig(openai=OpenAIConfig(api_key="${OPENAI_API_KEY}"))
        return yaml.dump(template_config.model_dump(), default_flow_style=False, sort_keys=False)


# Usage example and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Load configuration
        loader = ConfigLoader("ingestion_config.yaml")
        config = loader.load()

        # Display key configuration values
        print("=== Configuration Loaded Successfully ===")
        print(f"Embedding Model: {config.openai.model_name}")
        print(f"Embedding Dimensions: {config.openai.dimensions}")
        print(f"Milvus Host: {config.milvus.connection.host}:{config.milvus.connection.port}")
        print(f"Collection Name: {config.milvus.collection.name}")
        print(f"Index Type: {config.milvus.index.type}")
        print(f"Metric Type: {config.milvus.index.metric_type}")
        print(f"Input Files: {len(config.data_sources.input_files)} files")
        print(f"Max input-file errors: {config.error_handling.max_errors}")
        print(f"Search text segments: {len(config.search_text.segments)}")

    except Exception as e:
        print(f"Failed to load configuration: {e}")
        exit(1)
