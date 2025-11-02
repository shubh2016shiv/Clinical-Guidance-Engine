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

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Union
from pydantic import (
    BaseModel,
    Field,
    root_validator,
    ValidationError,
    ConfigDict,
    field_validator,
)
import logging

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = logging.getLogger(__name__)


class GeminiEmbeddingConfig(BaseModel):
    """Gemini embedding configuration"""

    dimensions: int = Field(default=768, ge=256, le=3072)
    normalize: bool = Field(default=True)

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v):
        """Validate embedding dimensions are supported"""
        valid_dims = [256, 512, 768, 1024, 1536, 2048, 3072]
        if v not in valid_dims:
            logger.warning(
                f"Embedding dimension {v} may not be supported. Valid: {valid_dims}"
            )
        return v


class GeminiConfig(BaseModel):
    """Gemini API configuration"""

    api_key: str = Field(..., env="GEMINI_API_KEY")
    model_name: str = Field(default="models/text-embedding-004")
    task_type: str = Field(default="retrieval_document")
    embedding: GeminiEmbeddingConfig = Field(default_factory=GeminiEmbeddingConfig)

    # Rate limiting and retry configuration
    requests_per_second: float = Field(default=2.0, gt=0)
    batch_size: int = Field(default=100, ge=1, le=1000)
    retry_attempts: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, gt=0)
    exponential_backoff: bool = Field(default=True)
    timeout_seconds: int = Field(default=30, gt=0)
    skip_on_error: bool = Field(default=False)
    log_failures: bool = Field(default=True)


class MilvusConnectionConfig(BaseModel):
    """Milvus connection configuration"""

    host: str = Field(default="localhost")
    port: int = Field(default=19530, ge=1, le=65535)
    alias: str = Field(default="default")
    timeout: int = Field(default=30, gt=0)
    user: str = Field(default="")
    password: str = Field(default="")
    secure: bool = Field(default=False)
    server_pem_path: str = Field(default="")
    server_name: str = Field(default="")


class MilvusIndexConfig(BaseModel):
    """Milvus index configuration"""

    type: str = Field(default="HNSW", alias="index_type")
    metric_type: str = Field(default="IP")
    params: Dict[str, Any] = Field(default_factory=dict)

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
            raise ValueError(
                f"Invalid metric type: {v}. Milvus supports: {valid_metrics}"
            )
        return v

    @root_validator(skip_on_failure=True)
    @classmethod
    def set_default_params(cls, values):
        """Set default params based on index type if not provided"""
        index_type = values.get("type")
        params = values.get("params", {})

        if not params:
            if index_type == "HNSW":
                params = {"M": 16, "efConstruction": 200}
            elif index_type in ["IVF_FLAT", "IVF_SQ8"]:
                params = {"nlist": 128}
            elif index_type == "IVF_PQ":
                params = {"nlist": 128, "m": 8, "nbits": 8}
            values["params"] = params

        return values


class MilvusSearchConfig(BaseModel):
    """Milvus search configuration"""

    params: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1)
    offset: int = Field(default=0, ge=0)
    output_fields: List[str] = Field(default_factory=list)
    consistency_level: str = Field(default="Strong")

    @field_validator("consistency_level")
    @classmethod
    def validate_consistency_level(cls, v):
        """Validate consistency level is supported"""
        valid_levels = ["Strong", "Bounded", "Eventually", "Session"]
        if v not in valid_levels:
            raise ValueError(f"Invalid consistency level: {v}. Valid: {valid_levels}")
        return v

    @root_validator(skip_on_failure=True)
    @classmethod
    def set_default_search_params(cls, values):
        """Set default search params if not provided"""
        params = values.get("params", {})
        if not params:
            values["params"] = {"ef": 64, "nprobe": 10}
        return values


class MilvusCollectionConfig(BaseModel):
    """Milvus collection configuration"""

    name: str = Field(default="pharmaceutical_drugs")
    description: str = Field(
        default="Pharmaceutical drug database with embeddings for LLM lookup"
    )
    drop_existing: bool = Field(default=False)
    embedding_dimension: int = Field(default=768, ge=1)


class MilvusInsertionConfig(BaseModel):
    """Milvus insertion configuration"""

    batch_size: int = Field(default=10, ge=1)
    timeout: int = Field(default=300, gt=0)
    validate_before_insert: bool = Field(default=True)
    skip_invalid_vectors: bool = Field(default=False)


class MilvusPerformanceConfig(BaseModel):
    """Milvus performance tuning configuration"""

    load_on_startup: bool = Field(default=True)
    replicas: int = Field(default=1, ge=1)
    max_memory_usage: str = Field(default="2GB")
    cpu_threads: int = Field(default=4, ge=1)
    cache_enabled: bool = Field(default=True)
    cache_size: str = Field(default="500MB")


class MilvusConfig(BaseModel):
    """Complete Milvus configuration"""

    connection: MilvusConnectionConfig = Field(default_factory=MilvusConnectionConfig)
    collection: MilvusCollectionConfig = Field(default_factory=MilvusCollectionConfig)
    index: MilvusIndexConfig = Field(default_factory=MilvusIndexConfig)
    search: MilvusSearchConfig = Field(default_factory=MilvusSearchConfig)
    insertion: MilvusInsertionConfig = Field(default_factory=MilvusInsertionConfig)
    performance: MilvusPerformanceConfig = Field(
        default_factory=MilvusPerformanceConfig
    )


class ETLParserConfig(BaseModel):
    """ETL parser configuration"""

    extract_dosages: bool = Field(default=True)
    extract_routes: bool = Field(default=True)
    extract_formulations: bool = Field(default=True)
    normalize_drug_names: bool = Field(default=True)
    recognized_routes: List[str] = Field(
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

    recognized_formulations: List[str] = Field(
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

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/etl_pipeline.log")
    batch_size: int = Field(default=1000, ge=1)
    parallel_processing: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1)
    validate_fields: bool = Field(default=True)
    remove_invalid_records: bool = Field(default=True)
    parser: ETLParserConfig = Field(default_factory=ETLParserConfig)


class DataSourceConfig(BaseModel):
    """Data source configuration"""

    input_files: List[str] = Field(default_factory=list)
    output_directory: str = Field(default="output")
    standardized_data_file: str = Field(default="standardized_drugs.csv")
    encoding: str = Field(default="utf-8")
    fallback_encoding: str = Field(default="latin-1")
    skip_empty_rows: bool = Field(default=True)
    min_drug_name_length: int = Field(default=2, ge=1)
    deduplicate: bool = Field(default=True)


class SearchTextConfig(BaseModel):
    """Search text generation configuration"""

    template: str = Field(
        default="Drug: {drug_name} | Dosages: [{dosages}] | Form: {formulation} | Route: {route} | Class: {drug_class} | Subclass: {drug_sub_class} | Therapeutic: {therapeutic_category}"
    )
    include_fields: Dict[str, bool] = Field(
        default_factory=lambda: {
            "drug_name": True,
            "dosages": True,
            "formulation": True,
            "route_of_administration": True,
            "drug_class": True,
            "drug_sub_class": True,
            "therapeutic_category": True,
        }
    )
    separator: str = Field(default=" | ")
    dosage_delimiter: str = Field(default=", ")
    empty_field_placeholder: str = Field(default="")
    include_empty_fields: bool = Field(default=False)
    progress_enabled: bool = Field(default=True, alias="progress.enabled")
    progress_interval: int = Field(default=100, ge=1, alias="progress.interval")
    progress_detailed: bool = Field(default=False, alias="progress.detailed")


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""

    continue_on_error: bool = Field(default=True)
    max_errors: int = Field(default=2, ge=0)
    strict_validation: bool = Field(default=True)
    retry_on_failure: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=2.0, gt=0)


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
    gemini: GeminiConfig
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    search_text: SearchTextConfig = Field(default_factory=SearchTextConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)

    @field_validator("gemini", mode="before")
    @classmethod
    def validate_gemini_config(cls, v):
        """Ensure gemini config has required api_key"""
        if isinstance(v, dict) and "api_key" not in v:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable or provide in config."
            )
        return v


class ConfigLoader:
    """
    Load and validate configuration from YAML file using Pydantic.

    Automatically loads .env file from the same directory as the config file
    to populate environment variables (e.g., GEMINI_API_KEY).

    The .env file should be located in the same directory as ingestion_config.yaml.
    If not found there, it will attempt to load from the current working directory.
    """

    def __init__(self, config_path: Union[str, Path] = "ingestion_config.yaml"):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
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
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

        # Resolve environment variables (now .env file is already loaded)
        self._resolve_environment_variables()

        # Auto-sync embedding dimensions before building config if Milvus dimension not explicitly set
        self._sync_embedding_dimensions_in_yaml()

        # Build and validate configuration using Pydantic
        try:
            config = IngestionConfig(**self.config_data)
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

        # Additional custom validations
        self._validate_file_paths(config)
        self._validate_batch_sizes(config)
        self._validate_dimensions(config)

        self.logger.info("Configuration loaded and validated successfully")
        return config

    def _resolve_environment_variables(self):
        """Resolve ${VAR_NAME} environment variables in config"""

        def resolve_value(value: Any) -> Any:
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
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
        """Validate that input files exist"""
        if not config.data_sources.input_files:
            raise ValueError("No input files specified in configuration")

        missing_files = []
        for file_path in config.data_sources.input_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            self.logger.warning(f"Input files not found: {missing_files}")
            if not config.error_handling.continue_on_error:
                raise FileNotFoundError(f"Missing input files: {missing_files}")

    def _validate_batch_sizes(self, config: IngestionConfig):
        """Validate batch sizes are reasonable"""
        if config.gemini.batch_size < 1 or config.gemini.batch_size > 1000:
            self.logger.warning(
                f"Gemini batch size {config.gemini.batch_size} may be suboptimal. "
                "Recommended range: 50-200"
            )

        if config.milvus.insertion.batch_size < 1:
            raise ValueError("Milvus insertion batch size must be positive")

    def _sync_embedding_dimensions_in_yaml(self):
        """Auto-sync milvus.collection.embedding_dimension from gemini.embedding.dimensions if not explicitly set"""
        gemini_dims = (
            self.config_data.get("gemini", {}).get("embedding", {}).get("dimensions")
        )
        milvus_collection = self.config_data.get("milvus", {}).get("collection", {})

        if gemini_dims:
            # If embedding_dimension not explicitly set in YAML (defaults to 768), sync from Gemini
            if "embedding_dimension" not in milvus_collection:
                milvus_collection["embedding_dimension"] = gemini_dims
                self.logger.info(
                    f"Auto-synced milvus.collection.embedding_dimension to {gemini_dims} "
                    f"from gemini.embedding.dimensions"
                )
            elif milvus_collection.get("embedding_dimension") == 768:
                # If using default value (768) but Gemini has different value, sync it
                milvus_collection["embedding_dimension"] = gemini_dims
                self.logger.info(
                    f"Auto-synced milvus.collection.embedding_dimension from default (768) to {gemini_dims} "
                    f"to match gemini.embedding.dimensions"
                )

    def _validate_dimensions(self, config: IngestionConfig):
        """Validate that embedding dimensions match between Gemini and Milvus"""
        gemini_dims = config.gemini.embedding.dimensions
        milvus_dims = config.milvus.collection.embedding_dimension

        if gemini_dims != milvus_dims:
            error_msg = (
                f"Embedding dimension mismatch: gemini.embedding.dimensions={gemini_dims} "
                f"but milvus.collection.embedding_dimension={milvus_dims}. "
                f"They must match for embeddings to be inserted into Milvus."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(
            f"[DEBUG] Embedding dimensions validated: {gemini_dims} (Gemini and Milvus match)"
        )

    def save_config(self, config: IngestionConfig, output_path: Union[str, Path]):
        """Save configuration to YAML file"""

        output_path = Path(output_path)

        # Convert config to dict, masking sensitive data
        config_dict = config.model_dump()

        # Mask API key for security
        if "gemini" in config_dict and "api_key" in config_dict["gemini"]:
            config_dict["gemini"]["api_key"] = "${GEMINI_API_KEY}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise IOError(f"Failed to save configuration to {output_path}: {e}")

        self.logger.info(f"Configuration saved to: {output_path}")

    def get_config_template(self) -> str:
        """Generate a configuration template as YAML string"""
        template_config = IngestionConfig(
            gemini=GeminiConfig(api_key="${GEMINI_API_KEY}")
        )
        return yaml.dump(
            template_config.model_dump(), default_flow_style=False, sort_keys=False
        )


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
        print(f"Gemini Model: {config.gemini.model_name}")
        print(f"Embedding Dimensions: {config.gemini.embedding.dimensions}")
        print(
            f"Milvus Host: {config.milvus.connection.host}:{config.milvus.connection.port}"
        )
        print(f"Collection Name: {config.milvus.collection.name}")
        print(f"Index Type: {config.milvus.index.type}")
        print(f"Metric Type: {config.milvus.index.metric_type}")
        print(f"Input Files: {len(config.data_sources.input_files)} files")
        print(f"Continue on Error: {config.error_handling.continue_on_error}")

    except Exception as e:
        print(f"Failed to load configuration: {e}")
        exit(1)
