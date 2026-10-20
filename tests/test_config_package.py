"""Guard the flat environment interface after splitting settings by service."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from src.config import ENV_FILE, PROJECT_ROOT, Settings


class ConfigPackageTests(unittest.TestCase):
    def test_service_fields_keep_flat_environment_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL_NAME": "test-model",
                "MILVUS_PORT": "19531",
                "MINIO_BUCKET_NAME": "test-bucket",
                "REDIS_PORT": "6382",
                "MONGO_DB_NAME": "test-database",
            },
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.openai_model_name, "test-model")
        self.assertEqual(settings.milvus_port, 19531)
        self.assertEqual(settings.minio_bucket_name, "test-bucket")
        self.assertEqual(settings.redis_port, 6382)
        self.assertEqual(settings.mongo_db_name, "test-database")

    def test_validation_and_paths_are_preserved(self) -> None:
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parent.parent)
        self.assertEqual(ENV_FILE, PROJECT_ROOT / ".env")
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, milvus_port=0)

    def test_logging_masks_secrets(self) -> None:
        settings = Settings(
            _env_file=None,
            openai_api_key="sk-example-secret",
            minio_root_password="example-minio-secret",
        )
        logger = MagicMock()

        settings.log_loaded_settings(logger=logger)

        logged_values = {
            call.kwargs["field"]: call.kwargs["value"] for call in logger.debug.call_args_list
        }
        self.assertNotIn("sk-example-secret", logged_values["openai_api_key"])
        self.assertNotIn("example-minio-secret", logged_values["minio_root_password"])


if __name__ == "__main__":
    unittest.main()
