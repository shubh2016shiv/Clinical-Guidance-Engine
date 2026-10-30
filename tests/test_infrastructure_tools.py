"""Regression checks for infrastructure startup and Redis deletion safeguards."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infrastructure"))

import flush_redis  # noqa: E402
import manage_infrastructure  # noqa: E402


class InfrastructureManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = manage_infrastructure.InfrastructureManager()

    def test_foreign_container_blocks_start_without_removal(self) -> None:
        with (
            patch.object(self.manager, "_check_docker_available", return_value=(True, "")),
            patch.object(
                self.manager,
                "_get_container_info",
                side_effect=lambda name: {"state": "running"} if name == "redis" else None,
            ),
            patch.object(self.manager, "_check_container_belongs_to_project", return_value=False),
            patch.object(self.manager, "_force_remove_container") as remove,
            patch.object(self.manager, "_run_compose_up_with_retry") as compose_up,
        ):
            success, message = self.manager.start_services()

        self.assertFalse(success)
        self.assertIn("redis", message)
        remove.assert_not_called()
        compose_up.assert_not_called()

    def test_matching_project_label_in_another_directory_is_not_ownership(self) -> None:
        labels = {
            "com.docker.compose.project": self.manager._get_compose_project_name(),
            "com.docker.compose.project.working_dir": str(self.manager.compose_dir / "other"),
        }
        with patch.object(
            manage_infrastructure.subprocess,
            "run",
            return_value=Mock(returncode=0, stdout=json.dumps(labels)),
        ):
            self.assertFalse(self.manager._check_container_belongs_to_project("redis"))

    def test_running_but_unhealthy_container_is_not_ready(self) -> None:
        def status(name: str) -> dict[str, str]:
            return {
                "state": "running",
                "status": "Up 20 seconds (unhealthy)"
                if name == "redis"
                else "Up 20 seconds (healthy)",
            }

        with patch.object(self.manager, "_get_container_info", side_effect=status):
            success, message = self.manager._wait_for_services_healthy(timeout=0)

        self.assertFalse(success)
        self.assertIn("redis", message)

    def test_ensure_does_not_start_services_that_are_already_healthy(self) -> None:
        with (
            patch.object(self.manager, "_check_docker_available", return_value=(True, "")),
            patch.object(
                self.manager,
                "_get_container_info",
                return_value={"state": "running", "status": "Up 1 minute (healthy)"},
            ),
            patch.object(self.manager, "start_services") as start_services,
        ):
            success, message = self.manager.ensure_services_running()

        self.assertTrue(success)
        self.assertIn("already running", message)
        start_services.assert_not_called()

    def test_ensure_starts_services_when_one_is_not_running(self) -> None:
        def container_info(name: str) -> dict[str, str] | None:
            if name == "redis":
                return None
            return {"state": "running", "status": "Up 1 minute (healthy)"}

        with (
            patch.object(self.manager, "_check_docker_available", return_value=(True, "")),
            patch.object(self.manager, "_get_container_info", side_effect=container_info),
            patch.object(
                self.manager, "start_services", return_value=(True, "started")
            ) as start_services,
        ):
            success, message = self.manager.ensure_services_running()

        self.assertTrue(success)
        self.assertEqual(message, "started")
        start_services.assert_called_once()

    def test_ensure_fails_without_trying_to_start_when_docker_is_unavailable(self) -> None:
        with (
            patch.object(
                self.manager, "_check_docker_available", return_value=(False, "no docker")
            ),
            patch.object(self.manager, "start_services") as start_services,
        ):
            success, message = self.manager.ensure_services_running()

        self.assertEqual((success, message), (False, "no docker"))
        start_services.assert_not_called()


class RedisFlushTests(unittest.TestCase):
    def test_explicit_database_zero_overrides_environment(self) -> None:
        args = flush_redis.create_argument_parser().parse_args(["--db", "0"])
        with patch.dict(os.environ, {"REDIS_DB": "1"}):
            config = flush_redis.ConfigurationLoader(Mock()).get_redis_config_from_environment(args)
        self.assertEqual(config["db"], 0)

    def test_glob_characters_are_rejected_in_literal_prefix(self) -> None:
        args = flush_redis.create_argument_parser().parse_args(["--prefix", "session*"])
        valid, message = flush_redis.validate_arguments(args)
        self.assertFalse(valid)
        self.assertIn("glob", message)

    def test_failed_key_scan_does_not_look_like_zero_matches(self) -> None:
        client = Mock()
        client.scan_iter.side_effect = RuntimeError("scan failed")
        inspector = flush_redis.RedisKeyInspector(client, Mock())
        with self.assertRaisesRegex(RuntimeError, "scan failed"):
            inspector.count_keys_matching_pattern("session*")

    def test_failed_database_size_does_not_look_empty(self) -> None:
        client = Mock()
        client.dbsize.side_effect = RuntimeError("database unavailable")
        connection = flush_redis.RedisConnectionManager("localhost", 6381, 0, None, 5, 5, Mock())
        connection.client = client
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            connection.get_database_size()


if __name__ == "__main__":
    unittest.main()
