"""Check the infrastructure connection report against a sample resolved compose file."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infrastructure"))

import connection_info  # noqa: E402


def sample_compose():
    return {
        "services": {
            "standalone": {
                "container_name": "milvus-standalone",
                "ports": [
                    {"target": 19530, "published": "19530"},
                    {"target": 9091, "published": "9091"},
                ],
            },
            "attu": {
                "container_name": "attu",
                "ports": [{"target": 3000, "published": "8002"}],
            },
            "minio": {
                "container_name": "milvus-minio",
                "environment": {"MINIO_ROOT_USER": "root", "MINIO_ROOT_PASSWORD": "secret"},
                "ports": [
                    {"target": 9000, "published": "9000"},
                    {"target": 9001, "published": "9001"},
                ],
            },
            "etcd": {"container_name": "milvus-etcd"},
            "redis": {
                "container_name": "redis",
                "command": ["redis-server", "--requirepass", "redis123", "--appendonly", "yes"],
                "ports": ["127.0.0.1:6381:6379/tcp"],
            },
        }
    }


def rows_of(groups, title_prefix):
    for group in groups:
        for service in group.services:
            if service.title.startswith(title_prefix):
                return dict(service.rows), service.status
    raise AssertionError(f"no service titled {title_prefix}")


class ConnectionInfoTests(unittest.TestCase):
    def test_redis_uses_published_host_port_and_password_from_the_command(self) -> None:
        groups = connection_info.build_groups(sample_compose(), {"redis": "running (healthy)"})

        rows, status = rows_of(groups, "REDIS")

        self.assertEqual(rows["Host port"], "localhost:6381  (container 6379)")
        self.assertEqual(rows["Password"], "redis123")
        self.assertEqual(rows["Connection string"], "redis://:redis123@localhost:6381/0")
        self.assertIn("REDIS_PORT=6381", rows["App .env"])
        self.assertEqual(status, "running (healthy)")

    def test_attu_tells_the_user_to_connect_by_container_name(self) -> None:
        rows, _ = rows_of(connection_info.build_groups(sample_compose(), {}), "ATTU")

        self.assertEqual(rows["Open in browser"], "http://localhost:8002  (container 3000)")
        self.assertEqual(rows["Connect to Milvus at"], "milvus-standalone:19530")

    def test_etcd_is_reported_as_not_published(self) -> None:
        rows, _ = rows_of(connection_info.build_groups(sample_compose(), {}), "ETCD")

        self.assertEqual(rows["Host access"], "not published (used only by Milvus)")

    def test_unknown_status_is_shown_when_a_container_was_not_inspected(self) -> None:
        _, status = rows_of(connection_info.build_groups(sample_compose(), {}), "MILVUS")

        self.assertEqual(status, "unknown")

    def test_services_missing_from_the_compose_file_are_skipped(self) -> None:
        compose = {"services": {"redis": sample_compose()["services"]["redis"]}}

        groups = connection_info.build_groups(compose, {})
        report = connection_info.render(compose, {}, "demo")

        self.assertEqual([group.title for group in groups], ["CACHE"])
        self.assertNotIn("WEB UIs", report)
        self.assertNotIn("MILVUS", report)

    def test_report_groups_services_and_lists_web_uis(self) -> None:
        report = connection_info.render(sample_compose(), {}, "demo")

        for heading in (
            "VECTOR DATABASE",
            "CACHE",
            "WEB UIs",
            "project: demo",
        ):
            self.assertIn(heading, report)
        self.assertIn("http://localhost:9001", report)
        self.assertIn("root / secret", report)
        self.assertNotIn("MONGO", report.upper())

    def test_ascii_table_is_used_when_the_console_cannot_print_box_characters(self) -> None:
        with patch.object(connection_info.sys, "stdout", SimpleNamespace(encoding="cp1252")):
            self.assertEqual(connection_info._table_format(), "outline")
        with patch.object(connection_info.sys, "stdout", SimpleNamespace(encoding="utf-8")):
            self.assertEqual(connection_info._table_format(), "rounded_outline")


if __name__ == "__main__":
    unittest.main()
