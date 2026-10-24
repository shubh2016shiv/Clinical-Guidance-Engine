"""Connection details for the local infrastructure, read from the resolved compose file.

Configured ports and credentials come from `docker compose config`. Container
status is queried separately; changed settings require container recreation.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

from tabulate import tabulate

HOST = "localhost"


@dataclass(frozen=True)
class ServiceInfo:
    title: str
    status: str
    rows: list[tuple[str, str]]


@dataclass(frozen=True)
class Group:
    title: str
    services: list[ServiceInfo]


def _ports(service: dict[str, Any]) -> dict[int, int]:
    """Map container port -> host port for every published port."""
    mapping: dict[int, int] = {}
    for entry in service.get("ports", []):
        if isinstance(entry, dict):
            target, published = entry.get("target"), entry.get("published")
        else:
            parts = str(entry).split("/")[0].split(":")
            target, published = parts[-1], (parts[-2] if len(parts) > 1 else None)
        if target and published:
            mapping[int(target)] = int(published)
    return mapping


def _environment(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment", {})
    if isinstance(environment, list):
        return dict(item.split("=", 1) for item in environment if "=" in item)
    return {name: str(value) for name, value in environment.items()}


def _redis_password(service: dict[str, Any]) -> str:
    command = service.get("command", "")
    text = " ".join(command) if isinstance(command, list) else str(command)
    match = re.search(r"--requirepass\s+(\S+)", text)
    return match.group(1) if match else ""


def _at(host_port: int, container_port: int) -> str:
    return f"{HOST}:{host_port}  (container {container_port})"


def _milvus(service: dict[str, Any], status: str) -> ServiceInfo:
    ports = _ports(service)
    return ServiceInfo(
        "MILVUS  (vector database)",
        status,
        [
            ("gRPC API", _at(ports[19530], 19530)),
            ("Health check", f"http://{HOST}:{ports[9091]}/healthz  (container 9091)"),
            ("Login", "none (authentication is disabled)"),
            ("Inside Docker", f"{service['container_name']}:19530"),
            ("App .env", f"MILVUS_HOST={HOST}  MILVUS_PORT={ports[19530]}"),
        ],
    )


def _attu(service: dict[str, Any], status: str, milvus_container: str) -> ServiceInfo:
    ports = _ports(service)
    return ServiceInfo(
        "ATTU  (Milvus web UI)",
        status,
        [
            ("Open in browser", f"http://{HOST}:{ports[3000]}  (container 3000)"),
            ("Login", "none"),
            ("Connect to Milvus at", f"{milvus_container}:19530"),
            ("Note", "Attu runs inside Docker: use the container name above, not localhost"),
        ],
    )


def _minio(service: dict[str, Any], status: str) -> ServiceInfo:
    ports, environment = _ports(service), _environment(service)
    user, password = environment["MINIO_ROOT_USER"], environment["MINIO_ROOT_PASSWORD"]
    return ServiceInfo(
        "MINIO  (object storage for Milvus)",
        status,
        [
            ("Console (web UI)", f"http://{HOST}:{ports[9001]}  (container 9001)"),
            ("S3 API", f"http://{HOST}:{ports[9000]}  (container 9000)"),
            ("Username", user),
            ("Password", password),
            ("Inside Docker", f"{service['container_name']}:9000"),
            ("App .env", f"MINIO_ROOT_USER={user}  MINIO_ROOT_PASSWORD={password}"),
        ],
    )


def _etcd(service: dict[str, Any], status: str) -> ServiceInfo:
    return ServiceInfo(
        "ETCD  (Milvus metadata store)",
        status,
        [
            ("Host access", "not published (used only by Milvus)"),
            ("Login", "none"),
            ("Inside Docker", f"{service['container_name']}:2379"),
        ],
    )


def _mongodb(service: dict[str, Any], status: str) -> ServiceInfo:
    ports, environment = _ports(service), _environment(service)
    user = environment["MONGO_INITDB_ROOT_USERNAME"]
    password = environment["MONGO_INITDB_ROOT_PASSWORD"]
    uri = f"mongodb://{quote_plus(user)}:{quote_plus(password)}@{HOST}:{ports[27017]}/?authSource=admin"
    return ServiceInfo(
        "MONGODB  (document database)",
        status,
        [
            ("Host port", _at(ports[27017], 27017)),
            ("Username", user),
            ("Password", password),
            ("Auth database", "admin"),
            ("Connection string", uri),
            ("Shell", f'mongosh "{uri}"'),
            ("Inside Docker", f"{service['container_name']}:27017"),
            (
                "App .env",
                f"MONGO_PORT={ports[27017]}  MONGO_ROOT_USER={user}  MONGO_ROOT_PASSWORD={password}",
            ),
        ],
    )


def _redis(service: dict[str, Any], status: str) -> ServiceInfo:
    ports, password = _ports(service), _redis_password(service)
    return ServiceInfo(
        "REDIS  (cache)",
        status,
        [
            ("Host port", _at(ports[6379], 6379)),
            ("Username", "none (default user)"),
            ("Password", password),
            ("Connection string", f"redis://:{quote_plus(password)}@{HOST}:{ports[6379]}/0"),
            ("Shell", f"redis-cli -h {HOST} -p {ports[6379]} -a {password}"),
            ("Inside Docker", f"{service['container_name']}:6379"),
            ("App .env", f"REDIS_HOST={HOST}  REDIS_PORT={ports[6379]}  REDIS_PASSWORD={password}"),
        ],
    )


def build_groups(compose: dict[str, Any], statuses: dict[str, str]) -> list[Group]:
    """Group the known services by infrastructure, skipping any not in the compose file."""
    services = compose.get("services", {})

    def status_of(name: str) -> str:
        return statuses.get(services[name]["container_name"], "unknown")

    milvus_stack = []
    if "standalone" in services:
        milvus_stack.append(_milvus(services["standalone"], status_of("standalone")))
    if "attu" in services and "standalone" in services:
        container = services["standalone"]["container_name"]
        milvus_stack.append(_attu(services["attu"], status_of("attu"), container))
    for name, build in (("minio", _minio), ("etcd", _etcd)):
        if name in services:
            milvus_stack.append(build(services[name], status_of(name)))

    groups = [Group("VECTOR DATABASE  -  Milvus stack", milvus_stack)]
    if "mongodb" in services:
        groups.append(
            Group("DOCUMENT DATABASE", [_mongodb(services["mongodb"], status_of("mongodb"))])
        )
    if "redis" in services:
        groups.append(Group("CACHE", [_redis(services["redis"], status_of("redis"))]))
    return [group for group in groups if group.services]


def _web_uis(compose: dict[str, Any]) -> list[list[str]]:
    services, rows = compose.get("services", {}), []
    if "attu" in services:
        rows.append(
            ["Attu (Milvus)", f"http://{HOST}:{_ports(services['attu'])[3000]}", "no login"]
        )
    if "minio" in services:
        environment = _environment(services["minio"])
        login = f"{environment['MINIO_ROOT_USER']} / {environment['MINIO_ROOT_PASSWORD']}"
        rows.append(["MinIO Console", f"http://{HOST}:{_ports(services['minio'])[9001]}", login])
    return rows


def _table_format() -> str:
    """Use box-drawing characters only when the terminal can print them."""
    return "rounded_outline" if "utf" in (sys.stdout.encoding or "").lower() else "outline"


def render(compose: dict[str, Any], statuses: dict[str, str], project: str) -> str:
    """Return the complete connection-details report as text."""
    table_format, rule = _table_format(), "=" * 88
    lines = [f"\n{rule}", f"  INFRASTRUCTURE CONNECTION DETAILS   (project: {project})", rule]
    for group in build_groups(compose, statuses):
        lines.append(f"\n  {group.title}\n")
        for service in group.services:
            table = tabulate(
                service.rows, headers=[service.title, service.status], tablefmt=table_format
            )
            lines.append(table + "\n")
    uis = _web_uis(compose)
    if uis:
        lines.append("  WEB UIs\n")
        lines.append(tabulate(uis, headers=["UI", "URL", "Login"], tablefmt=table_format) + "\n")
    lines.append(
        "  These are local-development defaults. To change them, set REDIS_PASSWORD,\n"
        "  MINIO_ROOT_USER or MINIO_ROOT_PASSWORD in\n"
        "  infrastructure/.env, restart the services, and update the matching app .env values."
    )
    lines.append(rule + "\n")
    return "\n".join(lines)
