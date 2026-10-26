"""
Asclepius Healthcare Chatbot - Infrastructure Management Script

This script manages Docker Compose infrastructure services for the Asclepius Healthcare Chatbot,
including Milvus (vector database) and Redis (cache).

Features:
- Starting services with health checks
- Stopping services gracefully
- Restarting services
- Displaying connection information
- Detection of container-name conflicts without removing other projects' containers
- Retry logic for transient start failures
- Volume preservation

Usage:
    python manage_infrastructure.py --start
    python manage_infrastructure.py --stop
    python manage_infrastructure.py --restart
    python manage_infrastructure.py --info
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from connection_info import render as render_connection_info


class InfrastructureManager:
    """Manages Docker Compose infrastructure services for Asclepius Healthcare Chatbot."""

    # Long-running containers expected by the all profile.
    EXPECTED_CONTAINERS = [
        "milvus-etcd",
        "milvus-minio",
        "attu",
        "milvus-standalone",
        "redis",
    ]
    HEALTHCHECK_CONTAINERS = {"milvus-etcd", "milvus-minio", "milvus-standalone", "redis"}

    # Every service in docker-compose.yml belongs to a profile, so Compose selects
    # nothing unless one is passed. "all" covers the core services plus Attu.
    COMPOSE_PROFILE = "all"

    # Seconds allowed for pulling all images (Milvus alone is over 1 GB)
    PULL_TIMEOUT = 1800

    # Maximum retry attempts for operations
    MAX_RETRIES = 5
    RETRY_DELAY = 2  # seconds

    def __init__(self, compose_file_path: Path | None = None):
        """
        Initialize the infrastructure manager.

        Args:
            compose_file_path: Path to docker-compose.yml file.
                              If None, uses script directory/docker-compose.yml
        """
        if compose_file_path is None:
            script_dir = Path(__file__).parent
            compose_file_path = script_dir / "docker-compose.yml"

        self.compose_file_path = compose_file_path.resolve()
        self.compose_dir = self.compose_file_path.parent

        if not self.compose_file_path.exists():
            raise FileNotFoundError(f"Docker Compose file not found: {self.compose_file_path}")

    def _check_docker_available(self) -> tuple[bool, str]:
        """
        Check if Docker is installed and the daemon is running.

        Returns:
            Tuple of (is_available, error_message)
            If available, error_message is empty string.
        """
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False, "Docker is not installed or not in PATH"
        except FileNotFoundError:
            return False, "Docker is not installed. Install Docker Engine or Docker Desktop."
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out. Docker may not be responding."

        try:
            result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Docker daemon is not running"
                if "Cannot connect" in error_msg or "Is the docker daemon running" in error_msg:
                    return (
                        False,
                        "Docker daemon is not running. Start Docker Engine or Docker Desktop.",
                    )
                return False, f"Docker daemon error: {error_msg}"
        except subprocess.TimeoutExpired:
            return (
                False,
                "Docker daemon is not responding. Check the Docker service.",
            )

        return True, ""

    def _get_compose_command(self) -> list[str]:
        """
        Get the appropriate docker compose command.

        Returns:
            List of command parts (e.g., ['docker', 'compose'] or ['docker-compose'])
        """
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["docker", "compose"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ["docker-compose"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        raise RuntimeError(
            "Neither 'docker compose' nor 'docker-compose' is available. "
            "Please install Docker Compose."
        )

    def _get_compose_project_name(self) -> str:
        """
        Get the Docker Compose project name.

        First tries to read the 'name' field from docker-compose.yml.
        Falls back to directory name if not found.

        Returns:
            Project name string
        """
        try:
            import yaml

            with open(self.compose_file_path, encoding="utf-8") as f:
                compose_data = yaml.safe_load(f)
                if compose_data and "name" in compose_data:
                    return compose_data["name"].lower()
        except (ImportError, FileNotFoundError, KeyError, Exception):
            pass

        project_name = self.compose_dir.name.lower().replace(" ", "-").replace("_", "-")
        return project_name

    def _get_container_info(self, container_name: str) -> dict[str, str] | None:
        """
        Get detailed information about a container.

        Args:
            container_name: Name of the container to check

        Returns:
            Dictionary with container info (id, state, status) or None if not found
        """
        try:
            cmd = [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{json .}}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                container_info = json.loads(result.stdout.strip().split("\n")[0])
                return {
                    "id": container_info.get("ID", ""),
                    "state": container_info.get("State", ""),
                    "status": container_info.get("Status", ""),
                    "names": container_info.get("Names", ""),
                }
            return None
        except Exception as e:
            print(f"  Warning: Could not get info for {container_name}: {str(e)}")
            return None

    def _check_container_belongs_to_project(self, container_name: str) -> bool:
        """
        Check if a container belongs to this Docker Compose project.

        Args:
            container_name: Name of the container to check

        Returns:
            True if container belongs to this project, False otherwise
        """
        try:
            cmd = [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{json .Config.Labels}}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return False

            labels = json.loads(result.stdout.strip())

            # Both labels must match; the custom project label is shared by clones.
            compose_project = labels.get("com.docker.compose.project", "")
            compose_working_dir = labels.get("com.docker.compose.project.working_dir", "")

            project_name = self._get_compose_project_name()
            current_working_dir = str(self.compose_dir.resolve())

            return bool(
                compose_working_dir
                and compose_project.lower() == project_name.lower()
                and Path(compose_working_dir).resolve() == Path(current_working_dir).resolve()
            )
        except Exception as e:
            print(f"  Warning: Could not check project ownership for {container_name}: {str(e)}")
            return False

    def _force_remove_container(self, container_name: str) -> tuple[bool, str]:
        """
        Force remove a container with retries, ensuring it's completely gone.

        Args:
            container_name: Name of the container to remove

        Returns:
            Tuple of (success, message)
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                # First, try to stop the container if running
                info = self._get_container_info(container_name)
                if info and info["state"] in ["running", "paused"]:
                    stop_cmd = ["docker", "stop", "-t", "10", container_name]
                    subprocess.run(stop_cmd, capture_output=True, text=True, timeout=30)
                    time.sleep(1)  # Brief wait for graceful shutdown

                # Force remove the container (preserves volumes by default)
                remove_cmd = ["docker", "rm", "-f", container_name]
                result = subprocess.run(remove_cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    # Verify removal with a short wait
                    time.sleep(0.5)
                    verify_info = self._get_container_info(container_name)
                    if verify_info is None:
                        return True, f"Successfully removed container {container_name}"
                    else:
                        print(f"  Attempt {attempt + 1}: Container still exists, retrying...")
                        time.sleep(self.RETRY_DELAY)
                        continue
                else:
                    error_msg = result.stderr.strip()
                    if "No such container" in error_msg:
                        return True, f"Container {container_name} already removed"

                    if attempt < self.MAX_RETRIES - 1:
                        print(f"  Attempt {attempt + 1} failed: {error_msg}, retrying...")
                        time.sleep(self.RETRY_DELAY)
                    else:
                        return False, f"Failed to remove {container_name}: {error_msg}"

            except subprocess.TimeoutExpired:
                if attempt < self.MAX_RETRIES - 1:
                    print(f"  Attempt {attempt + 1} timed out, retrying...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    return (
                        False,
                        f"Timeout removing {container_name} after {self.MAX_RETRIES} attempts",
                    )
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    print(f"  Attempt {attempt + 1} error: {str(e)}, retrying...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    return False, f"Error removing {container_name}: {str(e)}"

        return (
            False,
            f"Failed to remove {container_name} after {self.MAX_RETRIES} attempts",
        )

    def _check_container_conflicts(self) -> tuple[bool, str]:
        """Stop startup if a required name is owned by another project."""
        conflicts = []
        for container_name in self.EXPECTED_CONTAINERS:
            info = self._get_container_info(container_name)
            if info and not self._check_container_belongs_to_project(container_name):
                conflicts.append(container_name)
        if conflicts:
            names = ", ".join(conflicts)
            return False, f"Container name conflict: {names}. Inspect these containers manually."
        return True, "No container name conflicts"

    def _wait_for_services_healthy(self, timeout: float = 120) -> tuple[bool, str]:
        """Require configured health checks to pass before reporting startup success."""
        deadline = time.monotonic() + timeout
        while True:
            pending = []
            for name in self.EXPECTED_CONTAINERS:
                info = self._get_container_info(name)
                if not info or info["state"] != "running":
                    pending.append(f"{name}: not running")
                elif name in self.HEALTHCHECK_CONTAINERS and "(healthy)" not in info["status"]:
                    pending.append(f"{name}: {info['status']}")
            if not pending:
                return True, f"All {len(self.EXPECTED_CONTAINERS)} services are ready"
            if time.monotonic() >= deadline:
                return False, "Services did not become ready: " + "; ".join(pending)
            time.sleep(min(3, max(0, deadline - time.monotonic())))

    def _pull_images(self, compose_cmd: list[str], project_name: str) -> tuple[bool, str]:
        """
        Pull every image once, showing Docker's own progress.

        Pulling first means a missing or renamed image fails here, immediately and
        visibly, instead of being retried silently by the start loop. A pull
        error such as "pull access denied" or "manifest unknown" does not fix
        itself, so there is no retry.

        Returns:
            Tuple of (success, message)
        """
        cmd = compose_cmd + [
            "-f",
            str(self.compose_file_path.name),
            "-p",
            project_name,
            "--profile",
            self.COMPOSE_PROFILE,
            "pull",
        ]
        print("Pulling images (the first run can take several minutes)...")
        result = subprocess.run(cmd, timeout=self.PULL_TIMEOUT)
        if result.returncode != 0:
            return (
                False,
                "Image pull failed; see the Docker output above. Check that every "
                "image in docker-compose.yml still exists in its registry.",
            )
        return True, "Images pulled"

    def _run_compose_up_with_retry(self) -> tuple[bool, str]:
        """Pull images, start Compose, and wait for service readiness."""
        compose_cmd = self._get_compose_command()
        project_name = self._get_compose_project_name()
        original_cwd = os.getcwd()

        try:
            os.chdir(self.compose_dir)
            pulled, pull_message = self._pull_images(compose_cmd, project_name)
            if not pulled:
                return False, pull_message

            cmd = compose_cmd + [
                "-f",
                self.compose_file_path.name,
                "-p",
                project_name,
                "--profile",
                self.COMPOSE_PROFILE,
                "up",
                "-d",
            ]
            for attempt in range(1, self.MAX_RETRIES + 1):
                print(f"Attempt {attempt}/{self.MAX_RETRIES}: Starting services...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    return self._wait_for_services_healthy()

                error_msg = result.stderr.strip() or result.stdout.strip()
                if "Conflict" in error_msg and "container name" in error_msg:
                    return (
                        False,
                        f"Docker reports a container name conflict. Inspect it manually: {error_msg}",
                    )
                if attempt == self.MAX_RETRIES:
                    return False, f"Failed to start services after {attempt} attempts: {error_msg}"
                print(f"  Error: {error_msg}")
                time.sleep(self.RETRY_DELAY)
        finally:
            os.chdir(original_cwd)

    def start_services(self) -> tuple[bool, str]:
        """Start this project's services without modifying foreign containers."""
        docker_available, error_msg = self._check_docker_available()
        if not docker_available:
            return False, error_msg

        no_conflicts, message = self._check_container_conflicts()
        if not no_conflicts:
            return False, message

        try:
            return self._run_compose_up_with_retry()
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            return False, f"Could not start services: {exc}"

    def stop_services(self) -> tuple[bool, str]:
        """
        Stop all infrastructure services gracefully.

        Returns:
            Tuple of (success, message)
        """
        docker_available, error_msg = self._check_docker_available()
        if not docker_available:
            return False, error_msg

        try:
            compose_cmd = self._get_compose_command()
        except RuntimeError as e:
            return False, str(e)

        original_cwd = os.getcwd()

        try:
            os.chdir(self.compose_dir)

            project_name = self._get_compose_project_name()
            cmd = compose_cmd + [
                "-f",
                str(self.compose_file_path.name),
                "-p",
                project_name,
                "--profile",
                self.COMPOSE_PROFILE,
                "down",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()

                # Try to stop containers individually as fallback
                print(f"⚠ Compose down failed: {error_msg}")
                print("Attempting to stop containers individually...")

                stopped_count = 0
                for container_name in self.EXPECTED_CONTAINERS:
                    info = self._get_container_info(container_name)
                    if info:
                        belongs = self._check_container_belongs_to_project(container_name)
                        if belongs:
                            success, message = self._force_remove_container(container_name)
                            if success:
                                stopped_count += 1
                                print(f"  ✓ Stopped {container_name}")

                if stopped_count > 0:
                    return True, f"Stopped {stopped_count} container(s) manually"

                return False, f"Failed to stop services: {error_msg}"

            return True, "Services stopped successfully"

        except subprocess.TimeoutExpired:
            return False, "Command timed out while stopping services"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
        finally:
            os.chdir(original_cwd)

    def restart_services(self) -> tuple[bool, str]:
        """
        Restart all infrastructure services.

        Returns:
            Tuple of (success, message)
        """
        print("Stopping services...")
        stop_success, stop_msg = self.stop_services()

        if not stop_success:
            return False, f"Failed to stop services: {stop_msg}"

        print(f"✓ {stop_msg}")
        print("\nWaiting for cleanup...")
        time.sleep(3)

        print("\nStarting services...")
        return self.start_services()

    def _load_resolved_compose(self) -> dict:
        """Return the compose file with all variables and defaults resolved, or {}."""
        import yaml

        try:
            cmd = self._get_compose_command() + [
                "-f",
                str(self.compose_file_path.name),
                "-p",
                self._get_compose_project_name(),
                "--profile",
                self.COMPOSE_PROFILE,
                "config",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=self.compose_dir
            )
            if result.returncode == 0:
                return yaml.safe_load(result.stdout) or {}
            print(f"Could not read the compose configuration: {result.stderr.strip()}")
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            print(f"Could not read the compose configuration: {e}")
        return {}

    def _service_status(self, container_name: str) -> str:
        """Describe a container as 'running (healthy)', 'exited', 'not created', ..."""
        info = self._get_container_info(container_name)
        if not info:
            return "not created"
        if info["state"] != "running":
            return info["state"]
        health = re.search(r"\((healthy|unhealthy|health: starting)\)", info["status"])
        return f"running ({health.group(1)})" if health else "running"

    def display_connection_info(self):
        """
        Display connection details for every service, grouped by infrastructure.

        Ports and credentials are read from Compose configuration; container
        status is queried separately. Changed settings need a restart.
        """
        compose = self._load_resolved_compose()
        if not compose:
            return

        statuses = {
            service["container_name"]: self._service_status(service["container_name"])
            for service in compose.get("services", {}).values()
            if "container_name" in service
        }
        print(render_connection_info(compose, statuses, self._get_compose_project_name()))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Asclepius Healthcare Chatbot - Manage Docker Compose infrastructure services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_infrastructure.py --start     Start all services
  python manage_infrastructure.py --stop      Stop all services
  python manage_infrastructure.py --restart   Restart all services
  python manage_infrastructure.py --info      Show ports, URLs and credentials
        """,
    )

    parser.add_argument("--start", action="store_true", help="Start all infrastructure services")
    parser.add_argument("--stop", action="store_true", help="Stop all infrastructure services")
    parser.add_argument(
        "--restart", action="store_true", help="Restart all infrastructure services"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show service status, ports, URLs and credentials",
    )

    args = parser.parse_args()

    actions = [args.start, args.stop, args.restart, args.info]
    if sum(actions) != 1:
        parser.error("Please specify exactly one action: --start, --stop, --restart, or --info")

    try:
        manager = InfrastructureManager()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    success = False
    message = ""

    if args.start:
        print("Starting infrastructure services...")
        success, message = manager.start_services()
        if success:
            print(f"\n✓ {message}")
            manager.display_connection_info()
        else:
            print(f"\n✗ {message}", file=sys.stderr)
            sys.exit(1)

    elif args.stop:
        print("Stopping infrastructure services...")
        success, message = manager.stop_services()
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}", file=sys.stderr)
            sys.exit(1)

    elif args.info:
        manager.display_connection_info()

    elif args.restart:
        print("Restarting infrastructure services...")
        success, message = manager.restart_services()
        if success:
            print(f"\n✓ {message}")
            manager.display_connection_info()
        else:
            print(f"\n✗ {message}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
