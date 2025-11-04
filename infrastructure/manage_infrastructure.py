"""
Asclepius Healthcare Chatbot - Infrastructure Management Script

This script manages Docker Compose infrastructure services for the Asclepius Healthcare Chatbot,
including Milvus (vector database), MongoDB (document database), and Redis (cache).

Features:
- Starting services with health checks
- Stopping services gracefully
- Restarting services
- Displaying connection information
- Handling conflicting containers from other projects

Usage:
    python manage_infrastructure.py --start
    python manage_infrastructure.py --stop
    python manage_infrastructure.py --restart
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple, Optional
from tabulate import tabulate


class InfrastructureManager:
    """Manages Docker Compose infrastructure services for Asclepius Healthcare Chatbot."""

    def __init__(self, compose_file_path: Optional[Path] = None):
        """
        Initialize the infrastructure manager.

        Args:
            compose_file_path: Path to docker-compose.yml file.
                              If None, uses script directory/docker-compose.yml
        """
        if compose_file_path is None:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent
            compose_file_path = script_dir / "docker-compose.yml"

        self.compose_file_path = compose_file_path.resolve()
        self.compose_dir = self.compose_file_path.parent

        if not self.compose_file_path.exists():
            raise FileNotFoundError(
                f"Docker Compose file not found: {self.compose_file_path}"
            )

    def _check_docker_available(self) -> Tuple[bool, str]:
        """
        Check if Docker is installed and the daemon is running.

        Returns:
            Tuple of (is_available, error_message)
            If available, error_message is empty string.
        """
        try:
            # Check if docker command exists
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False, "Docker is not installed or not in PATH"
        except FileNotFoundError:
            return False, "Docker is not installed. Please install Docker Desktop."
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out. Docker may not be responding."

        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Docker daemon is not running"
                if (
                    "Cannot connect" in error_msg
                    or "Is the docker daemon running" in error_msg
                ):
                    return (
                        False,
                        "Docker daemon is not running. Please start Docker Desktop.",
                    )
                return False, f"Docker daemon error: {error_msg}"
        except subprocess.TimeoutExpired:
            return (
                False,
                "Docker daemon is not responding. Please check Docker Desktop status.",
            )

        return True, ""

    def _get_compose_command(self) -> list[str]:
        """
        Get the appropriate docker compose command.

        Returns:
            List of command parts (e.g., ['docker', 'compose'] or ['docker-compose'])
        """
        # Try docker compose (v2) first
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

        # Fall back to docker-compose (v1)
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
        # Try to read the project name from docker-compose.yml
        try:
            import yaml

            with open(self.compose_file_path, "r", encoding="utf-8") as f:
                compose_data = yaml.safe_load(f)
                if compose_data and "name" in compose_data:
                    return compose_data["name"].lower()
        except (ImportError, FileNotFoundError, KeyError, Exception):
            # Fall back to directory name if YAML parsing fails or name not found
            pass

        # Docker Compose uses the directory name (lowercased) as project name
        project_name = self.compose_dir.name.lower().replace(" ", "-").replace("_", "-")
        return project_name

    def _check_container_exists(
        self, container_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a container exists (running or stopped).

        Args:
            container_name: Name of the container to check

        Returns:
            Tuple of (exists, container_id). container_id is None if doesn't exist.
        """
        try:
            # Check all containers (including stopped)
            cmd = [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.ID}}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                container_id = result.stdout.strip().split("\n")[0]
                return True, container_id
            return False, None
        except Exception:
            return False, None

    def _check_container_belongs_to_project(self, container_name: str) -> bool:
        """
        Check if a container belongs to this Docker Compose project.

        Docker Compose adds labels to containers:
        - com.docker.compose.project: project name
        - com.docker.compose.project.working_dir: working directory

        Args:
            container_name: Name of the container to check

        Returns:
            True if container belongs to this project, False otherwise
        """
        try:
            # Get container labels
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

            # Check for custom Asclepius project label first
            custom_project = labels.get("project", "")
            if custom_project == "asclepius-healthcare-chatbot":
                return True

            # Check if container has compose project label
            compose_project = labels.get("com.docker.compose.project", "")
            compose_working_dir = labels.get(
                "com.docker.compose.project.working_dir", ""
            )

            # Check if project name matches or working directory matches
            project_name = self._get_compose_project_name()
            current_working_dir = str(self.compose_dir.resolve())

            # Normalize paths for comparison (handle Windows/Unix path differences)
            if compose_working_dir:
                compose_working_dir_normalized = Path(compose_working_dir).resolve()
                current_working_dir_normalized = Path(current_working_dir).resolve()

                if compose_project.lower() == project_name.lower() or str(
                    compose_working_dir_normalized
                ) == str(current_working_dir_normalized):
                    return True

            return False
        except Exception:
            # If we can't determine, assume it doesn't belong (safer to remove)
            return False

    def _stop_and_remove_container(
        self, container_name: str, preserve_volumes: bool = True
    ) -> Tuple[bool, str]:
        """
        Stop and remove a container, optionally preserving volumes.

        Args:
            container_name: Name of the container to stop/remove
            preserve_volumes: If True, don't remove volumes (default: True)

        Returns:
            Tuple of (success, message)
        """
        try:
            # First, stop the container if it's running
            stop_cmd = ["docker", "stop", container_name]
            stop_result = subprocess.run(
                stop_cmd, capture_output=True, text=True, timeout=30
            )

            # Stop command might fail if container is already stopped, that's okay
            if (
                stop_result.returncode != 0
                and "No such container" not in stop_result.stderr
            ):
                # If container doesn't exist, that's fine
                if "No such container" not in stop_result.stderr:
                    pass  # Continue anyway

            # Remove the container (without volumes if preserve_volumes is True)
            remove_cmd = ["docker", "rm", container_name]
            if preserve_volumes:
                # Default docker rm doesn't remove volumes, so we're good
                pass
            else:
                remove_cmd.append("-v")  # Remove volumes

            remove_result = subprocess.run(
                remove_cmd, capture_output=True, text=True, timeout=30
            )

            if remove_result.returncode != 0:
                if "No such container" in remove_result.stderr:
                    return True, f"Container {container_name} already removed"
                return (
                    False,
                    f"Failed to remove container {container_name}: {remove_result.stderr}",
                )

            return (
                True,
                f"Successfully stopped and removed container {container_name} (volumes preserved)",
            )

        except subprocess.TimeoutExpired:
            return False, f"Timeout while removing container {container_name}"
        except Exception as e:
            return False, f"Error removing container {container_name}: {str(e)}"

    def _handle_conflicting_containers(self) -> Tuple[bool, str, list[str]]:
        """
        Check for conflicting containers and handle them.

        If containers with the same names exist but don't belong to this project,
        stop and remove them (preserving volumes).

        Returns:
            Tuple of (success, message, list_of_handled_containers)
        """
        expected_containers = [
            "milvus-etcd",
            "milvus-minio",
            "attu",
            "milvus-standalone",
            "mongodb",
            "redis",
        ]

        conflicting_containers = []
        handled_containers = []

        # Check each expected container
        for container_name in expected_containers:
            exists, container_id = self._check_container_exists(container_name)

            if exists:
                # Check if it belongs to this project
                belongs_to_project = self._check_container_belongs_to_project(
                    container_name
                )

                if not belongs_to_project:
                    conflicting_containers.append(container_name)

        if not conflicting_containers:
            return True, "No conflicting containers found", []

        # Handle conflicting containers
        print(
            f"Found {len(conflicting_containers)} conflicting container(s) from other projects:"
        )
        for container in conflicting_containers:
            print(f"  - {container}")
        print("Stopping and removing them (volumes will be preserved)...")

        errors = []
        for container_name in conflicting_containers:
            success, message = self._stop_and_remove_container(
                container_name, preserve_volumes=True
            )
            if success:
                handled_containers.append(container_name)
                print(f"  ✓ {message}")
            else:
                errors.append(f"{container_name}: {message}")
                print(f"  ✗ Failed to handle {container_name}: {message}")

        if errors:
            return (
                False,
                f"Failed to handle some conflicting containers: {'; '.join(errors)}",
                handled_containers,
            )

        return (
            True,
            f"Successfully handled {len(handled_containers)} conflicting container(s)",
            handled_containers,
        )

    def _run_compose_command(
        self, action: str, check_existing: bool = False
    ) -> Tuple[bool, str]:
        """
        Run a docker compose command.

        Args:
            action: Action to perform ('up', 'down', 'restart', 'ps')
            check_existing: If True, check container status before action

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

        # Change to compose file directory for relative paths
        original_cwd = os.getcwd()

        try:
            os.chdir(self.compose_dir)

            if action == "ps":
                # Check status of containers
                cmd = compose_cmd + [
                    "-f",
                    str(self.compose_file_path.name),
                    "ps",
                    "--format",
                    "json",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                return result.returncode == 0, result.stdout

            elif action == "up":
                # Handle conflicting containers from other projects first
                conflict_success, conflict_message, handled_containers = (
                    self._handle_conflicting_containers()
                )
                if not conflict_success:
                    return (
                        False,
                        f"Failed to handle conflicting containers: {conflict_message}",
                    )

                if handled_containers:
                    print(
                        f"  Handled {len(handled_containers)} conflicting container(s)\n"
                    )

                # Start services in detached mode
                cmd = compose_cmd + ["-f", str(self.compose_file_path.name), "up", "-d"]

                if check_existing:
                    # Check if containers from this project are already running
                    status_success, status_output = self._run_compose_command("ps")
                    if status_success and status_output:
                        # Parse JSON output to check if containers are running
                        containers = [
                            "milvus-etcd",
                            "milvus-minio",
                            "attu",
                            "milvus-standalone",
                            "mongodb",
                            "redis",
                        ]
                        running_containers = []
                        for container in containers:
                            # Check if container exists and belongs to this project
                            exists, _ = self._check_container_exists(container)
                            if exists:
                                belongs = self._check_container_belongs_to_project(
                                    container
                                )
                                if belongs:
                                    # Check if it's actually running
                                    check_cmd = [
                                        "docker",
                                        "ps",
                                        "--filter",
                                        f"name=^{container}$",
                                        "--format",
                                        "{{.Names}}",
                                    ]
                                    check_result = subprocess.run(
                                        check_cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=10,
                                    )
                                    if (
                                        check_result.returncode == 0
                                        and container in check_result.stdout
                                    ):
                                        running_containers.append(container)

                        if running_containers:
                            return (
                                True,
                                f"Some containers are already running: {', '.join(running_containers)}. Continuing with start...",
                            )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # Increased timeout for pulling images
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return False, f"Failed to start services: {error_msg}"

                return True, "Services started successfully"

            elif action == "down":
                # Stop services
                cmd = compose_cmd + ["-f", str(self.compose_file_path.name), "down"]

                if check_existing:
                    # Check if containers are already stopped
                    containers = [
                        "milvus-etcd",
                        "milvus-minio",
                        "attu",
                        "milvus-standalone",
                        "mongodb",
                        "redis",
                    ]
                    running_containers = []
                    for container in containers:
                        check_cmd = [
                            "docker",
                            "ps",
                            "--filter",
                            f"name={container}",
                            "--format",
                            "{{.Names}}",
                        ]
                        check_result = subprocess.run(
                            check_cmd, capture_output=True, text=True, timeout=10
                        )
                        if (
                            check_result.returncode == 0
                            and container in check_result.stdout
                        ):
                            running_containers.append(container)

                    if not running_containers:
                        return True, "All containers are already stopped."

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return False, f"Failed to stop services: {error_msg}"

                return True, "Services stopped successfully"

            elif action == "restart":
                # Restart services
                cmd = compose_cmd + ["-f", str(self.compose_file_path.name), "restart"]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    return False, f"Failed to restart services: {error_msg}"

                return True, "Services restarted successfully"

            else:
                return False, f"Unknown action: {action}"

        except subprocess.TimeoutExpired:
            return False, f"Command timed out while executing: {action}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
        finally:
            os.chdir(original_cwd)

    def start_services(self) -> Tuple[bool, str]:
        """
        Start all infrastructure services.

        Returns:
            Tuple of (success, message)
        """
        return self._run_compose_command("up", check_existing=True)

    def stop_services(self) -> Tuple[bool, str]:
        """
        Stop all infrastructure services.

        Returns:
            Tuple of (success, message)
        """
        return self._run_compose_command("down", check_existing=True)

    def restart_services(self) -> Tuple[bool, str]:
        """
        Restart all infrastructure services.

        Returns:
            Tuple of (success, message)
        """
        return self._run_compose_command("restart")

    def display_connection_info(self):
        """
        Display connection information table for all services.
        """
        connection_data = [
            ["Milvus", "localhost:19530", "-", "-"],
            ["Attu UI", "http://localhost:8000", "-", "-"],
            ["MongoDB", "mongodb://localhost:27017", "admin", "password123"],
            ["MinIO", "http://localhost:9000", "minioadmin", "minioadmin"],
            ["Redis", "redis://localhost:6379", "-", "redis123"],
        ]

        headers = ["Service", "URL/Connection String", "Username", "Password"]

        print("\n" + "=" * 80)
        print("ASCLEPIUS HEALTHCARE CHATBOT - INFRASTRUCTURE CONNECTION INFORMATION")
        print("=" * 80)
        print(tabulate(connection_data, headers=headers, tablefmt="grid"))
        print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Asclepius Healthcare Chatbot - Manage Docker Compose infrastructure services (Milvus, MongoDB & Redis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_infrastructure.py --start     Start all services
  python manage_infrastructure.py --stop      Stop all services
  python manage_infrastructure.py --restart   Restart all services
        """,
    )

    parser.add_argument(
        "--start", action="store_true", help="Start all infrastructure services"
    )

    parser.add_argument(
        "--stop", action="store_true", help="Stop all infrastructure services"
    )

    parser.add_argument(
        "--restart", action="store_true", help="Restart all infrastructure services"
    )

    args = parser.parse_args()

    # Validate that exactly one action is specified
    actions = [args.start, args.stop, args.restart]
    if sum(actions) != 1:
        parser.error("Please specify exactly one action: --start, --stop, or --restart")

    try:
        manager = InfrastructureManager()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute the requested action
    success = False
    message = ""

    if args.start:
        print("Starting infrastructure services...")
        success, message = manager.start_services()
        if success:
            print(f"✓ {message}")
            manager.display_connection_info()
        else:
            print(f"✗ {message}", file=sys.stderr)
            sys.exit(1)

    elif args.stop:
        print("Stopping infrastructure services...")
        success, message = manager.stop_services()
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}", file=sys.stderr)
            sys.exit(1)

    elif args.restart:
        print("Restarting infrastructure services...")
        success, message = manager.restart_services()
        if success:
            print(f"✓ {message}")
            manager.display_connection_info()
        else:
            print(f"✗ {message}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
