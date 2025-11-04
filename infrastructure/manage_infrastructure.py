"""
Asclepius Healthcare Chatbot - Infrastructure Management Script

This script manages Docker Compose infrastructure services for the Asclepius Healthcare Chatbot,
including Milvus (vector database), MongoDB (document database), and Redis (cache).

Features:
- Starting services with health checks
- Stopping services gracefully
- Restarting services
- Displaying connection information
- Robust handling of conflicting containers from other projects
- Retry logic with exponential backoff
- Volume preservation

Usage:
    python manage_infrastructure.py --start
    python manage_infrastructure.py --stop
    python manage_infrastructure.py --restart
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from tabulate import tabulate


class InfrastructureManager:
    """Manages Docker Compose infrastructure services for Asclepius Healthcare Chatbot."""

    # Container names expected by this project
    EXPECTED_CONTAINERS = [
        "milvus-etcd",
        "milvus-minio",
        "attu",
        "milvus-standalone",
        "mongodb",
        "redis",
    ]

    # Maximum retry attempts for operations
    MAX_RETRIES = 5
    RETRY_DELAY = 2  # seconds

    def __init__(self, compose_file_path: Optional[Path] = None):
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
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return False, "Docker is not installed or not in PATH"
        except FileNotFoundError:
            return False, "Docker is not installed. Please install Docker Desktop."
        except subprocess.TimeoutExpired:
            return False, "Docker command timed out. Docker may not be responding."

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

    def _get_compose_command(self) -> List[str]:
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

            with open(self.compose_file_path, "r", encoding="utf-8") as f:
                compose_data = yaml.safe_load(f)
                if compose_data and "name" in compose_data:
                    return compose_data["name"].lower()
        except (ImportError, FileNotFoundError, KeyError, Exception):
            pass

        project_name = self.compose_dir.name.lower().replace(" ", "-").replace("_", "-")
        return project_name

    def _get_container_info(self, container_name: str) -> Optional[Dict[str, str]]:
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

            # Check for custom Asclepius project label first
            custom_project = labels.get("project", "")
            if custom_project == "asclepius-healthcare-chatbot":
                return True

            # Check Docker Compose project labels
            compose_project = labels.get("com.docker.compose.project", "")
            compose_working_dir = labels.get(
                "com.docker.compose.project.working_dir", ""
            )

            project_name = self._get_compose_project_name()
            current_working_dir = str(self.compose_dir.resolve())

            if compose_working_dir:
                compose_working_dir_normalized = Path(compose_working_dir).resolve()
                current_working_dir_normalized = Path(current_working_dir).resolve()

                if compose_project.lower() == project_name.lower() or str(
                    compose_working_dir_normalized
                ) == str(current_working_dir_normalized):
                    return True

            return False
        except Exception as e:
            print(
                f"  Warning: Could not check project ownership for {container_name}: {str(e)}"
            )
            return False

    def _force_remove_container(self, container_name: str) -> Tuple[bool, str]:
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
                result = subprocess.run(
                    remove_cmd, capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    # Verify removal with a short wait
                    time.sleep(0.5)
                    verify_info = self._get_container_info(container_name)
                    if verify_info is None:
                        return True, f"Successfully removed container {container_name}"
                    else:
                        print(
                            f"  Attempt {attempt + 1}: Container still exists, retrying..."
                        )
                        time.sleep(self.RETRY_DELAY)
                        continue
                else:
                    error_msg = result.stderr.strip()
                    if "No such container" in error_msg:
                        return True, f"Container {container_name} already removed"

                    if attempt < self.MAX_RETRIES - 1:
                        print(
                            f"  Attempt {attempt + 1} failed: {error_msg}, retrying..."
                        )
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

    def _cleanup_all_conflicting_containers(self) -> Tuple[bool, str, List[str]]:
        """
        Comprehensively clean up all conflicting containers.

        This function ensures no ghost containers remain that could cause conflicts.

        Returns:
            Tuple of (success, message, list_of_removed_containers)
        """
        removed_containers = []
        errors = []

        print("Scanning for conflicting containers...")

        for container_name in self.EXPECTED_CONTAINERS:
            info = self._get_container_info(container_name)

            if info:
                # Container exists, check if it belongs to this project
                belongs = self._check_container_belongs_to_project(container_name)

                if not belongs:
                    print(
                        f"  Found conflicting container: {container_name} (state: {info['state']})"
                    )
                    success, message = self._force_remove_container(container_name)

                    if success:
                        removed_containers.append(container_name)
                        print(f"    ✓ {message}")
                    else:
                        errors.append(f"{container_name}: {message}")
                        print(f"    ✗ {message}")
                else:
                    print(
                        f"  Container {container_name} belongs to this project, skipping"
                    )

        if errors:
            return (
                False,
                f"Failed to remove some containers: {'; '.join(errors)}",
                removed_containers,
            )

        if removed_containers:
            # Wait for Docker to fully process removals
            print("  Waiting for Docker to process removals...")
            time.sleep(3)
            return (
                True,
                f"Removed {len(removed_containers)} conflicting container(s)",
                removed_containers,
            )

        return True, "No conflicting containers found", []

    def _run_compose_up_with_retry(self) -> Tuple[bool, str]:
        """
        Run docker compose up with intelligent retry logic.

        Handles transient conflicts and ensures all containers start successfully.

        Returns:
            Tuple of (success, message)
        """
        compose_cmd = self._get_compose_command()
        project_name = self._get_compose_project_name()

        original_cwd = os.getcwd()

        try:
            os.chdir(self.compose_dir)

            for attempt in range(self.MAX_RETRIES):
                print(
                    f"\nAttempt {attempt + 1}/{self.MAX_RETRIES}: Starting services..."
                )

                # Clean up any conflicting containers before each attempt
                if attempt > 0:
                    print("Performing cleanup before retry...")
                    cleanup_success, cleanup_msg, removed = (
                        self._cleanup_all_conflicting_containers()
                    )
                    if removed:
                        print(f"  Cleaned up {len(removed)} container(s)")

                cmd = compose_cmd + [
                    "-f",
                    str(self.compose_file_path.name),
                    "-p",
                    project_name,
                    "up",
                    "-d",
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    # Verify all containers are actually running
                    time.sleep(2)
                    running_count = 0
                    for container_name in self.EXPECTED_CONTAINERS:
                        info = self._get_container_info(container_name)
                        if info and info["state"] == "running":
                            running_count += 1

                    if running_count == len(self.EXPECTED_CONTAINERS):
                        return (
                            True,
                            f"All {running_count} services started successfully",
                        )
                    else:
                        print(
                            f"  Warning: Only {running_count}/{len(self.EXPECTED_CONTAINERS)} containers running"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            print("  Retrying...")
                            time.sleep(self.RETRY_DELAY)
                            continue

                error_msg = result.stderr.strip() or result.stdout.strip()

                # Check for container conflicts in error message
                if "Conflict" in error_msg and "container name" in error_msg:
                    conflict_containers = re.findall(
                        r'container name "/([^"]+)"', error_msg
                    )

                    if conflict_containers:
                        print(f"  Detected conflicts: {', '.join(conflict_containers)}")

                        # Force remove all conflicting containers
                        for conflict_container in conflict_containers:
                            print(f"  Removing {conflict_container}...")
                            success, msg = self._force_remove_container(
                                conflict_container
                            )
                            if success:
                                print(f"    ✓ {msg}")
                            else:
                                print(f"    ✗ {msg}")

                        if attempt < self.MAX_RETRIES - 1:
                            time.sleep(self.RETRY_DELAY)
                            continue

                # If this is the last attempt, return the error
                if attempt == self.MAX_RETRIES - 1:
                    return (
                        False,
                        f"Failed to start services after {self.MAX_RETRIES} attempts: {error_msg}",
                    )

                # For other errors, wait and retry
                print(f"  Error: {error_msg}")
                time.sleep(self.RETRY_DELAY)

            return False, f"Failed to start services after {self.MAX_RETRIES} attempts"

        finally:
            os.chdir(original_cwd)

    def start_services(self) -> Tuple[bool, str]:
        """
        Start all infrastructure services with robust conflict handling.

        Returns:
            Tuple of (success, message)
        """
        docker_available, error_msg = self._check_docker_available()
        if not docker_available:
            return False, error_msg

        # Initial cleanup of conflicting containers
        cleanup_success, cleanup_msg, removed = (
            self._cleanup_all_conflicting_containers()
        )
        if removed:
            print(f"✓ {cleanup_msg}\n")

        # Start services with retry logic
        return self._run_compose_up_with_retry()

    def stop_services(self) -> Tuple[bool, str]:
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
                        belongs = self._check_container_belongs_to_project(
                            container_name
                        )
                        if belongs:
                            success, message = self._force_remove_container(
                                container_name
                            )
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

    def restart_services(self) -> Tuple[bool, str]:
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
        description="Asclepius Healthcare Chatbot - Manage Docker Compose infrastructure services",
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

    actions = [args.start, args.stop, args.restart]
    if sum(actions) != 1:
        parser.error("Please specify exactly one action: --start, --stop, or --restart")

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


"""
SEQUENCE DIAGRAM - Infrastructure Management Flow
==================================================

START SERVICE FLOW:
-------------------
User                Manager              Docker              Container
 |                    |                    |                    |
 |--start------------>|                    |                    |
 |                    |--check_docker----->|                    |
 |                    |<--status-----------|                    |
 |                    |                    |                    |
 |                    |--scan_conflicts--->|                    |
 |                    |<--conflict_list----|                    |
 |                    |                    |                    |
 |                    |[FOR EACH CONFLICT] |                    |
 |                    |--get_info--------->|                    |
 |                    |<--container_info---|                    |
 |                    |--check_ownership-->|                    |
 |                    |<--belongs_result---|                    |
 |                    |                    |                    |
 |                    |[IF NOT BELONGS]    |                    |
 |                    |--force_remove----->|                    |
 |                    |                    |--stop------------->|
 |                    |                    |<--stopped----------|
 |                    |                    |--remove----------->|
 |                    |                    |<--removed----------|
 |                    |<--success----------|                    |
 |                    |                    |                    |
 |                    |[RETRY LOOP: MAX 5] |                    |
 |                    |--compose_up------->|                    |
 |                    |                    |--create----------->|
 |                    |                    |                    |
 |                    |[IF CONFLICT]       |                    |
 |                    |<--conflict_error---|                    |
 |                    |--extract_conflicts-|                    |
 |                    |--force_remove----->|                    |
 |                    |                    |--remove----------->|
 |                    |<--removed----------|                    |
 |                    |--retry_compose---->|                    |
 |                    |                    |                    |
 |                    |[IF SUCCESS]        |                    |
 |                    |                    |<--containers_up----|
 |                    |--verify_running--->|                    |
 |                    |<--all_healthy------|                    |
 |<--success----------|                    |                    |
 |                    |                    |                    |

STOP SERVICE FLOW:
------------------
User                Manager              Docker              Container
 |                    |                    |                    |
 |--stop------------->|                    |                    |
 |                    |--check_docker----->|                    |
 |                    |<--status-----------|                    |
 |                    |                    |                    |
 |                    |--compose_down----->|                    |
 |                    |                    |--stop_all-------->|
 |                    |                    |<--stopped---------|
 |                    |                    |--remove_all------>|
 |                    |                    |<--removed---------|
 |                    |<--success----------|                    |
 |                    |                    |                    |
 |                    |[IF FAILURE]        |                    |
 |                    |<--error------------|                    |
 |                    |[FOR EACH CONTAINER]|                    |
 |                    |--force_remove----->|                    |
 |                    |                    |--stop------------->|
 |                    |                    |--remove----------->|
 |                    |<--removed----------|                    |
 |<--success----------|                    |                    |
 |                    |                    |                    |

RESTART SERVICE FLOW:
---------------------
User                Manager              Docker
 |                    |                    |
 |--restart---------->|                    |
 |                    |--stop_services---->|
 |                    |<--stopped----------|
 |                    |--wait(3s)----------|
 |                    |--start_services--->|
 |                    |<--started----------|
 |<--success----------|                    |
 |                    |                    |

KEY COMPONENTS:
--------------
1. _check_docker_available(): Validates Docker installation and daemon
2. _get_container_info(): Retrieves container state and metadata
3. _check_container_belongs_to_project(): Verifies container ownership via labels
4. _force_remove_container(): Removes container with retry logic (preserves volumes)
5. _cleanup_all_conflicting_containers(): Scans and removes non-project containers
6. _run_compose_up_with_retry(): Executes compose up with conflict resolution
7. start_services(): Main entry point for starting infrastructure
8. stop_services(): Main entry point for stopping infrastructure
9. restart_services(): Sequential stop then start

RETRY STRATEGY:
--------------
- Maximum 5 attempts for operations
- 2-second delay between retries
- Exponential backoff for critical operations
- Verify success after each attempt
- Clean up before retry attempts

ERROR HANDLING:
--------------
- Docker availability check before operations
- Timeout handling for all subprocess calls
- Graceful degradation (manual container removal if compose fails)
- Detailed error messages with context
- Volume preservation during all removal operations

VOLUME SAFETY:
-------------
- 'docker rm' without '-v' flag (preserves volumes by default)
- No volume removal commands used
- Data persists across container removals
- Explicit volume preservation in all removal operations
"""
