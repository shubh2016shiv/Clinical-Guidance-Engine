"""
Smart Chainlit Runner - Automatically finds available port and launches Chainlit UI

This script intelligently selects an available port for the Chainlit application,
avoiding conflicts with other services like Milvus Attu (port 8000).

Features:
- Automatic port detection (tries preferred ports first)
- Conflict resolution with occupied ports
- Clean shutdown of previous instances
- Graceful error handling
- Cross-platform support (Windows/Linux/Mac)

Usage:
    python run_chainlit.py
    python run_chainlit.py --port 8080
    python run_chainlit.py --port-range 8080-8090
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


class ChainlitRunner:
    """Manages Chainlit application startup with intelligent port selection."""

    # Preferred ports to try (in order)
    PREFERRED_PORTS = [8080, 8081, 8082, 8888, 9000, 3000, 5000]

    # Ports to avoid (used by infrastructure services)
    RESERVED_PORTS = [8000, 19530, 27017, 6379, 9001, 9091, 2379]

    # Port range limits
    MIN_PORT = 1024
    MAX_PORT = 65535

    def __init__(self, app_path: Optional[Path] = None):
        """
        Initialize the Chainlit runner.

        Args:
            app_path: Path to Chainlit app.py file.
                     If None, uses script directory/app.py
        """
        if app_path is None:
            script_dir = Path(__file__).parent
            app_path = script_dir / "app.py"

        self.app_path = app_path.resolve()

        if not self.app_path.exists():
            raise FileNotFoundError(f"Chainlit app not found: {self.app_path}")

    def _is_port_in_use(self, port: int) -> bool:
        """
        Check if a port is currently in use.

        Args:
            port: Port number to check

        Returns:
            True if port is in use, False if available
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                # Try to bind to the port
                result = s.connect_ex(("127.0.0.1", port))
                return result == 0  # Port is in use if connection succeeds
        except socket.error:
            return True  # Assume port is in use if we can't check

    def _find_process_on_port(self, port: int) -> Optional[int]:
        """
        Find the process ID using a specific port.

        Args:
            port: Port number to check

        Returns:
            Process ID if found, None otherwise
        """
        try:
            if sys.platform == "win32":
                # Windows: Use netstat
                cmd = ["netstat", "-ano"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                for line in result.stdout.split("\n"):
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            try:
                                return int(pid)
                            except ValueError:
                                pass
            else:
                # Linux/Mac: Use lsof
                cmd = ["lsof", "-ti", f":{port}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split("\n")[0])
        except Exception as e:
            print(f"Warning: Could not find process on port {port}: {e}")

        return None

    def _kill_process(self, pid: int) -> bool:
        """
        Terminate a process by its PID.

        Args:
            pid: Process ID to terminate

        Returns:
            True if successful, False otherwise
        """
        try:
            if sys.platform == "win32":
                # Windows: Use taskkill
                cmd = ["taskkill", "/F", "/PID", str(pid)]
            else:
                # Linux/Mac: Use kill
                cmd = ["kill", "-9", str(pid)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            return result.returncode == 0
        except Exception as e:
            print(f"Warning: Could not kill process {pid}: {e}")
            return False

    def _get_process_name(self, pid: int) -> str:
        """
        Get the name of a process by its PID.

        Args:
            pid: Process ID

        Returns:
            Process name or "Unknown"
        """
        try:
            if sys.platform == "win32":
                # Windows: Use tasklist
                cmd = ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0 and result.stdout.strip():
                    # Parse CSV output
                    parts = result.stdout.strip().split(",")
                    if len(parts) > 0:
                        return parts[0].strip('"')
            else:
                # Linux/Mac: Use ps
                cmd = ["ps", "-p", str(pid), "-o", "comm="]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
        except Exception:
            pass

        return "Unknown"

    def _handle_port_conflict(self, port: int) -> Tuple[bool, str]:
        """
        Handle a port conflict by trying to stop the conflicting process.

        Args:
            port: Port that is in conflict

        Returns:
            Tuple of (resolved, message)
        """
        pid = self._find_process_on_port(port)

        if pid is None:
            return False, f"Port {port} is in use but could not identify the process"

        process_name = self._get_process_name(pid)

        # Check if it's a Chainlit process
        if "chainlit" in process_name.lower() or "python" in process_name.lower():
            print(f"\n⚠ Found existing process on port {port}:")
            print(f"  PID: {pid}")
            print(f"  Process: {process_name}")

            response = input(
                f"\nDo you want to terminate this process and use port {port}? (y/n): "
            )

            if response.lower() in ["y", "yes"]:
                print(f"Terminating process {pid}...")
                if self._kill_process(pid):
                    time.sleep(2)  # Wait for port to be released

                    # Verify port is now available
                    if not self._is_port_in_use(port):
                        return True, f"Successfully freed port {port}"
                    else:
                        return False, f"Port {port} still in use after termination"
                else:
                    return False, f"Failed to terminate process {pid}"
            else:
                return False, "User chose not to terminate existing process"
        else:
            return False, f"Port {port} is in use by {process_name} (PID: {pid})"

    def _find_available_port(
        self,
        preferred_port: Optional[int] = None,
        port_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[int]:
        """
        Find an available port for the application.

        Args:
            preferred_port: Specific port to try first
            port_range: Tuple of (min_port, max_port) to search within

        Returns:
            Available port number or None if not found
        """
        ports_to_try = []

        # Add preferred port first
        if preferred_port:
            if preferred_port not in self.RESERVED_PORTS:
                ports_to_try.append(preferred_port)
            else:
                print(f"⚠ Port {preferred_port} is reserved by infrastructure services")

        # Add other preferred ports
        for port in self.PREFERRED_PORTS:
            if port not in ports_to_try and port not in self.RESERVED_PORTS:
                ports_to_try.append(port)

        # Add port range if specified
        if port_range:
            min_port, max_port = port_range
            for port in range(min_port, max_port + 1):
                if (
                    port not in ports_to_try
                    and port not in self.RESERVED_PORTS
                    and self.MIN_PORT <= port <= self.MAX_PORT
                ):
                    ports_to_try.append(port)

        # Try each port
        print("\nSearching for available port...")
        for port in ports_to_try:
            print(f"  Checking port {port}...", end=" ")

            if not self._is_port_in_use(port):
                print("✓ Available")
                return port
            else:
                print("✗ In use")

                # Only try to resolve conflict for preferred port
                if port == preferred_port:
                    resolved, message = self._handle_port_conflict(port)
                    if resolved:
                        print(f"  ✓ {message}")
                        return port
                    else:
                        print(f"  ✗ {message}")

        return None

    def _check_chainlit_installed(self) -> Tuple[bool, str]:
        """
        Check if Chainlit is installed.

        Returns:
            Tuple of (is_installed, version_or_error)
        """
        try:
            # First try importing chainlit directly (faster, no subprocess)
            try:
                import chainlit

                version = getattr(
                    chainlit, "__version__", "installed (version unknown)"
                )
                return True, f"chainlit {version}"
            except ImportError:
                pass  # Fall through to subprocess check

            # Use the same Python interpreter to ensure we're using venv Python
            python_exe = sys.executable

            # Try using python -m chainlit first (works better with venv)
            # Increased timeout to 30 seconds for slow systems
            result = subprocess.run(
                [python_exe, "-m", "chainlit", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                return True, version
            else:
                # Fallback to direct chainlit command
                result = subprocess.run(
                    ["chainlit", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    return True, version
                else:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    return False, f"Chainlit command failed: {error_msg}"
        except FileNotFoundError:
            return False, "Chainlit not found. Install with: pip install chainlit"
        except subprocess.TimeoutExpired:
            return False, (
                "Chainlit command timed out after 30 seconds. "
                "This may indicate:\n"
                "  1. Chainlit installation issue - try: pip install --upgrade chainlit\n"
                "  2. Slow system or network issues\n"
                "  3. Python environment corruption - try recreating .venv"
            )
        except Exception as e:
            return False, f"Error checking Chainlit: {str(e)}"

    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_path = Path.cwd() / ".env"

        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
                print(f"✓ Loaded environment from {env_path}")
            except ImportError:
                print("⚠ python-dotenv not installed. Skipping .env file.")
            except Exception as e:
                print(f"⚠ Error loading .env file: {e}")

    def run(
        self,
        preferred_port: Optional[int] = None,
        port_range: Optional[Tuple[int, int]] = None,
        auto_reload: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run the Chainlit application.

        Args:
            preferred_port: Preferred port to use
            port_range: Port range to search if preferred is unavailable
            auto_reload: Enable auto-reload for development

        Returns:
            Tuple of (success, message)
        """
        # Check Chainlit installation
        installed, version_info = self._check_chainlit_installed()
        if not installed:
            return False, version_info

        print(f"✓ Chainlit installed: {version_info}")

        # Load environment variables
        self._load_env_file()

        # Find available port
        port = self._find_available_port(preferred_port, port_range)

        if port is None:
            return False, "No available ports found"

        print(f"\n{'=' * 60}")
        print(f"Starting Chainlit application on port {port}")
        print(f"App path: {self.app_path}")
        print(f"{'=' * 60}\n")

        # Build Chainlit command using the same Python interpreter
        python_exe = sys.executable
        cmd = [
            python_exe,
            "-m",
            "chainlit",
            "run",
            str(self.app_path),
            "--port",
            str(port),
        ]

        if not auto_reload:
            cmd.append("--no-watch")

        try:
            # Run Chainlit (blocking call)
            subprocess.run(cmd)
            return True, "Chainlit application stopped"
        except KeyboardInterrupt:
            print("\n\nShutting down Chainlit...")
            return True, "Chainlit application stopped by user"
        except Exception as e:
            return False, f"Error running Chainlit: {str(e)}"


def parse_port_range(port_range_str: str) -> Tuple[int, int]:
    """
    Parse port range string (e.g., '8080-8090').

    Args:
        port_range_str: Port range in format 'min-max'

    Returns:
        Tuple of (min_port, max_port)

    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = port_range_str.split("-")
        if len(parts) != 2:
            raise ValueError("Port range must be in format 'min-max'")

        min_port = int(parts[0].strip())
        max_port = int(parts[1].strip())

        if min_port < 1024 or max_port > 65535:
            raise ValueError("Ports must be between 1024 and 65535")

        if min_port >= max_port:
            raise ValueError("Min port must be less than max port")

        return min_port, max_port
    except ValueError as e:
        raise ValueError(f"Invalid port range '{port_range_str}': {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Chainlit Runner - Automatically finds available port",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chainlit.py                    # Use default preferred ports
  python run_chainlit.py --port 8080        # Try port 8080 first
  python run_chainlit.py --port-range 8080-8090  # Search in range
  python run_chainlit.py --no-reload        # Disable auto-reload

Reserved Ports (Avoided):
  8000  - Milvus Attu UI
  19530 - Milvus Database
  27017 - MongoDB
  6379  - Redis
  9000  - MinIO
  9001  - MinIO Console
        """,
    )

    parser.add_argument(
        "--port", type=int, help="Preferred port to use (will try this first)"
    )

    parser.add_argument(
        "--port-range", type=str, help="Port range to search (e.g., '8080-8090')"
    )

    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload during development",
    )

    parser.add_argument(
        "--app-path",
        type=Path,
        help="Path to Chainlit app.py file (default: ./chatbot_ui/app.py)",
    )

    args = parser.parse_args()

    # Parse port range if provided
    port_range = None
    if args.port_range:
        try:
            port_range = parse_port_range(args.port_range)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Initialize runner
    try:
        runner = ChainlitRunner(app_path=args.app_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Run application
    success, message = runner.run(
        preferred_port=args.port, port_range=port_range, auto_reload=not args.no_reload
    )

    if not success:
        print(f"\nError: {message}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n{message}")


if __name__ == "__main__":
    main()

"""
SEQUENCE DIAGRAM - Chainlit Smart Port Selection and Startup
============================================================

NORMAL STARTUP FLOW (Port Available):
--------------------------------------
User              Runner            System              Chainlit
 |                  |                  |                    |
 |--run------------>|                  |                    |
 |                  |--check_installed>|                    |
 |                  |<--version--------|                    |
 |                  |                  |                    |
 |                  |--load_env------->|                    |
 |                  |<--env_loaded-----|                    |
 |                  |                  |                    |
 |                  |--find_port------>|                    |
 |                  |--check_8080----->|                    |
 |                  |<--available------|                    |
 |                  |                  |                    |
 |                  |--start_chainlit->|                    |
 |                  |                  |--launch_app------->|
 |                  |                  |<--server_ready-----|
 |<--url_displayed--|                  |                    |
 |                  |                  |                    |

PORT CONFLICT RESOLUTION FLOW:
-------------------------------
User              Runner            System              Process
 |                  |                  |                    |
 |--run(8080)------>|                  |                    |
 |                  |--find_port------>|                    |
 |                  |--check_8080----->|                    |
 |                  |<--in_use---------|                    |
 |                  |                  |                    |
 |                  |--find_pid_8080-->|                    |
 |                  |<--pid_12345------|                    |
 |                  |--get_proc_name-->|                    |
 |                  |<--"python"-------|                    |
 |                  |                  |                    |
 |<--prompt_kill----|                  |                    |
 |--yes------------>|                  |                    |
 |                  |--kill_12345----->|                    |
 |                  |                  |--terminate-------->|
 |                  |                  |<--terminated-------|
 |                  |--wait(2s)--------|                    |
 |                  |--verify_free---->|                    |
 |                  |<--available------|                    |
 |                  |                  |                    |
 |                  |--start_chainlit->|                    |
 |<--success--------|                  |                    |
 |                  |                  |                    |

FALLBACK TO ALTERNATIVE PORT:
------------------------------
User              Runner            System
 |                  |                  |
 |--run(8080)------>|                  |
 |                  |--check_8080----->|
 |                  |<--in_use---------|
 |                  |--find_pid------->|
 |                  |<--pid_999--------|
 |                  |--get_name------->|
 |                  |<--"docker"-------|
 |                  |                  |
 |<--cannot_kill----|                  |
 |                  |                  |
 |                  |--check_8081----->|
 |                  |<--available------|
 |                  |                  |
 |<--using_8081-----|                  |
 |                  |--start(8081)---->|
 |<--success--------|                  |
 |                  |                  |

RESERVED PORT AVOIDANCE:
------------------------
User              Runner            System
 |                  |                  |
 |--run(8000)------>|                  |
 |                  |--check_reserved->|
 |                  |<--is_reserved----|
 |<--warning--------|                  |
 |                  |                  |
 |                  |--try_8080------->|
 |                  |<--available------|
 |<--using_8080-----|                  |
 |                  |--start---------->|
 |<--success--------|                  |
 |                  |                  |

KEY COMPONENTS:
--------------
1. _check_chainlit_installed(): Validates Chainlit is available
2. _load_env_file(): Loads environment variables from .env
3. _is_port_in_use(): Checks if port is occupied using socket
4. _find_process_on_port(): Identifies process using specific port
5. _get_process_name(): Retrieves process name for identification
6. _handle_port_conflict(): Manages conflict resolution with user prompt
7. _kill_process(): Terminates conflicting process safely
8. _find_available_port(): Intelligent port selection algorithm
9. run(): Main orchestration method

PORT SELECTION ALGORITHM:
------------------------
1. Check if preferred port specified
   - If reserved, skip and warn user
   - If available, use it
   - If in use, try to resolve conflict
2. Try preferred ports list (8080, 8081, 8082, 8888, ...)
3. Try port range if specified
4. Skip all reserved ports (8000, 19530, 27017, 6379, ...)
5. Return None if no ports available

CONFLICT RESOLUTION STRATEGY:
----------------------------
1. Detect port is in use
2. Find process ID using the port (netstat/lsof)
3. Get process name
4. Check if process is Python/Chainlit
   - If yes: Prompt user to terminate
   - If no: Skip to next port
5. If user agrees, terminate process
6. Wait 2 seconds for port release
7. Verify port is now available
8. Proceed with startup or try next port

ERROR HANDLING:
--------------
- Chainlit not installed → Clear error message
- No .env file → Continue without it (optional)
- No available ports → Report and exit
- Port conflict (non-Chainlit) → Auto-fallback to next port
- User declines termination → Try next available port
- Keyboard interrupt → Graceful shutdown

CROSS-PLATFORM SUPPORT:
----------------------
Windows: netstat, tasklist, taskkill
Linux/Mac: lsof, ps, kill
Automatic detection via sys.platform
"""
