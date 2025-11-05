"""
Redis Cache Flush Utility - Production Grade

A comprehensive utility to safely flush Redis cache data with advanced features:
- Multiple flush strategies (full DB, prefix-based, pattern matching)
- Robust connection handling with retry logic
- Detailed operation statistics and progress tracking
- Safe confirmation prompts with preview
- Production-grade error handling and logging
- Environment configuration support

Usage:
    python infrastructure/flush_redis.py                    # Flush keys with default prefix
    python infrastructure/flush_redis.py --full             # Flush entire database
    python infrastructure/flush_redis.py --prefix "custom"  # Custom prefix
    python infrastructure/flush_redis.py --no-confirm       # Skip confirmation
    python infrastructure/flush_redis.py --dry-run          # Preview without deletion
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


# ============================================================================
# Configuration Constants
# ============================================================================


class RedisDefaults:
    """Default Redis configuration values."""

    HOST = "localhost"
    PORT = 6379
    DB = 0
    PASSWORD = "redis123"
    KEY_PREFIX = "drug_reco"
    SOCKET_TIMEOUT = 5
    SOCKET_CONNECT_TIMEOUT = 5
    RETRY_ON_TIMEOUT = True


class OperationConfig:
    """Operation configuration constants."""

    SCAN_BATCH_SIZE = 1000
    DELETE_BATCH_SIZE = 1000
    PROGRESS_UPDATE_INTERVAL = 10  # Show progress every N batches
    MAX_KEY_SAMPLES = 10
    CONNECTION_RETRY_ATTEMPTS = 3
    CONNECTION_RETRY_DELAY = 1.0  # seconds


# ============================================================================
# ANSI Color Codes for Terminal Output
# ============================================================================


class TerminalColors:
    """ANSI color codes for styled terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class FlushStrategy(Enum):
    """Available flush strategies."""

    FULL_DATABASE = "full_database"
    PREFIX_PATTERN = "prefix_pattern"


@dataclass
class FlushOperationStats:
    """Statistics for a flush operation."""

    strategy: FlushStrategy
    prefix_pattern: Optional[str]
    total_keys_before: int
    total_keys_after: int
    keys_deleted: int
    batches_processed: int
    duration_seconds: float
    keys_per_second: float

    def __str__(self) -> str:
        """Format statistics as string."""
        return (
            f"Strategy: {self.strategy.value}\n"
            f"Keys Deleted: {self.keys_deleted:,}\n"
            f"Duration: {self.duration_seconds:.2f}s\n"
            f"Speed: {self.keys_per_second:.0f} keys/sec"
        )


# ============================================================================
# Terminal Output Formatter
# ============================================================================


class TerminalFormatter:
    """Handles formatted terminal output with color support."""

    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.

        Args:
            use_colors: Whether to use ANSI color codes
        """
        self.use_colors = use_colors and self._terminal_supports_colors()

    @staticmethod
    def _terminal_supports_colors() -> bool:
        """Check if terminal supports ANSI color codes."""
        if sys.platform == "win32":
            return os.getenv("TERM") in ("xterm", "xterm-256color") or bool(
                os.getenv("ANSICON")
            )
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _colorize(self, text: str, color: str = "", bold: bool = False) -> str:
        """Apply color and style to text."""
        if not self.use_colors or not color:
            return text

        style = TerminalColors.BOLD if bold else ""
        return f"{style}{color}{text}{TerminalColors.RESET}"

    def print_header(self, title: str, width: int = 80) -> None:
        """Print a formatted header."""
        separator = "=" * width
        print(f"\n{self._colorize(separator, TerminalColors.CYAN, bold=True)}")
        print(self._colorize(f"  {title}", TerminalColors.CYAN, bold=True))
        print(f"{self._colorize(separator, TerminalColors.CYAN, bold=True)}\n")

    def print_subheader(self, title: str, width: int = 80) -> None:
        """Print a formatted subheader."""
        separator = "-" * width
        print(f"\n{self._colorize(separator, TerminalColors.GRAY)}")
        print(self._colorize(f"  {title}", TerminalColors.WHITE, bold=True))
        print(f"{self._colorize(separator, TerminalColors.GRAY)}\n")

    def print_success(self, message: str) -> None:
        """Print success message."""
        print(self._colorize(f"✓ {message}", TerminalColors.GREEN))

    def print_error(self, message: str) -> None:
        """Print error message."""
        print(self._colorize(f"✗ {message}", TerminalColors.RED, bold=True))

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(self._colorize(f"⚠ {message}", TerminalColors.YELLOW))

    def print_info(self, message: str) -> None:
        """Print info message."""
        print(self._colorize(f"ℹ {message}", TerminalColors.BLUE))

    def print_detail(
        self, label: str, value: Any, color: str = TerminalColors.CYAN
    ) -> None:
        """Print a labeled detail."""
        formatted_label = self._colorize(f"{label}:", color, bold=True)
        print(f"  {formatted_label} {value}")

    def print_key_value(self, key: str, value: Any) -> None:
        """Print key-value pair."""
        key_formatted = self._colorize(key, TerminalColors.GRAY)
        print(f"  {key_formatted}: {value}")


# ============================================================================
# Redis Connection Manager
# ============================================================================


class RedisConnectionManager:
    """Manages Redis connections with retry logic and validation."""

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        socket_timeout: int,
        socket_connect_timeout: int,
        formatter: TerminalFormatter,
    ):
        """
        Initialize connection manager.

        Args:
            host: Redis host address
            port: Redis port number
            db: Redis database index
            password: Redis password (None if no auth)
            socket_timeout: Socket operation timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            formatter: Terminal output formatter
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.formatter = formatter
        self.client = None

    def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        Establish connection to Redis with retry logic.

        Args:
            max_retries: Maximum connection attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            import redis
        except ImportError:
            self.formatter.print_error(
                "Redis package not installed. Install with: pip install redis"
            )
            return False

        for attempt in range(1, max_retries + 1):
            try:
                self.formatter.print_info(
                    f"Connecting to Redis at {self.host}:{self.port} "
                    f"(DB {self.db}) - Attempt {attempt}/{max_retries}"
                )

                # Build connection parameters
                connection_params = {
                    "host": self.host,
                    "port": self.port,
                    "db": self.db,
                    "socket_timeout": self.socket_timeout,
                    "socket_connect_timeout": self.socket_connect_timeout,
                    "decode_responses": False,  # Work with raw bytes
                    "health_check_interval": 30,
                }

                # Add password if provided
                if self.password:
                    connection_params["password"] = self.password

                # Create client
                self.client = redis.Redis(**connection_params)

                # Test connection with PING
                self.client.ping()

                self.formatter.print_success("Connected to Redis successfully")
                return True

            except redis.AuthenticationError as e:
                self.formatter.print_error(f"Redis authentication failed: {e}")
                self.formatter.print_info(
                    "Check your REDIS_PASSWORD in .env file or command line arguments"
                )
                return False

            except redis.ConnectionError as e:
                self.formatter.print_error(f"Connection failed: {e}")

                if attempt < max_retries:
                    self.formatter.print_info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.formatter.print_error("Max connection retries reached")
                    self.formatter.print_info(
                        "Ensure Redis is running: "
                        "python infrastructure/manage_infrastructure.py --start"
                    )
                    return False

            except Exception as e:
                self.formatter.print_error(f"Unexpected error: {e}")
                return False

        return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Retrieve Redis server information.

        Returns:
            Dictionary containing server info
        """
        try:
            info = self.client.info()
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "redis_mode": info.get("redis_mode", "standalone"),
                "os": info.get("os", "unknown"),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_days": info.get("uptime_in_days", 0),
            }
        except Exception as e:
            self.formatter.print_warning(f"Could not retrieve server info: {e}")
            return {}

    def get_database_size(self) -> int:
        """
        Get total number of keys in current database.

        Returns:
            Number of keys
        """
        try:
            return self.client.dbsize()
        except Exception as e:
            self.formatter.print_error(f"Failed to get database size: {e}")
            return 0

    def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors on close


# ============================================================================
# Redis Key Inspector
# ============================================================================


class RedisKeyInspector:
    """Inspects Redis keys for preview and analysis."""

    def __init__(self, client, formatter: TerminalFormatter):
        """
        Initialize key inspector.

        Args:
            client: Redis client instance
            formatter: Terminal output formatter
        """
        self.client = client
        self.formatter = formatter

    def count_keys_matching_pattern(self, pattern: str) -> int:
        """
        Count keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "prefix*")

        Returns:
            Number of matching keys
        """
        try:
            count = 0
            for _ in self.client.scan_iter(
                match=pattern, count=OperationConfig.SCAN_BATCH_SIZE
            ):
                count += 1
            return count
        except Exception as e:
            self.formatter.print_error(f"Error counting keys: {e}")
            return 0

    def get_sample_keys(
        self, pattern: str, max_samples: int = OperationConfig.MAX_KEY_SAMPLES
    ) -> List[str]:
        """
        Get sample keys matching a pattern.

        Args:
            pattern: Redis key pattern (use "*" for all keys)
            max_samples: Maximum number of samples to retrieve

        Returns:
            List of sample key names
        """
        try:
            samples = []
            for key in self.client.scan_iter(
                match=pattern, count=OperationConfig.SCAN_BATCH_SIZE
            ):
                decoded_key = key.decode("utf-8") if isinstance(key, bytes) else key
                samples.append(decoded_key)
                if len(samples) >= max_samples:
                    break
            return samples
        except Exception as e:
            self.formatter.print_warning(f"Could not retrieve key samples: {e}")
            return []

    def get_all_keys_sample(
        self, max_samples: int = OperationConfig.MAX_KEY_SAMPLES
    ) -> List[str]:
        """
        Get a sample of all keys in the database (regardless of pattern).

        Args:
            max_samples: Maximum number of samples to retrieve

        Returns:
            List of sample key names
        """
        return self.get_sample_keys("*", max_samples)

    def analyze_key_patterns(self, sample_keys: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns in sample keys.

        Args:
            sample_keys: List of key names to analyze

        Returns:
            Dictionary with pattern analysis
        """
        if not sample_keys:
            return {}

        # Extract common prefixes
        prefixes = set()
        for key in sample_keys:
            if ":" in key:
                prefix = key.split(":")[0]
                prefixes.add(prefix)

        # Calculate average key length
        avg_length = sum(len(k) for k in sample_keys) / len(sample_keys)

        return {
            "common_prefixes": list(prefixes),
            "average_key_length": avg_length,
            "sample_count": len(sample_keys),
        }


# ============================================================================
# Redis Flush Executor
# ============================================================================


class RedisFlushExecutor:
    """Executes Redis flush operations with progress tracking."""

    def __init__(self, client, formatter: TerminalFormatter):
        """
        Initialize flush executor.

        Args:
            client: Redis client instance
            formatter: Terminal output formatter
        """
        self.client = client
        self.formatter = formatter

    def flush_full_database(self) -> FlushOperationStats:
        """
        Flush entire Redis database.

        Returns:
            Operation statistics
        """
        self.formatter.print_info("Executing full database flush (FLUSHDB)...")

        start_time = time.time()
        keys_before = self.client.dbsize()

        try:
            self.client.flushdb()
            duration = time.time() - start_time
            keys_after = self.client.dbsize()
            keys_deleted = keys_before - keys_after

            stats = FlushOperationStats(
                strategy=FlushStrategy.FULL_DATABASE,
                prefix_pattern=None,
                total_keys_before=keys_before,
                total_keys_after=keys_after,
                keys_deleted=keys_deleted,
                batches_processed=1,
                duration_seconds=duration,
                keys_per_second=keys_deleted / duration if duration > 0 else 0,
            )

            self.formatter.print_success(
                f"Database flushed: {keys_deleted:,} keys deleted"
            )
            return stats

        except Exception as e:
            self.formatter.print_error(f"Flush failed: {e}")
            raise

    def flush_by_prefix_pattern(self, prefix: str) -> FlushOperationStats:
        """
        Delete all keys matching a prefix pattern.

        Args:
            prefix: Key prefix to match

        Returns:
            Operation statistics
        """
        pattern = f"{prefix}*"
        self.formatter.print_info(f"Deleting keys matching pattern: {pattern}")

        start_time = time.time()
        keys_before = self.client.dbsize()

        total_keys_deleted = 0
        batch_count = 0
        current_batch = []

        try:
            # Use SCAN to safely iterate through keys
            for key in self.client.scan_iter(
                match=pattern, count=OperationConfig.SCAN_BATCH_SIZE
            ):
                current_batch.append(key)

                # Delete when batch is full
                if len(current_batch) >= OperationConfig.DELETE_BATCH_SIZE:
                    deleted_count = self.client.delete(*current_batch)
                    total_keys_deleted += deleted_count
                    batch_count += 1
                    current_batch = []

                    # Show progress periodically
                    if batch_count % OperationConfig.PROGRESS_UPDATE_INTERVAL == 0:
                        self.formatter.print_info(
                            f"Progress: {total_keys_deleted:,} keys deleted "
                            f"({batch_count} batches processed)"
                        )

            # Delete remaining keys in final batch
            if current_batch:
                deleted_count = self.client.delete(*current_batch)
                total_keys_deleted += deleted_count
                batch_count += 1

            duration = time.time() - start_time
            keys_after = self.client.dbsize()

            stats = FlushOperationStats(
                strategy=FlushStrategy.PREFIX_PATTERN,
                prefix_pattern=pattern,
                total_keys_before=keys_before,
                total_keys_after=keys_after,
                keys_deleted=total_keys_deleted,
                batches_processed=batch_count,
                duration_seconds=duration,
                keys_per_second=total_keys_deleted / duration if duration > 0 else 0,
            )

            self.formatter.print_success(
                f"Pattern flush complete: {total_keys_deleted:,} keys deleted "
                f"in {batch_count} batches"
            )
            return stats

        except Exception as e:
            self.formatter.print_error(f"Pattern flush failed: {e}")
            raise


# ============================================================================
# Configuration Loader
# ============================================================================


class ConfigurationLoader:
    """Loads configuration from environment variables and .env file."""

    def __init__(self, formatter: TerminalFormatter):
        """
        Initialize configuration loader.

        Args:
            formatter: Terminal output formatter
        """
        self.formatter = formatter

    def load_environment_file(self) -> bool:
        """
        Load environment variables from .env file.

        Returns:
            True if .env file was loaded, False otherwise
        """
        # Find project root (parent of infrastructure directory)
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        env_file_path = project_root / ".env"

        if not env_file_path.exists():
            self.formatter.print_warning(f".env file not found at {env_file_path}")
            return False

        try:
            from dotenv import load_dotenv

            load_dotenv(env_file_path)
            self.formatter.print_success(f"Loaded environment from {env_file_path}")
            return True
        except ImportError:
            self.formatter.print_warning(
                "python-dotenv not installed. Install with: pip install python-dotenv"
            )
            return False
        except Exception as e:
            self.formatter.print_warning(f"Error loading .env file: {e}")
            return False

    def get_redis_config_from_environment(
        self, cli_args: argparse.Namespace
    ) -> Dict[str, Any]:
        """
        Build Redis configuration from environment and CLI arguments.

        Args:
            cli_args: Parsed command-line arguments

        Returns:
            Dictionary with Redis configuration
        """
        return {
            "host": cli_args.host or os.getenv("REDIS_HOST", RedisDefaults.HOST),
            "port": cli_args.port or int(os.getenv("REDIS_PORT", RedisDefaults.PORT)),
            "db": cli_args.db or int(os.getenv("REDIS_DB", RedisDefaults.DB)),
            "password": (
                cli_args.password or os.getenv("REDIS_PASSWORD", RedisDefaults.PASSWORD)
            ),
            "socket_timeout": RedisDefaults.SOCKET_TIMEOUT,
            "socket_connect_timeout": RedisDefaults.SOCKET_CONNECT_TIMEOUT,
            "key_prefix": (
                cli_args.prefix
                or os.getenv("REDIS_KEY_PREFIX", RedisDefaults.KEY_PREFIX)
            ),
        }


# ============================================================================
# Main Flush Orchestrator
# ============================================================================


class RedisFlushOrchestrator:
    """Orchestrates the complete Redis flush operation."""

    def __init__(self, config: Dict[str, Any], formatter: TerminalFormatter):
        """
        Initialize orchestrator.

        Args:
            config: Redis configuration dictionary
            formatter: Terminal output formatter
        """
        self.config = config
        self.formatter = formatter
        self.connection_manager = None
        self.key_inspector = None
        self.flush_executor = None

    def execute(
        self,
        strategy: FlushStrategy,
        prefix: Optional[str] = None,
        require_confirmation: bool = True,
        dry_run: bool = False,
    ) -> bool:
        """
        Execute the flush operation.

        Args:
            strategy: Flush strategy to use
            prefix: Key prefix for pattern-based flush
            require_confirmation: Whether to ask for user confirmation
            dry_run: If True, preview only without deleting

        Returns:
            True if successful, False otherwise
        """
        try:
            # Display main header
            self.formatter.print_header("REDIS CACHE FLUSH UTILITY")

            if dry_run:
                self.formatter.print_warning("DRY RUN MODE - No keys will be deleted")

            # Initialize connection
            if not self._initialize_connection():
                return False

            # Display connection info
            self._display_connection_information()

            # Preview operation
            keys_to_affect = self._preview_operation(strategy, prefix)

            if keys_to_affect == 0:
                self.formatter.print_warning("No keys found to delete")
                return True

            # Confirmation (unless dry run or confirmation disabled)
            if not dry_run and require_confirmation:
                if not self._confirm_operation(strategy, keys_to_affect):
                    self.formatter.print_info("Operation cancelled by user")
                    return False

            # Execute flush (skip if dry run)
            if dry_run:
                self.formatter.print_success("Dry run complete - no changes made")
                return True

            stats = self._execute_flush_operation(strategy, prefix)

            # Display results
            self._display_operation_results(stats)

            return True

        except KeyboardInterrupt:
            self.formatter.print_warning("\nOperation interrupted by user")
            return False
        except Exception as e:
            self.formatter.print_error(f"Operation failed: {e}")
            return False
        finally:
            self._cleanup()

    def _initialize_connection(self) -> bool:
        """Initialize Redis connection."""
        self.connection_manager = RedisConnectionManager(
            host=self.config["host"],
            port=self.config["port"],
            db=self.config["db"],
            password=self.config["password"],
            socket_timeout=self.config["socket_timeout"],
            socket_connect_timeout=self.config["socket_connect_timeout"],
            formatter=self.formatter,
        )

        if not self.connection_manager.connect(
            max_retries=OperationConfig.CONNECTION_RETRY_ATTEMPTS,
            retry_delay=OperationConfig.CONNECTION_RETRY_DELAY,
        ):
            return False

        # Initialize helper components
        self.key_inspector = RedisKeyInspector(
            self.connection_manager.client, self.formatter
        )
        self.flush_executor = RedisFlushExecutor(
            self.connection_manager.client, self.formatter
        )

        return True

    def _display_connection_information(self) -> None:
        """Display Redis connection information."""
        self.formatter.print_subheader("Connection Information")

        # Configuration details
        self.formatter.print_detail("Host", self.config["host"])
        self.formatter.print_detail("Port", self.config["port"])
        self.formatter.print_detail("Database", self.config["db"])
        self.formatter.print_detail(
            "Authentication", "Enabled" if self.config["password"] else "Disabled"
        )

        # Server information
        server_info = self.connection_manager.get_server_info()
        if server_info:
            print()
            self.formatter.print_detail(
                "Redis Version", server_info.get("redis_version")
            )
            self.formatter.print_detail(
                "Memory Used", server_info.get("used_memory_human")
            )
            self.formatter.print_detail(
                "Connected Clients", server_info.get("connected_clients")
            )
            self.formatter.print_detail(
                "Uptime (days)", server_info.get("uptime_in_days")
            )

        # Database size
        db_size = self.connection_manager.get_database_size()
        print()
        self.formatter.print_detail("Total Keys in Database", f"{db_size:,}")

    def _preview_operation(self, strategy: FlushStrategy, prefix: Optional[str]) -> int:
        """
        Preview the operation and return number of keys to be affected.

        Args:
            strategy: Flush strategy
            prefix: Key prefix for pattern-based flush

        Returns:
            Number of keys that will be affected
        """
        self.formatter.print_subheader("Operation Preview")

        if strategy == FlushStrategy.FULL_DATABASE:
            self.formatter.print_detail(
                "Strategy", "Full Database Flush (FLUSHDB)", TerminalColors.YELLOW
            )
            keys_affected = self.connection_manager.get_database_size()
            self.formatter.print_detail(
                "Keys to Delete", f"{keys_affected:,}", TerminalColors.YELLOW
            )
        else:
            pattern = f"{prefix}*"
            self.formatter.print_detail(
                "Strategy", "Prefix Pattern Deletion", TerminalColors.YELLOW
            )
            self.formatter.print_detail("Pattern", pattern, TerminalColors.YELLOW)

            # Count matching keys
            self.formatter.print_info("Scanning for matching keys...")
            keys_affected = self.key_inspector.count_keys_matching_pattern(pattern)
            self.formatter.print_detail(
                "Keys to Delete", f"{keys_affected:,}", TerminalColors.YELLOW
            )

            # Show sample keys
            if keys_affected > 0:
                samples = self.key_inspector.get_sample_keys(pattern)
                if samples:
                    print()
                    self.formatter.print_info(
                        f"Sample keys matching pattern (showing {len(samples)} of {keys_affected:,}):"
                    )
                    for sample in samples[:5]:
                        print(f"    • {sample}")
                    if len(samples) > 5:
                        print(f"    ... and {len(samples) - 5} more samples")
            else:
                # No keys matched - show all keys to help user find the right pattern
                print()
                self.formatter.print_warning(
                    f"No keys found matching pattern '{pattern}'"
                )
                total_keys = self.connection_manager.get_database_size()
                if total_keys > 0:
                    print()
                    self.formatter.print_info(
                        f"Showing all {total_keys} keys in database to help you find the correct pattern:"
                    )
                    all_keys = self.key_inspector.get_all_keys_sample(max_samples=20)
                    for key in all_keys:
                        print(f"    • {key}")
                    if total_keys > len(all_keys):
                        print(f"    ... and {total_keys - len(all_keys)} more keys")
                    print()
                    self.formatter.print_info(
                        "Tip: Use --prefix 'your_prefix' to specify a custom pattern, "
                        "or --full to delete all keys"
                    )

        return keys_affected

    def _confirm_operation(self, strategy: FlushStrategy, keys_count: int) -> bool:
        """
        Ask user to confirm the operation.

        Args:
            strategy: Flush strategy
            keys_count: Number of keys to be affected

        Returns:
            True if user confirms, False otherwise
        """
        self.formatter.print_subheader("Confirmation Required")

        # Display warning
        print(
            self.formatter._colorize(
                "⚠️  WARNING: This operation will permanently delete data!",
                TerminalColors.RED,
                bold=True,
            )
        )
        print(
            self.formatter._colorize(
                f"⚠️  {keys_count:,} keys will be deleted and cannot be recovered!",
                TerminalColors.RED,
                bold=True,
            )
        )
        print()

        # Get confirmation
        response = (
            input(
                self.formatter._colorize(
                    "Type 'yes' to proceed or anything else to cancel: ",
                    TerminalColors.YELLOW,
                    bold=True,
                )
            )
            .strip()
            .lower()
        )

        return response in ("yes", "y")

    def _execute_flush_operation(
        self, strategy: FlushStrategy, prefix: Optional[str]
    ) -> FlushOperationStats:
        """
        Execute the flush operation.

        Args:
            strategy: Flush strategy
            prefix: Key prefix for pattern-based flush

        Returns:
            Operation statistics
        """
        self.formatter.print_subheader("Executing Flush Operation")

        if strategy == FlushStrategy.FULL_DATABASE:
            return self.flush_executor.flush_full_database()
        else:
            return self.flush_executor.flush_by_prefix_pattern(prefix)

    def _display_operation_results(self, stats: FlushOperationStats) -> None:
        """Display operation results and statistics."""
        self.formatter.print_subheader("Operation Results")

        self.formatter.print_detail("Strategy", stats.strategy.value)
        if stats.prefix_pattern:
            self.formatter.print_detail("Pattern", stats.prefix_pattern)

        print()
        self.formatter.print_detail(
            "Keys Before", f"{stats.total_keys_before:,}", TerminalColors.CYAN
        )
        self.formatter.print_detail(
            "Keys Deleted", f"{stats.keys_deleted:,}", TerminalColors.GREEN
        )
        self.formatter.print_detail(
            "Keys After", f"{stats.total_keys_after:,}", TerminalColors.CYAN
        )

        print()
        self.formatter.print_detail(
            "Duration", f"{stats.duration_seconds:.2f} seconds", TerminalColors.MAGENTA
        )
        self.formatter.print_detail(
            "Throughput",
            f"{stats.keys_per_second:.0f} keys/second",
            TerminalColors.MAGENTA,
        )
        self.formatter.print_detail(
            "Batches Processed", f"{stats.batches_processed:,}", TerminalColors.MAGENTA
        )

        print()
        self.formatter.print_success("✓ Flush operation completed successfully!")

    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.connection_manager:
            self.connection_manager.close()


# ============================================================================
# Command-Line Interface
# ============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Redis Cache Flush Utility - Production Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview deletion with default prefix (dry run)
  python infrastructure/flush_redis.py --dry-run

  # Flush keys with default prefix
  python infrastructure/flush_redis.py

  # Flush entire Redis database
  python infrastructure/flush_redis.py --full

  # Flush keys with custom prefix
  python infrastructure/flush_redis.py --prefix "my_prefix"

  # Skip confirmation prompt (dangerous!)
  python infrastructure/flush_redis.py --no-confirm

  # Custom Redis connection
  python infrastructure/flush_redis.py --host 192.168.1.100 --port 6380

Configuration Priority:
  1. Command-line arguments (highest priority)
  2. Environment variables
  3. .env file
  4. Default values (lowest priority)

Environment Variables:
  REDIS_HOST          - Redis server host
  REDIS_PORT          - Redis server port
  REDIS_DB            - Redis database index
  REDIS_PASSWORD      - Redis authentication password
  REDIS_KEY_PREFIX    - Default key prefix for pattern matching

Safety Features:
  - Confirmation prompt before deletion (use --no-confirm to skip)
  - Dry run mode to preview operations (--dry-run)
  - Connection retry logic with timeout
  - Batch processing for large key sets
  - Detailed operation statistics and progress tracking

Warning:
  The --full flag will delete ALL keys in the selected database.
  Use with extreme caution in production environments!
        """,
    )

    # Operation mode
    mode_group = parser.add_argument_group("Operation Mode")
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Flush entire database (FLUSHDB). WARNING: Deletes ALL keys!",
    )
    mode_group.add_argument(
        "--prefix",
        type=str,
        default=None,
        metavar="PREFIX",
        help=f"Key prefix for pattern matching (default: {RedisDefaults.KEY_PREFIX})",
    )
    mode_group.add_argument(
        "--dry-run", action="store_true", help="Preview operation without deleting keys"
    )

    # Connection settings
    conn_group = parser.add_argument_group("Redis Connection")
    conn_group.add_argument(
        "--host",
        type=str,
        default=None,
        metavar="HOST",
        help=f"Redis server host (default: {RedisDefaults.HOST})",
    )
    conn_group.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"Redis server port (default: {RedisDefaults.PORT})",
    )
    conn_group.add_argument(
        "--db",
        type=int,
        default=None,
        metavar="DB",
        help=f"Redis database index (default: {RedisDefaults.DB})",
    )
    conn_group.add_argument(
        "--password",
        type=str,
        default=None,
        metavar="PASSWORD",
        help="Redis password (default: from environment or redis123)",
    )

    # Behavior settings
    behavior_group = parser.add_argument_group("Behavior Options")
    behavior_group.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)",
    )
    behavior_group.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    behavior_group.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> Tuple[bool, Optional[str]]:
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Cannot use both --full and --prefix
    if args.full and args.prefix:
        return False, "Cannot use both --full and --prefix options"

    # Port validation
    if args.port is not None and (args.port < 1 or args.port > 65535):
        return False, f"Invalid port number: {args.port} (must be 1-65535)"

    # DB index validation
    if args.db is not None and (args.db < 0 or args.db > 15):
        return False, f"Invalid database index: {args.db} (must be 0-15)"

    return True, None


def main() -> int:
    """
    Main entry point for the Redis flush utility.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize formatter
    formatter = TerminalFormatter(use_colors=not args.no_color)

    # Validate arguments
    is_valid, error_message = validate_arguments(args)
    if not is_valid:
        formatter.print_error(f"Invalid arguments: {error_message}")
        parser.print_help()
        return 1

    try:
        # Load configuration
        config_loader = ConfigurationLoader(formatter)
        config_loader.load_environment_file()

        redis_config = config_loader.get_redis_config_from_environment(args)

        # Determine flush strategy
        if args.full:
            strategy = FlushStrategy.FULL_DATABASE
            prefix = None
        else:
            strategy = FlushStrategy.PREFIX_PATTERN
            prefix = redis_config["key_prefix"]

        # Create orchestrator
        orchestrator = RedisFlushOrchestrator(redis_config, formatter)

        # Execute operation
        success = orchestrator.execute(
            strategy=strategy,
            prefix=prefix,
            require_confirmation=not args.no_confirm,
            dry_run=args.dry_run,
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        formatter.print_warning("\n\nOperation interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        formatter.print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
