#!/usr/bin/env python3
"""
RxNorm Data Extraction Pipeline

OBJECTIVE:
This pipeline extracts comprehensive drug classification data from the RxNorm API
(National Library of Medicine) using the Anatomical Therapeutic Chemical (ATC)
classification system. The pipeline traverses the ATC hierarchy (Levels 1-4) and
exports drug ingredient mappings with their therapeutic classifications.

OUTPUT STRUCTURE:
- ingredient_rxcui: RxNorm Concept Unique Identifier for the drug ingredient
- primary_drug_name: Standard name of the drug ingredient
- therapeutic_class_l2: ATC Level 2 - Therapeutic subgroup
- drug_class_l3: ATC Level 3 - Pharmacological subgroup
- drug_subclass_l4: ATC Level 4 - Chemical subgroup

ATC CLASSIFICATION LEVELS:
- L1: Anatomical main group (single letter: A-V)
- L2: Therapeutic subgroup (e.g., A01)
- L3: Pharmacological subgroup (e.g., A01A)
- L4: Chemical subgroup (e.g., A01AA)

USAGE:
    python rxnorm_data_extraction_pipeline.py --output drug_classifications.csv
    python rxnorm_data_extraction_pipeline.py --roots A,B,C --delay 1.0
    python rxnorm_data_extraction_pipeline.py --smoke-test

REQUIREMENTS:
    - httpx: HTTP client library
    - tqdm: Progress bar library

API REFERENCE:
    RxNav REST API: https://rxnav.nlm.nih.gov/REST/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================


def ensure_package_installed(package_name: str) -> bool:
    """
    Verify package availability and install if missing.

    Args:
        package_name: Name of the Python package to check/install

    Returns:
        True if package is available, False if installation failed
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"Installing required package: {package_name}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(
                f"Failed to install {package_name}. Please install manually: pip install {package_name}"
            )
            return False


def initialize_dependencies() -> None:
    """
    Initialize all required dependencies before script execution.

    Raises:
        SystemExit: If critical dependencies cannot be installed
    """
    required_packages = ["httpx", "tqdm"]
    for package_name in required_packages:
        if not ensure_package_installed(package_name):
            print(f"Critical error: Cannot proceed without {package_name}")
            sys.exit(1)


# Initialize dependencies before importing them
initialize_dependencies()

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ATC code format patterns
ATC_LEVEL_2_PATTERN = re.compile(r"^[A-Z]\d{2}$")  # Example: A01
ATC_LEVEL_3_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]$")  # Example: A01A
ATC_LEVEL_4_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]{2}$")  # Example: A01AA

# All valid ATC Level 1 root categories
DEFAULT_ATC_ROOTS = [
    "A",
    "B",
    "C",
    "D",
    "G",
    "H",
    "J",
    "L",
    "M",
    "N",
    "P",
    "R",
    "S",
    "V",
]

# CSV output schema
OUTPUT_COLUMNS = [
    "ingredient_rxcui",
    "primary_drug_name",
    "therapeutic_class_l2",
    "drug_class_l3",
    "drug_subclass_l4",
]

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


def configure_logging() -> logging.Logger:
    """
    Configure logging to file with timestamp.

    Returns:
        Configured logger instance
    """
    log_filename = f'rxnorm_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename, encoding="utf-8")],
    )

    return logging.getLogger(__name__)


logger = configure_logging()

# ============================================================================
# HTTP CLIENT
# ============================================================================


class RxNavHttpClient:
    """
    HTTP client for RxNav API with retry logic and rate limiting.

    Attributes:
        request_delay: Seconds to wait between requests (rate limiting)
        request_timeout: Maximum seconds to wait for response
        max_retries: Maximum number of retry attempts
        backoff_multiplier: Exponential backoff multiplier for retries
    """

    def __init__(
        self,
        request_delay: float = 0.9,
        request_timeout: float = 40.0,
        max_retries: int = 3,
        backoff_multiplier: float = 1.8,
    ):
        """
        Initialize HTTP client with configurable parameters.

        Args:
            request_delay: Delay between requests in seconds
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_multiplier: Exponential backoff factor
        """
        self.request_delay = request_delay
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier

        self.client = httpx.Client(timeout=request_timeout, headers=HTTP_HEADERS)

    def fetch_json(
        self, endpoint_path: str, query_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch JSON data from RxNav API with retry logic.

        Args:
            endpoint_path: API endpoint path (e.g., "/rxclass/classTree.json")
            query_parameters: Optional query parameters dictionary

        Returns:
            JSON response as dictionary

        Raises:
            RuntimeError: If all retry attempts fail
        """
        url = f"{RXNAV_BASE_URL}{endpoint_path}"
        last_error = None

        for retry_attempt in range(self.max_retries):
            try:
                response = self.client.get(url, params=query_parameters or {})
                response.raise_for_status()
                data = response.json()

                if self.request_delay:
                    time.sleep(self.request_delay)

                return data

            except httpx.HTTPError as error:
                last_error = error
                backoff_delay = (self.backoff_multiplier**retry_attempt) * 0.4
                time.sleep(backoff_delay)

        # All retries failed
        status_code = getattr(
            getattr(last_error, "response", None), "status_code", "unknown"
        )
        error_body = ""

        try:
            error_body = last_error.response.text[:500]
        except Exception:
            pass

        error_message = (
            f"HTTP request failed after {self.max_retries} retries:\n"
            f"URL: {url}\n"
            f"Parameters: {query_parameters}\n"
            f"Status Code: {status_code}\n"
            f"Response: {error_body}"
        )

        raise RuntimeError(error_message) from last_error

    def try_fetch_json(
        self, endpoint_path: str, query_parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to fetch JSON data without raising exceptions.

        Args:
            endpoint_path: API endpoint path
            query_parameters: Query parameters dictionary

        Returns:
            JSON response dictionary or None if request fails
        """
        url = f"{RXNAV_BASE_URL}{endpoint_path}"

        try:
            response = self.client.get(url, params=query_parameters)
            response.raise_for_status()
            data = response.json()

            if self.request_delay:
                time.sleep(self.request_delay)

            return data

        except httpx.HTTPError:
            return None

    def close(self) -> None:
        """Close the HTTP client connection."""
        try:
            self.client.close()
        except Exception:
            pass


# ============================================================================
# ATC TREE NAVIGATION
# ============================================================================


def fetch_atc_classification_tree(
    http_client: RxNavHttpClient, root_category: str
) -> Dict[str, Any]:
    """
    Fetch the complete ATC classification tree for a root category.

    Args:
        http_client: Configured HTTP client instance
        root_category: ATC Level 1 root category (A-V)

    Returns:
        Complete classification tree as nested dictionary
    """
    return http_client.fetch_json(
        "/rxclass/classTree.json", {"classId": root_category, "classType": "ATC1-4"}
    )


def extract_child_nodes(parent_node: Any) -> List[Any]:
    """
    Extract child nodes from a parent node in the classification tree.

    Args:
        parent_node: Parent node dictionary

    Returns:
        List of child node dictionaries
    """
    if not isinstance(parent_node, dict):
        return []

    children = parent_node.get("rxclassTree")

    if not children:
        return []

    return children if isinstance(children, list) else [children]


def extract_node_classification(node: Any) -> Optional[Dict[str, str]]:
    """
    Extract classification information from a tree node.

    Args:
        node: Tree node dictionary

    Returns:
        Dictionary with 'classId' and 'className' or None if invalid
    """
    if not isinstance(node, dict):
        return None

    item = node.get("rxclassMinConceptItem")

    if isinstance(item, dict) and item.get("classId"):
        return {"classId": item["classId"], "className": item.get("className", "")}

    return None


def update_atc_hierarchy(
    atc_code: str,
    atc_name: str,
    current_level_2: Optional[str],
    current_level_3: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Update ATC hierarchy levels based on current code.

    Args:
        atc_code: Current ATC code being processed
        atc_name: Name of the current ATC classification
        current_level_2: Current Level 2 classification
        current_level_3: Current Level 3 classification

    Returns:
        Tuple of (updated_level_2, updated_level_3)
    """
    if ATC_LEVEL_2_PATTERN.match(atc_code):
        return (atc_name or atc_code), None

    if ATC_LEVEL_3_PATTERN.match(atc_code):
        return current_level_2, (atc_name or atc_code)

    return current_level_2, current_level_3


# ============================================================================
# CLASS MEMBERS EXTRACTION
# ============================================================================


def parse_class_members_response(api_response: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse class members from API response handling multiple response formats.

    The RxNav API has multiple response formats across versions. This function
    attempts to parse all known formats to ensure robustness.

    Args:
        api_response: Raw API response dictionary

    Returns:
        List of member dictionaries with 'rxcui', 'name', and 'tty' keys
    """
    members: List[Dict[str, str]] = []

    if not isinstance(api_response, dict):
        return members

    # Format 1: drugMemberGroup.drugMember (current API format)
    drug_member_group = api_response.get("drugMemberGroup") or {}
    drug_members = drug_member_group.get("drugMember", []) or []

    for drug_member in drug_members:
        if not isinstance(drug_member, dict):
            continue

        # Check for nested minConcept structure
        min_concept = drug_member.get("minConcept")
        if (
            isinstance(min_concept, dict)
            and min_concept.get("rxcui")
            and min_concept.get("name")
        ):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )
            continue

        # Check for direct structure
        if drug_member.get("rxcui") and drug_member.get("name"):
            members.append(
                {
                    "rxcui": str(drug_member["rxcui"]),
                    "name": drug_member["name"],
                    "tty": drug_member.get("tty", ""),
                }
            )

    if members:
        return members

    # Format 2: rxclassDrugInfoList.rxclassDrugInfo (legacy format)
    drug_info_list = api_response.get("rxclassDrugInfoList") or {}
    drug_infos = drug_info_list.get("rxclassDrugInfo", []) or []

    for drug_info in drug_infos:
        if not isinstance(drug_info, dict):
            continue

        # Check drugMember
        drug_member = drug_info.get("drugMember")
        if (
            isinstance(drug_member, dict)
            and drug_member.get("rxcui")
            and drug_member.get("name")
        ):
            members.append(
                {
                    "rxcui": str(drug_member["rxcui"]),
                    "name": drug_member["name"],
                    "tty": drug_member.get("tty", ""),
                }
            )
            continue

        # Check various minConcept fields
        min_concept = (
            drug_info.get("minConcept")
            or drug_info.get("rxclassMinConcept")
            or drug_info.get("minConceptItem")
        )

        if (
            isinstance(min_concept, dict)
            and min_concept.get("rxcui")
            and min_concept.get("name")
        ):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )

    if members:
        return members

    # Format 3: rxclassMinConceptList.rxclassMinConcept
    min_concept_list = api_response.get("rxclassMinConceptList") or {}
    min_concepts = min_concept_list.get("rxclassMinConcept", []) or []

    for min_concept in min_concepts:
        if not isinstance(min_concept, dict):
            continue

        if min_concept.get("rxcui") and min_concept.get("name"):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )

    return members


def fetch_ingredient_members(
    http_client: RxNavHttpClient, atc_level_4_code: str, enable_debug: bool = False
) -> List[Dict[str, str]]:
    """
    Fetch ingredient members for an ATC Level 4 classification.

    Attempts multiple query strategies to retrieve ingredients:
    1. Direct ingredient query (TTY=IN with ATC source)
    2. General member query (all types with ATC source)

    Args:
        http_client: Configured HTTP client
        atc_level_4_code: ATC Level 4 code (e.g., "A01AA")
        enable_debug: If True, write debug information for empty results

    Returns:
        List of ingredient member dictionaries
    """
    query_attempts: List[Tuple[Dict[str, Any], Optional[int], int]] = []

    # Strategy 1: Request only ingredients (TTY=IN)
    query_params_ingredient = {
        "classId": atc_level_4_code,
        "relaSource": "ATC",
        "ttys": "IN",
    }

    # Strategy 2: Request all members, filter ingredients later
    query_params_all = {"classId": atc_level_4_code, "relaSource": "ATC"}

    for query_parameters in [query_params_ingredient, query_params_all]:
        api_response = http_client.try_fetch_json(
            "/rxclass/classMembers.json", query_parameters
        )

        members = parse_class_members_response(api_response) if api_response else []

        # Determine error code if available
        error_code = None
        if api_response is None:
            error_code = None
        elif isinstance(api_response, dict) and "error" in api_response:
            error_code = api_response["error"].get("code")

        query_attempts.append((query_parameters, error_code, len(members)))

        if members:
            # If we didn't specifically request ingredients, filter them now
            if "ttys" not in query_parameters:
                ingredient_members = [m for m in members if m.get("tty") == "IN"]
                if ingredient_members:
                    return ingredient_members

            return members

    # All strategies failed - optionally write debug information
    if enable_debug:
        last_params, _, _ = query_attempts[-1]
        debug_data = {
            "atc_code": atc_level_4_code,
            "query_attempts": query_attempts,
            "last_parameters": last_params,
        }

        debug_filename = f"debug_classMembers_{atc_level_4_code}.json"
        with open(debug_filename, "w", encoding="utf-8") as debug_file:
            json.dump(debug_data, debug_file, ensure_ascii=False, indent=2)

    return []


# ============================================================================
# TREE TRAVERSAL AND DATA EXPORT
# ============================================================================


def traverse_and_export_tree(
    http_client: RxNavHttpClient,
    classification_tree: Dict[str, Any],
    csv_writer: csv.DictWriter,
    processed_entries: set,
    row_counter: Dict[str, int],
    enable_debug: bool,
) -> None:
    """
    Traverse ATC classification tree and export ingredient mappings.

    Performs depth-first traversal of the classification tree, extracting
    ingredient-to-classification mappings at ATC Level 4.

    Args:
        http_client: Configured HTTP client
        classification_tree: Root of classification tree
        csv_writer: CSV writer for output
        processed_entries: Set of processed (rxcui, atc_code) tuples
        row_counter: Dictionary tracking row count
        enable_debug: Enable debug output for troubleshooting
    """

    def depth_first_search(
        current_node: Dict[str, Any],
        level_2_classification: Optional[str],
        level_3_classification: Optional[str],
    ) -> None:
        """
        Recursively traverse classification tree nodes.

        Args:
            current_node: Current tree node
            level_2_classification: Current ATC Level 2 name
            level_3_classification: Current ATC Level 3 name
        """
        classification = extract_node_classification(current_node)

        if classification:
            atc_code = classification["classId"]
            atc_name = classification["className"]

            # Update hierarchy context
            updated_level_2, updated_level_3 = update_atc_hierarchy(
                atc_code, atc_name, level_2_classification, level_3_classification
            )

            # Process Level 4 classifications (leaf nodes with ingredients)
            if ATC_LEVEL_4_PATTERN.match(atc_code):
                try:
                    ingredient_members = fetch_ingredient_members(
                        http_client, atc_code, enable_debug=enable_debug
                    )

                    for member in ingredient_members:
                        entry_key = (member["rxcui"], atc_code)

                        # Skip duplicates
                        if entry_key in processed_entries:
                            continue

                        processed_entries.add(entry_key)

                        csv_writer.writerow(
                            {
                                "ingredient_rxcui": member["rxcui"],
                                "primary_drug_name": member["name"],
                                "therapeutic_class_l2": updated_level_2 or "",
                                "drug_class_l3": updated_level_3 or "",
                                "drug_subclass_l4": atc_name or atc_code,
                            }
                        )

                        row_counter["rows"] += 1

                except Exception as error:
                    logger.error(f"Error processing ATC code {atc_code}: {error}")

            # Continue traversing children with updated context
            for child_node in extract_child_nodes(current_node):
                depth_first_search(child_node, updated_level_2, updated_level_3)
        else:
            # Node has no classification data, traverse children with current context
            for child_node in extract_child_nodes(current_node):
                depth_first_search(
                    child_node, level_2_classification, level_3_classification
                )

    # Start traversal from root(s)
    root_nodes = classification_tree.get("rxclassTree")

    if isinstance(root_nodes, list):
        for root_node in root_nodes:
            depth_first_search(root_node, None, None)
    elif isinstance(root_nodes, dict):
        depth_first_search(root_nodes, None, None)


def export_drug_classifications(
    output_filepath: str,
    request_delay: float,
    atc_root_categories: List[str],
    enable_debug: bool,
) -> None:
    """
    Main export function - orchestrates complete data extraction pipeline.

    Args:
        output_filepath: Path to output CSV file
        request_delay: Delay between API requests
        atc_root_categories: List of ATC Level 1 root categories to process
        enable_debug: Enable debug output

    Raises:
        Exception: If critical error occurs during export
    """
    http_client = RxNavHttpClient(request_delay=request_delay)
    start_time = time.time()

    try:
        processed_entries: set = set()
        row_counter = {"rows": 0}

        # Ensure output directory exists
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\nInitializing export pipeline...")
        print(f"Output file: {output_path.absolute()}")
        print(f"Processing {len(atc_root_categories)} root categories\n")

        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            csv_writer.writeheader()

            for index, root_category in enumerate(atc_root_categories, start=1):
                print(
                    f"Processing root category {root_category} ({index}/{len(atc_root_categories)})..."
                )

                try:
                    classification_tree = fetch_atc_classification_tree(
                        http_client, root_category
                    )

                    initial_row_count = row_counter["rows"]

                    traverse_and_export_tree(
                        http_client,
                        classification_tree,
                        csv_writer,
                        processed_entries,
                        row_counter,
                        enable_debug,
                    )

                    rows_added = row_counter["rows"] - initial_row_count
                    print(
                        f"Completed root {root_category}: {rows_added:,} rows added "
                        f"(Total: {row_counter['rows']:,})"
                    )

                    # Flush to ensure data is written
                    output_file.flush()

                except Exception as error:
                    print(f"Failed to process root {root_category}: {error}")
                    logger.error(f"Failed to process root {root_category}: {error}")
                    continue

        total_duration = time.time() - start_time

        print(f"\n{'='*70}")
        print("Export completed successfully")
        print(f"{'='*70}")
        print(f"Total rows exported: {row_counter['rows']:,}")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Output file: {output_path.absolute()}")
        print(f"{'='*70}\n")

    except Exception as error:
        print(f"\nCritical error during export: {error}")
        logger.error(f"Critical error during export: {error}")
        raise

    finally:
        http_client.close()


# ============================================================================
# SMOKE TEST
# ============================================================================


def run_smoke_test(request_delay: float, enable_debug: bool) -> None:
    """
    Execute quick validation test on known ATC codes.

    Verifies that the API is accessible and returns expected data format.

    Args:
        request_delay: Delay between API requests
        enable_debug: Enable debug output
    """
    print("\nRunning smoke test to verify API connectivity...\n")

    http_client = RxNavHttpClient(request_delay=request_delay)
    test_codes = ["C07AB", "J01CA", "A10BA"]

    try:
        for atc_code in test_codes:
            try:
                members = fetch_ingredient_members(
                    http_client, atc_code, enable_debug=enable_debug
                )
                print(
                    f"ATC Level 4 code '{atc_code}': {len(members)} ingredient(s) found"
                )

                if members:
                    print(f"Sample ingredients from {atc_code}:")
                    for member in members[:5]:
                        print(
                            f"  - RXCUI: {member['rxcui']}, Name: {member['name']}, TTY: {member['tty']}"
                        )
                    print()
                    break

            except Exception as error:
                print(f"Smoke test failed for code {atc_code}: {error}")

    finally:
        http_client.close()

    print("Smoke test completed.\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="RxNorm Data Extraction Pipeline - Extract drug classifications from RxNav API",
        epilog="Example: python rxnorm_data_extraction_pipeline.py --output drug_classifications.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output",
        default="rxnorm_complete_drug_classification.csv",
        help="Output CSV filename (default: rxnorm_complete_drug_classification.csv)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.9,
        help="Seconds to wait between HTTP requests for rate limiting (default: 0.9)",
    )

    parser.add_argument(
        "--roots",
        help="Comma-separated ATC Level 1 root categories to process "
        "(default: all roots - A,B,C,D,G,H,J,L,M,N,P,R,S,V)",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run API connectivity test and display sample data",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode - write diagnostic files for troubleshooting",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the RxNorm data extraction pipeline.

    Raises:
        SystemExit: On user interrupt or critical error
    """
    args = parse_command_line_arguments()

    # Execute smoke test if requested
    if args.smoke_test:
        run_smoke_test(args.delay, args.debug)
        sys.exit(0)

    # Determine root categories to process
    if args.roots:
        atc_root_categories = [
            category.strip().upper()
            for category in args.roots.split(",")
            if category.strip()
        ]
        print(
            f"\nStarting RxNorm export for specified categories: {', '.join(atc_root_categories)}"
        )
    else:
        atc_root_categories = DEFAULT_ATC_ROOTS
        print("\nStarting COMPLETE RxNorm export (all therapeutic categories)")

    print(f"Request delay: {args.delay} seconds")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    # Execute main export pipeline
    try:
        export_drug_classifications(
            args.output, args.delay, atc_root_categories, args.debug
        )

    except KeyboardInterrupt:
        print("\n\nExport interrupted by user")
        sys.exit(1)

    except Exception as error:
        print(f"\n\nExport failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
