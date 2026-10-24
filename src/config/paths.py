"""Project paths and Docker volume location.

Big-picture flow:
1. Work out this project's root folder on disk, no matter where the app is
   run from.
2. Build the full path to the ".env" file (where secrets like API keys
   live) from that root.
3. Describe one extra setting: where Docker should store its persistent
   data on disk.
"""

from pathlib import Path

# Path is Python's standard way to represent a file or folder location that
# works correctly on both Windows and Linux/Mac, instead of manually joining
# strings with "/" or "\\".
from pydantic import BaseModel, Field

# Pydantic is a library for describing "what shape of data is valid" using
# plain Python classes. BaseModel is the class every settings group in this
# project inherits from. Field lets us attach a default value and a human
# description to each setting, which also shows up in error messages and
# generated docs.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# __file__ is this file's own path. ".resolve()" turns it into a full,
# absolute path (no "..", works from any starting folder). ".parents[2]"
# walks up two folder levels (config -> src -> project root), so this
# always points at the project's top folder, no matter where the app was
# started from.
ENV_FILE = PROJECT_ROOT / ".env"
# The "/" here is Path's way of joining a folder and a filename - it is not
# division. This is the file holding secrets like OPENAI_API_KEY.


# PathSettings: one small group of settings about file locations. Needed so
# every other settings group (see settings.py) can be combined with it
# without each one needing its own copy of path-related fields.
class PathSettings(BaseModel):
    docker_volume_directory: str = Field(
        default=".",
        description="Base directory for Docker Compose persistent volumes. "
        "Defaults to the project root ('.').",
    )
    # docker_volume_directory: where Docker Compose should store its
    # persistent data on disk (databases, caches, etc.), relative to
    # wherever docker-compose is run from.
