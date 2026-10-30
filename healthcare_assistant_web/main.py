"""Start the Healthcare Assistant web interface.

Run it from anywhere, for example:

    python healthcare_assistant_web/main.py
    python main.py                      (from inside healthcare_assistant_web)
    python -m healthcare_assistant_web.main    (from the project root)
"""

import sys
from pathlib import Path

# When this file is run directly, Python does not know about the project folder, so
# the "healthcare_assistant_web" and "src" packages could not be found. Add it.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nicegui import app, background_tasks, ui  # noqa: E402

from healthcare_assistant_web.components.page_layout import build_home_page  # noqa: E402
from healthcare_assistant_web.services.shared_services import SharedServicesManager  # noqa: E402
from healthcare_assistant_web.web_ui_settings import (  # noqa: E402
    APP_SUBTITLE,
    APP_TITLE,
    SERVER_PORT,
)

shared_services_manager = SharedServicesManager()


@ui.page("/")
def home_page() -> None:
    build_home_page(shared_services_manager)


# Connect to the drug catalog and create the guidelines store as the server
# starts, so the first visitor does not wait for them.
app.on_startup(lambda: background_tasks.create(shared_services_manager.warm_up()))
app.on_shutdown(shared_services_manager.close)


if __name__ == "__main__":
    ui.run(
        title=f"{APP_TITLE} | {APP_SUBTITLE}",
        port=SERVER_PORT,
        reload=False,
        show=False,
        favicon="🩺",
    )
