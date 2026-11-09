"""Benchmark-only app factory. Normal API startup never imports this module."""

import json
import os
from pathlib import Path


def create_app():
    # The launcher sets DATABASE_URL before importing the database's singleton engine.
    from app.core.config import Settings
    from app.db.seed import init_database
    from app.main import create_app as application
    from app.services import latency

    config = json.loads(Path(os.environ["LATENCY_CONFIG"]).read_text(encoding="utf-8"))
    settings = Settings(**config["settings"])
    latency.configure(config["events_path"], config["run_id"])
    init_database()
    return application(settings)
