from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

from run_config_loader import ensure_backend_import_path, load_run_settings, resolve_run_config_path


REPO_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_DIR / "backend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the backend with an experiment run config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Run config name or path, e.g. cpu_model or experiments/run_configs/cpu_model.json.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload. Uses an import-string factory and re-loads the run config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_run_config_path(args.config)
    print(f"Starting backend with run config: {config_path}")

    ensure_backend_import_path()

    if args.reload:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        uvicorn.run(
            "run_backend:create_configured_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(BACKEND_DIR), str(config_path.parent)],
            app_dir=str(Path(__file__).resolve().parent),
            env_file=None,
        )
        return

    from app.main import create_app  # noqa: PLC0415

    settings = load_run_settings(str(config_path))
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port)


def create_configured_app():
    args = parse_args()
    ensure_backend_import_path()

    from app.main import create_app  # noqa: PLC0415

    return create_app(load_run_settings(args.config))


if __name__ == "__main__":
    main()
