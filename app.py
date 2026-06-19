"""Entry point for the TEAMx hackathon pipeline.

Run this file from the downloaded package. It reads ``settings.ini``, resolves
the project folders, creates missing output folders, and starts ``map.py``.
"""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS = APP_DIR / "settings.ini"
REQUIRED_PATH_KEYS = (
    "raw_dir",
    "topography_dir",
    "satellite_dir",
    "processed_dir",
    "results_dir",
)


def load_settings(settings_path: Path) -> configparser.ConfigParser:
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    config = configparser.ConfigParser()
    config.read(settings_path, encoding="utf-8")

    if "paths" not in config:
        raise KeyError("settings.ini is missing the [paths] section.")

    missing = [key for key in REQUIRED_PATH_KEYS if not config["paths"].get(key)]
    if missing:
        raise KeyError(f"settings.ini is missing required path keys: {', '.join(missing)}")

    return config


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_paths(config: configparser.ConfigParser, base_dir: Path) -> dict[str, Path]:
    return {
        key: resolve_path(config["paths"][key], base_dir)
        for key in REQUIRED_PATH_KEYS
    }


def ensure_pipeline_folders(paths: dict[str, Path]) -> None:
    for key in REQUIRED_PATH_KEYS:
        paths[key].mkdir(parents=True, exist_ok=True)


def pipeline_environment(paths: dict[str, Path], settings_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TEAMX_SETTINGS": str(settings_path),
            "TEAMX_RAW_DIR": str(paths["raw_dir"]),
            "TEAMX_TOPOGRAPHY_DIR": str(paths["topography_dir"]),
            "TEAMX_SATELLITE_DIR": str(paths["satellite_dir"]),
            "TEAMX_PROCESSED_DIR": str(paths["processed_dir"]),
            "TEAMX_RESULTS_DIR": str(paths["results_dir"]),
        }
    )
    return env


def run_map(config: configparser.ConfigParser, settings_path: Path, paths: dict[str, Path]) -> None:
    map_script_value = config.get("pipeline", "map_script", fallback="code/map.py")
    map_script = resolve_path(map_script_value, APP_DIR)
    if not map_script.exists():
        raise FileNotFoundError(
            f"map.py was not found at {map_script}. "
            "Put map.py there or change [pipeline] map_script in settings.ini."
        )

    settings_arg = config.get("pipeline", "map_settings_arg", fallback="--settings").strip()
    command = [sys.executable, str(map_script)]
    if settings_arg:
        command.extend([settings_arg, str(settings_path)])

    subprocess.run(
        command,
        cwd=APP_DIR,
        env=pipeline_environment(paths, settings_path),
        check=True,
    )


def main(settings_file: str | Path = DEFAULT_SETTINGS) -> None:
    settings_path = resolve_path(str(settings_file), APP_DIR)
    config = load_settings(settings_path)
    paths = resolve_paths(config, settings_path.parent)
    ensure_pipeline_folders(paths)
    run_map(config, settings_path, paths)


if __name__ == "__main__":
    selected_settings = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETTINGS
    main(selected_settings)
